import csv
import io
import json
import logging
import os
import random
import re
import time
from urllib.parse import urlparse

import boto3
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GoDaddy Mention Extractor",
    page_icon="🔍",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
FETCH_TIMEOUT = 20
HIT_SENTENCE_COL = "Hit Sentence"
DOMAIN_MIN_DELAY = 5.0

MELTWATER_DOMAINS = {
    "transition.meltwater.com",
    "app.meltwater.com",
    "meltwater.com",
}
MELTWATER_LOGIN_URL = "https://app.meltwater.com/"

BEDROCK_REGION = st.secrets.get("BEDROCK_REGION", os.environ.get("BEDROCK_REGION", "us-east-2"))
BEDROCK_MODEL_ID = st.secrets.get(
    "BEDROCK_MODEL_ID",
    os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_headers(url: str) -> dict:
    ua = random.choice(USER_AGENTS)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|Inc|Ltd|Corp|Co)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'\u2018\u2019\u201c\u201d\(\[])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                     "noscript", "iframe", "form"]):
        tag.decompose()
    for selector in [
        "article", "main", '[role="main"]', ".post-content", ".entry-content",
        ".article-body", ".story-body", ".article__body", ".content-body",
        "#article-body", ".post-body",
    ]:
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return text
    body = soup.find("body")
    return (body or soup).get_text(separator=" ", strip=True)


def extract_anchor_phrases(hit_sentence: str) -> list[str]:
    cleaned = re.sub(r"^[\s.…]+|[\s.…]+$", "", hit_sentence).strip()
    parts = re.split(r"\.{2,}|…", cleaned)
    return [p.strip() for p in parts if len(p.strip().split()) >= 4]


def find_best_context(sentences: list[str], anchor_phrases: list[str], keyword: str = "GoDaddy") -> str | None:
    keyword_lower = keyword.lower()
    best_idx, best_score = None, 0

    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        if keyword_lower not in sentence_lower:
            continue
        for phrase in anchor_phrases:
            phrase_words = set(re.findall(r"\w+", phrase.lower()))
            sent_words = set(re.findall(r"\w+", sentence_lower))
            overlap = len(phrase_words & sent_words)
            if overlap > best_score:
                best_score = overlap
                best_idx = i

    if best_idx is None:
        for i, sentence in enumerate(sentences):
            if keyword_lower in sentence.lower():
                best_idx = i
                break

    if best_idx is None:
        return None

    before = ""
    for j in range(best_idx - 1, -1, -1):
        if len(sentences[j].strip()) > 20:
            before = sentences[j]
            break

    mention = sentences[best_idx]

    after = ""
    for j in range(best_idx + 1, len(sentences)):
        if len(sentences[j].strip()) > 20:
            after = sentences[j]
            break

    result = ""
    if before:
        result += f"Before: {before}\n"
    result += f"Mention: {mention}\n"
    if after:
        result += f"After: {after}"
    return result.strip()


def fetch_with_retry(url: str, retries: int = 3) -> requests.Response | None:
    session = requests.Session()
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=make_headers(url), timeout=FETCH_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 429, 503) and attempt < retries - 1:
                time.sleep((attempt + 1) * 2 + random.uniform(0.5, 1.5))
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
                continue
            raise
    return None


def fetch_via_curl_cffi(url: str) -> str | None:
    try:
        from curl_cffi import requests as cffi_requests
        resp = cffi_requests.get(url, impersonate="chrome120", timeout=FETCH_TIMEOUT, allow_redirects=True)
        if resp.status_code == 200:
            return extract_article_text(resp.text)
        return None
    except Exception as exc:
        logger.warning("curl_cffi failed for %s: %s", url, exc)
        return None


def fetch_via_playwright(url: str) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-US",
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
            if "msn.com" in urlparse(url).netloc:
                try:
                    btn = page.get_by_text("Continue Reading", exact=False).first
                    if btn:
                        btn.click()
                        page.wait_for_timeout(1500)
                except Exception:
                    pass
            html = page.content()
            browser.close()
            return extract_article_text(html) if html else None
    except Exception as exc:
        logger.warning("Playwright failed for %s: %s", url, exc)
        return None


def is_meltwater_url(url: str) -> bool:
    """Return True if this URL belongs to the Meltwater platform."""
    return urlparse(url).netloc in MELTWATER_DOMAINS


def _apply_cookie_string_to_session(session: requests.Session, cookies_str: str) -> None:
    """Parse a browser-copied cookie string and add each cookie to a requests Session."""
    for chunk in cookies_str.split(";"):
        chunk = chunk.strip()
        if "=" in chunk:
            name, value = chunk.split("=", 1)
            session.cookies.set(name.strip(), value.strip(), domain=".meltwater.com")


def fetch_meltwater_with_cookies(url: str, cookies_str: str) -> str | None:
    """Fetch a Meltwater URL using a browser-copied session cookie string."""
    try:
        session = requests.Session()
        _apply_cookie_string_to_session(session, cookies_str)
        resp = session.get(url, headers=make_headers(url), timeout=FETCH_TIMEOUT, allow_redirects=True)
        if resp.status_code == 200:
            text = extract_article_text(resp.text)
            # Detect if we still landed on the login/paywall page
            if any(phrase in text.lower() for phrase in ("sign in to meltwater", "log in to meltwater", "mention not found")):
                logger.warning("Meltwater cookie auth appears expired for %s", url)
                return None
            return text
        logger.warning("Meltwater cookie fetch returned HTTP %s for %s", resp.status_code, url)
        return None
    except Exception as exc:
        logger.warning("Meltwater cookie fetch failed for %s: %s", url, exc)
        return None


def fetch_meltwater_login_and_get_cookies(email: str, password: str) -> dict | None:
    """
    Use Playwright to log in to Meltwater once and return a dict of cookies
    that can be reused for subsequent requests in this session.
    Returns None if login failed.
    """
    try:
        from playwright.sync_api import sync_playwright
        cookies_dict: dict | None = None
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-US",
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()

            # Navigate to the login page
            page.goto(MELTWATER_LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)

            # Fill email — try common selectors
            for selector in ['input[type="email"]', 'input[name="email"]', 'input[id*="email"]']:
                try:
                    page.fill(selector, email, timeout=3000)
                    break
                except Exception:
                    continue

            # Some flows show password on next screen after pressing Enter/Next
            try:
                page.get_by_role("button", name=re.compile(r"next|continue", re.IGNORECASE)).first.click(timeout=3000)
                page.wait_for_timeout(1500)
            except Exception:
                pass

            # Fill password
            for selector in ['input[type="password"]', 'input[name="password"]', 'input[id*="password"]']:
                try:
                    page.fill(selector, password, timeout=3000)
                    break
                except Exception:
                    continue

            # Submit
            try:
                page.get_by_role("button", name=re.compile(r"sign in|log in|login|submit", re.IGNORECASE)).first.click(timeout=3000)
            except Exception:
                page.keyboard.press("Enter")

            # Wait for post-login navigation
            page.wait_for_timeout(6000)

            # Check if we're past the login screen
            current_url = page.url
            if "login" in current_url.lower() or "signin" in current_url.lower():
                logger.warning("Meltwater login may have failed — still on login page: %s", current_url)
                browser.close()
                return None

            # Harvest all cookies
            raw_cookies = context.cookies()
            cookies_dict = {c["name"]: c["value"] for c in raw_cookies}
            browser.close()

        return cookies_dict if cookies_dict else None
    except Exception as exc:
        logger.warning("Meltwater Playwright login failed: %s", exc)
        return None


def fetch_meltwater_with_playwright_cookies(url: str, cookies_dict: dict) -> str | None:
    """
    Fetch a Meltwater URL using a pre-authenticated cookie dict (from a previous login).
    Uses a new Playwright context seeded with those cookies so JS-rendered content works.
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-US",
                viewport={"width": 1280, "height": 800},
            )
            # Seed cookies
            mw_cookies = [
                {"name": k, "value": v, "domain": ".meltwater.com", "path": "/"}
                for k, v in cookies_dict.items()
            ]
            context.add_cookies(mw_cookies)

            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
            html = page.content()
            browser.close()
            if not html:
                return None
            text = extract_article_text(html)
            # Detect paywall / login redirect
            if any(phrase in text.lower() for phrase in ("sign in to meltwater", "log in to meltwater")):
                logger.warning("Meltwater session expired mid-batch for %s", url)
                return None
            return text
    except Exception as exc:
        logger.warning("Meltwater cookie-seeded Playwright fetch failed for %s: %s", url, exc)
        return None


def fetch_and_extract(
    url: str,
    hit_sentence: str = "",
    mw_cookies_str: str = "",
    mw_playwright_cookies: dict | None = None,
) -> str:
    if not url.startswith("http://") and not url.startswith("https://"):
        return "ERROR: Not a web URL (internal document ID)"

    article_text: str | None = None

    # ── Meltwater-specific authenticated fetch ────────────────────────────────
    if is_meltwater_url(url):
        if mw_playwright_cookies:
            article_text = fetch_meltwater_with_playwright_cookies(url, mw_playwright_cookies)
        elif mw_cookies_str.strip():
            article_text = fetch_meltwater_with_cookies(url, mw_cookies_str)
        if article_text is None:
            if not mw_cookies_str.strip() and not mw_playwright_cookies:
                return (
                    "ERROR: Meltwater URL requires authentication — "
                    "add credentials in the Meltwater Auth section above"
                )
            return "ERROR: Meltwater auth failed or session expired — check credentials"

    # ── Standard fetch pipeline ───────────────────────────────────────────────
    if article_text is None:
        article_text = fetch_via_curl_cffi(url)
    if not article_text:
        article_text = fetch_via_playwright(url)
    if not article_text:
        try:
            resp = fetch_with_retry(url)
            if resp is None:
                return "ERROR: Failed after retries"
            article_text = extract_article_text(resp.text)
        except requests.exceptions.Timeout:
            return "ERROR: Request timed out"
        except requests.exceptions.TooManyRedirects:
            return "ERROR: Too many redirects"
        except requests.exceptions.RequestException as exc:
            return f"ERROR: {exc}"

    sentences = split_sentences(article_text)
    anchor_phrases = extract_anchor_phrases(hit_sentence) if hit_sentence else []
    context = find_best_context(sentences, anchor_phrases)
    return context if context is not None else "ERROR: Mention not found in fetched page"


def translate_to_english(text: str) -> str:
    try:
        bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        resp = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Translate the following text to English. "
                        "Preserve the Before:/Mention:/After: labels exactly as-is. "
                        "Return only the translated text, nothing else.\n\n"
                        f"{text}"
                    ),
                }],
            }),
        )
        return json.loads(resp["body"].read())["content"][0]["text"].strip()
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return text


def normalize_column(col: str, headers: list[str]) -> str | None:
    col = col.strip()
    for h in headers:
        if h.strip().lower() == col.lower():
            return h
    if re.fullmatch(r"[A-Za-z]{1,2}", col):
        idx = 0
        for ch in col.upper():
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        idx -= 1
        if 0 <= idx < len(headers):
            return headers[idx]
    return None


def interleave_by_domain(rows: list[dict], url_col: str) -> list[dict]:
    domain_groups: dict[str, list] = {}
    for row in rows:
        domain = urlparse(row.get(url_col, "").strip()).netloc or ""
        domain_groups.setdefault(domain, []).append(row)
    interleaved = []
    while any(domain_groups.values()):
        for d in list(domain_groups.keys()):
            if domain_groups[d]:
                interleaved.append(domain_groups[d].pop(0))
            if not domain_groups[d]:
                del domain_groups[d]
    return interleaved


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #f7f6f3;
    }
    .block-container {
        padding-top: 2.5rem;
        max-width: 760px;
    }
    h1 {
        font-family: 'DM Mono', monospace !important;
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em;
        color: #1a1a1a;
        border-bottom: 2px solid #1a1a1a;
        padding-bottom: 0.6rem;
        margin-bottom: 0.2rem !important;
    }
    .subtitle {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 2rem;
        font-family: 'DM Mono', monospace;
    }
    .stButton > button {
        background: #1a1a1a;
        color: #f7f6f3;
        border: none;
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        padding: 0.55rem 1.4rem;
        transition: opacity 0.15s;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.8;
        color: #f7f6f3;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        border-radius: 4px;
        border-color: #d0cfc9;
        background: #fff;
    }
    label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #444 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        flex: 1;
        background: #fff;
        border: 1px solid #e0dfd9;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .stat-num {
        font-family: 'DM Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: #1a1a1a;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-top: 0.3rem;
    }
    .stat-num.errors { color: #c0392b; }
    .stDownloadButton > button {
        background: #fff;
        color: #1a1a1a;
        border: 2px solid #1a1a1a;
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        width: 100%;
        margin-top: 0.5rem;
    }
    .stDownloadButton > button:hover {
        background: #1a1a1a;
        color: #f7f6f3;
    }
    hr {
        border: none;
        border-top: 1px solid #e0dfd9;
        margin: 1.5rem 0;
    }
    .stCheckbox label {
        text-transform: none !important;
        letter-spacing: normal !important;
        font-size: 0.9rem !important;
    }
    .stProgress > div > div > div {
        background: #1a1a1a;
    }
    [data-testid="stFileUploader"] {
        background: #fff;
        border: 1px dashed #c0bfba;
        border-radius: 6px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# GoDaddy Mention Extractor")
st.markdown('<p class="subtitle">Upload a CSV → extract mention context → download enriched CSV</p>', unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("CSV File", type=["csv"], label_visibility="visible")

col1, col2 = st.columns(2)
with col1:
    url_col_input = st.text_input("URL Column", value="Article URL",
                                   help="Column header name (e.g. 'Article URL') or letter (e.g. 'B')")
with col2:
    output_col_name = st.text_input("Output Column Name", value="GoDaddy Mentions")

col3, col4 = st.columns([2, 1])
with col3:
    delay = st.slider("Delay between requests (s)", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
with col4:
    translate = st.checkbox("Translate to English", value=False,
                             help="Uses AWS Bedrock (Claude) to translate non-English mentions")

# ── Meltwater Authentication ───────────────────────────────────────────────────
with st.expander("🔐 Meltwater Authentication", expanded=False):
    st.markdown(
        "<small style='color:#666;font-family:DM Mono,monospace'>"
        "Broadcast transcript URLs on Meltwater (transition.meltwater.com) require login. "
        "Choose how to authenticate below.</small>",
        unsafe_allow_html=True,
    )
    mw_auth_method = st.radio(
        "Authentication method",
        ["None", "Email & Password (Playwright)", "Paste Session Cookie"],
        horizontal=True,
        label_visibility="collapsed",
    )

    mw_email, mw_password, mw_cookie_str = "", "", ""

    if mw_auth_method == "Email & Password (Playwright)":
        st.markdown(
            "<small style='color:#888;font-family:DM Mono,monospace'>"
            "Playwright will log in once and reuse the session for all Meltwater URLs in this run. "
            "Credentials are never stored.</small>",
            unsafe_allow_html=True,
        )
        mw_col1, mw_col2 = st.columns(2)
        with mw_col1:
            mw_email = st.text_input("Meltwater Email", key="mw_email")
        with mw_col2:
            mw_password = st.text_input("Meltwater Password", type="password", key="mw_password")

    elif mw_auth_method == "Paste Session Cookie":
        st.markdown(
            "<small style='color:#888;font-family:DM Mono,monospace'>"
            "Open a Meltwater page while logged in → DevTools → Application → Cookies → "
            "copy the full cookie string and paste it here.</small>",
            unsafe_allow_html=True,
        )
        mw_cookie_str = st.text_area(
            "Cookie string (name=value; name2=value2; …)",
            height=80,
            key="mw_cookie_str",
            placeholder="__cf_bm=abc123; mw_session=xyz789; …",
        )

st.markdown("<hr>", unsafe_allow_html=True)
run_button = st.button("Extract Mentions")

# ── Processing ────────────────────────────────────────────────────────────────
if run_button:
    if not uploaded_file:
        st.error("Please upload a CSV file first.")
        st.stop()
    if not url_col_input.strip():
        st.error("Please specify the URL column.")
        st.stop()

    # Decode CSV
    raw = uploaded_file.read()
    content = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        st.error("Could not decode CSV. Please re-save it as UTF-8.")
        st.stop()

    reader = csv.DictReader(io.StringIO(content))
    headers = list(reader.fieldnames or [])
    if not headers:
        st.error("CSV appears to have no headers.")
        st.stop()

    url_col = normalize_column(url_col_input, headers)
    if url_col is None:
        st.error(f"Column '{url_col_input}' not found. Available: {', '.join(headers)}")
        st.stop()

    rows = list(reader)
    if not rows:
        st.error("CSV has no data rows.")
        st.stop()

    interleaved = interleave_by_domain(rows, url_col)
    out_headers = headers + ([output_col_name] if output_col_name not in headers else [])

    total = len(interleaved)
    ok_count = 0
    error_count = 0

    # ── Meltwater pre-login (once per run) ───────────────────────────────────
    mw_playwright_cookies: dict | None = None
    has_mw_urls = any(is_meltwater_url(r.get(url_col, "").strip()) for r in interleaved)

    if has_mw_urls and mw_auth_method == "Email & Password (Playwright)":
        if mw_email and mw_password:
            with st.spinner("🔐 Logging in to Meltwater…"):
                mw_playwright_cookies = fetch_meltwater_login_and_get_cookies(mw_email, mw_password)
            if mw_playwright_cookies:
                st.success(f"✓ Meltwater login succeeded ({len(mw_playwright_cookies)} cookies captured)")
            else:
                st.error("⚠ Meltwater login failed — Playwright could not authenticate. Check credentials.")
        else:
            st.warning("⚠ Meltwater URLs detected but no email/password provided — those rows will be skipped.")

    # Progress UI
    status_text = st.empty()
    progress_bar = st.progress(0)
    stats_placeholder = st.empty()

    completed: dict[int, dict] = {}
    domain_last_hit: dict[str, float] = {}

    for i, row in enumerate(interleaved):
        url = row.get(url_col, "").strip()
        hit_sentence = row.get(HIT_SENTENCE_COL, "").strip()

        # Domain throttle
        if url:
            domain = urlparse(url).netloc
            if domain in domain_last_hit:
                elapsed = time.time() - domain_last_hit[domain]
                if elapsed < DOMAIN_MIN_DELAY:
                    wait = DOMAIN_MIN_DELAY - elapsed + random.uniform(0.5, 1.5)
                    status_text.markdown(
                        f"<small style='font-family:DM Mono,monospace;color:#888'>⏳ Throttling `{domain}` — {wait:.1f}s</small>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(wait)

            status_text.markdown(
                f"<small style='font-family:DM Mono,monospace;color:#555'>🔍 {url[:80]}{'…' if len(url)>80 else ''}</small>",
                unsafe_allow_html=True,
            )
            result = fetch_and_extract(
                url,
                hit_sentence,
                mw_cookies_str=mw_cookie_str,
                mw_playwright_cookies=mw_playwright_cookies,
            )
            domain_last_hit[domain] = time.time()

            if translate and result and not result.startswith("ERROR"):
                lang = row.get("Language", "").strip().lower()
                if lang and lang != "english":
                    status_text.markdown(
                        f"<small style='font-family:DM Mono,monospace;color:#888'>🌐 Translating ({lang})…</small>",
                        unsafe_allow_html=True,
                    )
                    result = translate_to_english(result)

            row[output_col_name] = result
            if result.startswith("ERROR"):
                error_count += 1
            else:
                ok_count += 1

            if i < total - 1:
                time.sleep(delay + random.uniform(0.5, 1.0))
        else:
            row[output_col_name] = "NO_URL"

        completed[id(row)] = row
        progress_bar.progress((i + 1) / total)

        stats_placeholder.markdown(
            f"""<div class="stat-row">
                <div class="stat-box"><div class="stat-num">{i+1}/{total}</div><div class="stat-label">Processed</div></div>
                <div class="stat-box"><div class="stat-num">{ok_count}</div><div class="stat-label">Extracted</div></div>
                <div class="stat-box"><div class="stat-num errors">{error_count}</div><div class="stat-label">Errors</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Re-order to original row order
    ordered = [completed.get(id(r), r) for r in rows]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=out_headers, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(ordered)
    csv_bytes = output.getvalue().encode("utf-8-sig")

    status_text.markdown(
        "<small style='font-family:DM Mono,monospace;color:#1a1a1a'>✓ Done</small>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.download_button(
        label="⬇ Download Enriched CSV",
        data=csv_bytes,
        file_name="mentions_output.csv",
        mime="text/csv",
    )
