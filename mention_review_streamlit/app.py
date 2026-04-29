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


def fetch_and_extract(url: str, hit_sentence: str = "") -> str:
    if not url.startswith("http://") and not url.startswith("https://"):
        return "ERROR: Not a web URL (internal document ID)"

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
            result = fetch_and_extract(url, hit_sentence)
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
