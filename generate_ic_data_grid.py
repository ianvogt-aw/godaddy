"""
GoDaddy IC Data Grid Generator
================================
Pulls mentions from all 18 Cision One mention streams, tags T1 vs Non-T1,
and outputs an Excel workbook matching the IC Data grid format.

Usage:
    export CISION_API_TOKEN="your-token-here"
    python generate_ic_data_grid.py --after 2026-05-01 --before 2026-06-01

    # Append to an existing grid:
    python generate_ic_data_grid.py --after 2026-06-01 --before 2026-06-11 \
        --append-to GoDaddy_IC_Data_-_2026.xlsx

    # Also re-check Yellow-flagged mentions against the full article text
    # (fetches each flagged mention's own URL — off by default, adds runtime):
    python generate_ic_data_grid.py --after 2026-05-01 --before 2026-06-01 --fetch-full-text

Requirements:
    pip install requests openpyxl python-dateutil beautifulsoup4 lxml

    Optional, only used by --fetch-full-text as upgrades over plain `requests`
    (both are skipped silently if absent):
        export TAVILY_API_KEY="your-tavily-key"   # best at paywalls/bot-detection
        pip install curl_cffi                     # Chrome TLS-fingerprint spoofing
"""

import argparse
import os
import re
import sys
import time
import json
import requests
from copy import copy
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from collections import defaultdict

import openpyxl
from bs4 import BeautifulSoup
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.styles.colors import Color
from openpyxl.utils import get_column_letter
from dateutil import parser as dtparser

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "https://api.cision.one"
REQUEST_INTERVAL = 7.0
MAX_RETRIES = 5

# Stream ID → metadata mapping
# Each entry: (stream_id, label, tier, category, tab_name, input_name_template)
STREAMS = [
    # ── PRODUCTS T1 ───────────────────────────────────────────────────
    {"id": 941746, "label": "AGI",      "tier": "T1",     "group": "PRODUCTS",  "tab": "AGI (Product) + Int.",       "input_name": "GoDaddy Product - T1 - AGI"},
    {"id": 941752, "label": "ANS",      "tier": "T1",     "group": "PRODUCTS",  "tab": "ANS (Product) + Int.",       "input_name": "GoDaddy Product - T1 - ANS"},
    {"id": 941764, "label": "Airo",     "tier": "T1",     "group": "PRODUCTS",  "tab": "Airo (Product) + Int.",      "input_name": "GoDaddy Product - T1 - Airo"},
    {"id": 941768, "label": "Commerce", "tier": "T1",     "group": "PRODUCTS",  "tab": "Commerce (Product) + Int.",  "input_name": "GoDaddy Product - T1 - Commerce"},
    {"id": 941776, "label": "Other",    "tier": "T1",     "group": "PRODUCTS",  "tab": "Other (Product) + Int.",     "input_name": "GoDaddy Product - T1 - Other"},
    # ── CORPORATE T1 ──────────────────────────────────────────────────
    {"id": 941782, "label": "Brand",               "tier": "T1",  "group": "CORPORATE", "tab": "Brand + Int.",               "input_name": "GoDaddy Brand - T1"},
    {"id": 941789, "label": "Finance",             "tier": "T1",  "group": "CORPORATE", "tab": "Finance + Int.",             "input_name": "GoDaddy - Finance - T1"},
    {"id": 941792, "label": "Thought Leadership",  "tier": "T1",  "group": "CORPORATE", "tab": "Thought Leadership + Int.",  "input_name": "GoDaddy Thought Leadership - T1", "split_by_person": True},
    {"id": 964987, "label": "ANS Open Standard",   "tier": "T1",  "group": "ANS_OPEN_STANDARD", "tab": "ANS Open Standard", "input_name": "ANS Open Standard - T1"},
    {"id": 965728, "label": "Brand Identity",      "tier": "T1",  "group": "BRAND_IDENTITY", "tab": "Brand Identity", "input_name": "Brand Identity - T1"},
    # ── SBRL T1 ───────────────────────────────────────────────────────
    {"id": 941796, "label": "sbrl", "tier": "T1",     "group": "SBRL", "tab": "GDSBRL + Int.", "input_name": "GoDaddy Venture Forward - T1"},
    # ── PRODUCTS (Non-T1) ─────────────────────────────────────────────
    {"id": 943540, "label": "AGI",      "tier": "Non-T1", "group": "PRODUCTS",  "tab": "AGI (Product) + Int.",       "input_name": "GoDaddy Product - Non T1 - AGI"},
    {"id": 943645, "label": "ANS",      "tier": "Non-T1", "group": "PRODUCTS",  "tab": "ANS (Product) + Int.",       "input_name": "GoDaddy Product - Non T1 - ANS"},
    {"id": 943635, "label": "Airo",     "tier": "Non-T1", "group": "PRODUCTS",  "tab": "Airo (Product) + Int.",      "input_name": "GoDaddy Product - Non T1 - Airo"},
    {"id": 943636, "label": "Commerce", "tier": "Non-T1", "group": "PRODUCTS",  "tab": "Commerce (Product) + Int.",  "input_name": "GoDaddy Product - Non T1- Commerce"},
    {"id": 943638, "label": "Other",    "tier": "Non-T1", "group": "PRODUCTS",  "tab": "Other (Product) + Int.",     "input_name": "GoDaddy Product - Non T1 - Other"},
    # ── CORPORATE (Non-T1) ────────────────────────────────────────────
    {"id": 943639, "label": "Brand",               "tier": "Non-T1", "group": "CORPORATE", "tab": "Brand + Int.",               "input_name": "GoDaddy Brand - Non T1"},
    {"id": 943641, "label": "Finance",             "tier": "Non-T1", "group": "CORPORATE", "tab": "Finance + Int.",             "input_name": "GoDaddy - Finance - Non T1"},
    {"id": 943642, "label": "Thought Leadership",  "tier": "Non-T1", "group": "CORPORATE", "tab": "Thought Leadership + Int.",  "input_name": "GoDaddy Thought Leadership - Non T1", "split_by_person": True},
    {"id": 964967, "label": "ANS Open Standard",   "tier": "Non-T1", "group": "ANS_OPEN_STANDARD", "tab": "ANS Open Standard", "input_name": "ANS Open Standard - Non T1"},
    {"id": 965727, "label": "Brand Identity",      "tier": "Non-T1", "group": "BRAND_IDENTITY", "tab": "Brand Identity", "input_name": "Brand Identity - Non T1"},
    # ── SBRL (Non-T1) ────────────────────────────────────────────────
    {"id": 943644, "label": "sblr", "tier": "Non-T1", "group": "SBRL", "tab": "GDSBRL + Int.", "input_name": "GoDaddy Venture Forward - Non T1"},
]

# Thought Leadership mentions are split out of the combined stream into one tab per
# spokesperson, based on which person's name appears in the mention's Keywords field.
# A mention tagged to multiple people (e.g. "Travis Muhlestein;Jared Sine") lands in
# both tabs.
PEOPLE = [
    "Aman Bhutani",
    "Gourav Pani",
    "Mark McCaffrey",
    "Kasturi Mudulodu",
    "Jared Sine",
    "Sarfraz Nakai",
    "Charles Beadnall",
    "Travis Muhlestein",
    "Dimitria Elmore",
    "Paul Bindel",
    "Phontip Palitwanon",
]
PEOPLE_TABS = {name: f"{name} + Int." for name in PEOPLE}
PEOPLE_LOOKUP = {name.lower(): tab for name, tab in PEOPLE_TABS.items()}

# Desired tab ordering in the output workbook
TAB_ORDER = [
    "GDSBRL + Int.",
    "Commerce (Product) + Int.",
    "AGI (Product) + Int.",
    "Airo (Product) + Int.",
    "ANS (Product) + Int.",
    "Other (Product) + Int.",
    "ANS Open Standard",
    "Brand + Int.",
    "Finance + Int.",
] + list(PEOPLE_TABS.values()) + [
    "Brand Identity",  # left uncolored (no entry in TAB_COLOR_BY_TAB) — sits at the end
]

# The 42 columns matching the existing IC Data grid
IC_COLUMNS = [
    "Date", "Time", "Document ID", "URL", "Input Name", "Keywords",
    "Information Type", "Source Type", "Source Name", "Source Domain",
    "Content Type", "Author Name", "Author Handle", "Title",
    "Opening Text", "Hit Sentence", "Image", "Hashtags", "Links",
    "Country", "Region", "State", "City", "Language", "Sentiment",
    "Keyphrases", "Reach", "AVE", "Social Echo", "Editorial Echo",
    "Engagement", "Shares", "Quotes", "Likes", "Replies", "Reposts",
    "Comments", "Reactions", "Views", "Estimated Views",
    "Document Tags", "Custom Categories",
]

# Column widths for readability
COL_WIDTHS = {
    "A": 12, "B": 8, "C": 16, "D": 50, "E": 35, "F": 30,
    "G": 14, "H": 14, "I": 25, "J": 25, "K": 15, "L": 18,
    "M": 15, "N": 60, "O": 40, "P": 60, "Q": 40, "R": 20,
    "S": 40, "T": 15, "U": 15, "V": 15, "W": 15, "X": 12,
    "Y": 10, "Z": 30, "AA": 12, "AB": 12, "AC": 12, "AD": 12,
    "AE": 12, "AF": 10, "AG": 10, "AH": 10, "AI": 10, "AJ": 10,
    "AK": 10, "AL": 10, "AM": 10, "AN": 12, "AO": 15, "AP": 25,
}


# ============================================================================
# API CLIENT (rate-limited)
# ============================================================================

class CisionOneClient:
    def __init__(self, api_token: str):
        self.token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            "X-Auth-Token": self.token,
            "Accept": "application/json",
        })
        self._last_request_time = 0.0

    def _throttled_get(self, url, params):
        for attempt in range(1, MAX_RETRIES + 1):
            elapsed = time.time() - self._last_request_time
            if elapsed < REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.time()
            resp = self.session.get(url, params=params)
            if resp.status_code != 429:
                return resp
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else REQUEST_INTERVAL * (2 ** (attempt - 1))
            print(f"    ⏳ Rate-limited (429). Retry {attempt}/{MAX_RETRIES} — waiting {wait:.0f}s")
            time.sleep(wait)
        resp.raise_for_status()

    def get_mentions(self, stream_id, after, before, page=1, page_size=500):
        resp = self._throttled_get(
            f"{BASE_URL}/public/api/v2/mentions/{stream_id}",
            {
                "filter[range][after]": after,
                "filter[range][before]": before,
                "pagination[page]": page,
                "pagination[page_size]": page_size,
                "sort[field]": "timestamp",
                "sort[order]": "desc",
                "format": "json",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def get_all_mentions(self, stream_id, after, before, page_size=500, max_pages=9):
        all_mentions = []
        page = 1
        while page <= max_pages and (page * page_size) <= 5000:
            try:
                batch = self.get_mentions(stream_id, after, before, page=page, page_size=page_size)
            except requests.HTTPError as e:
                if e.response.status_code == 400:
                    print(f"    ⚠️  Hit API ceiling at page {page} — returning {len(all_mentions)} mentions")
                    break
                raise
            if not batch:
                break
            all_mentions.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return all_mentions

    def get_all_mentions_chunked(self, stream_id, after, before, chunk_days=1, page_size=500):
        start = dtparser.isoparse(after)
        end = dtparser.isoparse(before)
        chunk = timedelta(days=chunk_days)
        all_mentions = []
        window_start = start
        while window_start < end:
            window_end = min(window_start + chunk, end)
            w_after = window_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            w_before = window_end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            batch = self.get_all_mentions(stream_id, w_after, w_before, page_size=page_size)
            all_mentions.extend(batch)
            window_start = window_end
        # Deduplicate
        seen = set()
        deduped = []
        for m in all_mentions:
            mid = m.get("id")
            if mid not in seen:
                seen.add(mid)
                deduped.append(m)
        return deduped


# ============================================================================
# FIELD MAPPING: Cision One API → IC Data Grid columns
# ============================================================================

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        return domain.replace("www.", "")
    except Exception:
        return ""


def map_medium_to_source_type(medium: str, mention_type: str) -> str:
    """Map API medium/type to IC Data 'Source Type' values."""
    mapping = {
        "Online": "online news",
        "Print": "print",
        "TV": "tv",
        "Radio": "radio",
        "Social": "social network",
        "Podcast": "podcast",
        "Magazine": "magazine",
    }
    return mapping.get(medium, mention_type or "")


def map_type_to_content_type(mention_type: str) -> str:
    """Map API 'type' to IC Data 'Content Type' values."""
    mapping = {
        "onlineArticle": "News Article",
        "printArticle": "News Article",
        "tvClip": "Video",
        "radioClip": "Audio",
        "socialPost": "Social Post",
        "podcastEpisode": "Audio",
        "magazineArticle": "News Article",
        "blogPost": "Blog Post",
    }
    return mapping.get(mention_type, mention_type or "")


def map_info_type(medium: str) -> str:
    """Map API medium to IC Data 'Information Type'."""
    if medium in ("TV", "Radio"):
        return "broadcast"
    elif medium == "Social":
        return "social"
    return "news"


def map_sentiment_label(score) -> str:
    """Map numeric sentiment score to label."""
    if score is None:
        return ""
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    return "neutral"


def map_language_code(code: str) -> str:
    """Map ISO language codes to display names."""
    if not code:
        return ""
    lang_map = {
        "en": "English", "en-US": "English", "en-GB": "English", "en-AU": "English",
        "es": "Spanish", "es-ES": "Spanish", "es-MX": "Spanish",
        "fr": "French", "fr-FR": "French", "fr-CA": "French",
        "de": "German", "de-DE": "German",
        "pt": "Portuguese", "pt-BR": "Portuguese",
        "it": "Italian", "ja": "Japanese", "ko": "Korean",
        "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
        "nl": "Dutch", "sv": "Swedish", "da": "Danish",
        "no": "Norwegian", "fi": "Finnish", "pl": "Polish",
        "ru": "Russian", "tr": "Turkish", "th": "Thai",
        "id": "Indonesian", "ms": "Malay", "vi": "Vietnamese",
    }
    return lang_map.get(code, lang_map.get(code.split("-")[0], code))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def focus_hit_sentence(text: str, keyword: str = "GoDaddy") -> tuple[str, bool]:
    """Prefer the sentence mentioning `keyword` over Cision's raw excerpt/transcript
    blob. The API only gives us one fixed excerpt with no keyword offsets or full
    article body, so this can only re-center within text Cision already returned —
    if `keyword` isn't present anywhere in it, the text is returned unchanged.
    Returns (text, found) so callers can flag mentions where `keyword` never
    turned up at all."""
    if not text:
        return text, False
    found = keyword.lower() in text.lower()
    if not found:
        return text, False
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if keyword.lower() in sentence.lower():
            return sentence.strip(), True
    return text, True


# ============================================================================
# FULL-ARTICLE FALLBACK (--fetch-full-text)
# ============================================================================
# Cision's excerpt is a single fixed snippet — if "GoDaddy" isn't in it, that
# doesn't mean the article never mentions GoDaddy, just that Cision's snippet
# didn't happen to land there. For mentions flagged Yellow, we can fetch the
# article's own public URL and search the real full text ourselves, the same
# way mention_review_streamlit/app.py does for old Meltwater data (fetch ->
# strip to body text -> split into sentences -> find the one with "GoDaddy").

ARTICLE_FETCH_TIMEOUT = 15
ARTICLE_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
ARTICLE_FETCH_DELAY = 1.0  # seconds between fetches, so we don't hammer news sites

# Optional, in priority order: Tavily's Extract API (best at paywalls/bot-detection,
# no local browser needed — set TAVILY_API_KEY to enable) and curl_cffi (mimics
# Chrome's TLS fingerprint, gets past a lot of basic bot-blocking WAFs). Both are
# skipped silently if unconfigured/not installed — plain `requests` always runs last.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

_ABBREV_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|Inc|Ltd|Corp|Co)\.")
_ROBUST_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'‘’“”\(\[])")


def _split_sentences_robust(text: str) -> list[str]:
    """Sentence splitter for full article bodies — tolerant of abbreviations
    ("Inc.", "Corp.", etc.) that would otherwise cause false splits. Ported from
    mention_review_streamlit/app.py's split_sentences()."""
    text = re.sub(r"\s+", " ", text).strip()
    text = _ABBREV_RE.sub(r"\1<DOT>", text)
    parts = _ROBUST_SENTENCE_SPLIT_RE.split(text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _extract_article_text(html: str) -> str:
    """Best-effort article body extraction. Ported from
    mention_review_streamlit/app.py's extract_article_text()."""
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


def _fetch_via_tavily(url: str) -> str | None:
    """Fetch article text via Tavily's Extract API. Returns None if TAVILY_API_KEY
    isn't set or the call fails — this is a silent, optional upgrade."""
    if not TAVILY_API_KEY:
        return None
    try:
        resp = requests.post(
            TAVILY_EXTRACT_URL,
            json={"urls": [url], "extract_depth": "advanced", "format": "text"},
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0].get("raw_content", "") or results[0].get("content", "") or None
        return None
    except Exception:
        return None


def _fetch_via_curl_cffi(url: str) -> str | None:
    """Fetch via curl_cffi (Chrome TLS-fingerprint impersonation) if it's installed.
    Returns None if the package is missing or the fetch fails."""
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        return None
    try:
        resp = cffi_requests.get(url, impersonate="chrome120", timeout=ARTICLE_FETCH_TIMEOUT, allow_redirects=True)
        if resp.status_code != 200:
            return None
        return _extract_article_text(resp.text)
    except Exception:
        return None


def fetch_full_article_text(url: str) -> str | None:
    """Best-effort fetch of the full article body at `url`, trying (in order)
    Tavily's Extract API, curl_cffi, then plain `requests`. Returns None if every
    method fails — callers must fall back gracefully; this is a bonus check, not
    something the pipeline depends on."""
    text = _fetch_via_tavily(url)
    if text:
        return text

    text = _fetch_via_curl_cffi(url)
    if text:
        return text

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": ARTICLE_FETCH_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=ARTICLE_FETCH_TIMEOUT,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        return _extract_article_text(resp.text)
    except Exception:
        return None


def find_keyword_sentence(text: str, keyword: str = "GoDaddy") -> str | None:
    """Locate the first sentence in a full article body mentioning `keyword`.
    Returns None if it isn't there either."""
    if not text or keyword.lower() not in text.lower():
        return None
    for sentence in _split_sentences_robust(text):
        if keyword.lower() in sentence.lower():
            return sentence.strip()
    return None


class Row(list):
    """A data row that also remembers whether its Hit Sentence should be flagged
    Yellow (the focus keyword never turned up in Cision's excerpt/transcript)."""
    hit_sentence_flagged = False


def mention_to_row(mention: dict, stream_config: dict, fetch_full_text: bool = False) -> list:
    """Convert a single API mention + stream metadata → 42-element list matching
    IC_COLUMNS. If fetch_full_text is set and Cision's excerpt doesn't mention
    "GoDaddy", makes one best-effort fetch of the mention's own public URL and
    searches the real article body before giving up and flagging Yellow."""
    pub = mention.get("publishedAt", "")
    dt = None
    if pub:
        try:
            dt = dtparser.isoparse(pub)
        except Exception:
            pass

    medium = mention.get("medium", "")
    mtype = mention.get("type", "")
    social = mention.get("social") or {}
    social_echo = sum(v for v in [
        social.get("x"), social.get("facebook"),
        social.get("reddit"), social.get("pinterest"),
    ] if v is not None) or None

    keywords_raw = mention.get("keywords") or []
    keywords_str = ";".join(keywords_raw) if keywords_raw else ""

    content_text = mention.get("excerpt") or mention.get("transcript") or ""
    hit_sentence_flagged = False
    if stream_config["tab"] != "ANS Open Standard":
        content_text, keyword_found = focus_hit_sentence(content_text, "GoDaddy")
        if not keyword_found and fetch_full_text and mention.get("url"):
            time.sleep(ARTICLE_FETCH_DELAY)
            full_text = fetch_full_article_text(mention["url"])
            better_sentence = find_keyword_sentence(full_text, "GoDaddy") if full_text else None
            if better_sentence:
                content_text = better_sentence
                keyword_found = True
        hit_sentence_flagged = not keyword_found

    # Fall back to Cision's internal link when the mention has no public URL
    # (common for broadcast clips and some social posts).
    url = mention.get("url") or mention.get("internalLink") or ""

    row = Row([
        dt.date() if dt else None,                         # Date
        dt.time() if dt else None,                         # Time
        mention.get("id"),                                  # Document ID
        url,                                                 # URL
        stream_config["input_name"],                        # Input Name
        keywords_str,                                       # Keywords
        map_info_type(medium),                              # Information Type
        map_medium_to_source_type(medium, mtype),           # Source Type
        mention.get("source", ""),                          # Source Name
        extract_domain(url),                                # Source Domain
        map_type_to_content_type(mtype),                    # Content Type
        mention.get("author", ""),                          # Author Name
        None,                                               # Author Handle (not in API)
        mention.get("title", ""),                           # Title
        None,                                               # Opening Text (not reliably in API)
        content_text,                                       # Hit Sentence
        None,                                               # Image (not in API)
        None,                                               # Hashtags (not in API)
        None,                                               # Links (not in API)
        mention.get("locationCountry", ""),                 # Country
        None,                                               # Region (not in API)
        mention.get("locationState", ""),                   # State
        mention.get("locationCity", ""),                     # City
        map_language_code(mention.get("languageCode", "")), # Language
        map_sentiment_label(mention.get("sentiment")),      # Sentiment
        keywords_str,                                       # Keyphrases (same as Keywords)
        mention.get("audience"),                            # Reach
        mention.get("advertisingValue"),                    # AVE
        social_echo,                                        # Social Echo
        None,                                               # Editorial Echo (not in API)
        None,                                               # Engagement (not in API)
        None,                                               # Shares
        None,                                               # Quotes
        None,                                               # Likes
        None,                                               # Replies
        None,                                               # Reposts
        None,                                               # Comments
        None,                                               # Reactions
        None,                                               # Views
        None,                                               # Estimated Views
        None,                                               # Document Tags (not in API)
        "Y" if stream_config["tier"] == "T1" else None,     # Custom Categories: Y for T1 streams
    ])
    row.hit_sentence_flagged = hit_sentence_flagged
    return row


# ============================================================================
# EXCEL OUTPUT
# ============================================================================

HEADER_FONT = Font(name="Arial", bold=True, size=10, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="003366")
HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
CELL_FONT = Font(name="Arial", size=10)
THIN_BORDER = Border(
    bottom=Side(style="thin", color="D0D0D0"),
)

# Data-label row fills, matching the Legend tab: Orange = new T1 data to review,
# Pink = new Non-T1 data to review. Yellow overrides both — it means the mention
# was flagged by focus_hit_sentence ("GoDaddy" never turned up in the Hit Sentence
# text) and needs a human look, so it takes priority over the tier color. Red
# ("Removed") is a manual designation this script never applies and never clears.
# NO_FILL is used to "age out" a row from Orange/Pink to White once it's no longer new.
ORANGE_FILL_ARGB = "FFFBE2D5"
PINK_FILL_ARGB = "FFF2CEEF"
RED_FILL_ARGB = "FFFF0000"
YELLOW_FILL_ARGB = "FFFFFF00"
ORANGE_FILL = PatternFill("solid", fgColor=ORANGE_FILL_ARGB)
PINK_FILL = PatternFill("solid", fgColor=PINK_FILL_ARGB)
YELLOW_FILL = PatternFill("solid", fgColor=YELLOW_FILL_ARGB)
NO_FILL = PatternFill(fill_type=None)

# Toggle for the Yellow "GoDaddy not found in Hit Sentence" row highlight. Set to
# True to re-enable it — mention_to_row/focus_hit_sentence/fetch_full_article_text
# still run and compute Row.hit_sentence_flagged either way, this just controls
# whether that flag is allowed to affect row color (and the Legend entry).
ENABLE_YELLOW_HIGHLIGHTING = False


def _row_fill(row_data: list) -> PatternFill:
    """Yellow if focus_hit_sentence flagged this row (see Row/mention_to_row) and
    ENABLE_YELLOW_HIGHLIGHTING is on, otherwise the tier color: Custom Categories
    (last column) is "Y" only for T1 rows."""
    if ENABLE_YELLOW_HIGHLIGHTING and getattr(row_data, "hit_sentence_flagged", False):
        return YELLOW_FILL
    return ORANGE_FILL if row_data[-1] == "Y" else PINK_FILL


def write_tab(ws, rows: list[list]):
    """Write header + data rows to a worksheet with formatting."""
    # Header row
    for col_idx, header in enumerate(IC_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGN

    # Data rows (sorted by date desc, then time desc)
    rows_sorted = sorted(rows, key=lambda r: (r[0] or datetime.min.date(), r[1] or datetime.min.time()), reverse=True)

    for row_idx, row_data in enumerate(rows_sorted, start=2):
        fill = _row_fill(row_data)
        for col_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font = CELL_FONT
            cell.border = THIN_BORDER
            cell.fill = fill

    # Column widths
    for col_letter, width in COL_WIDTHS.items():
        ws.column_dimensions[col_letter].width = width

    # Freeze header row
    ws.freeze_panes = "A2"

    # Auto-filter
    if rows:
        last_col = get_column_letter(len(IC_COLUMNS))
        ws.auto_filter.ref = f"A1:{last_col}{len(rows) + 1}"


# Sheet-tab colors, matching godaddy_data_june_2026.xlsx exactly (theme index + tint).
SBRL_TAB_COLOR = Color(theme=8, tint=0.7999816888943144, type="theme")
PRODUCT_TAB_COLOR = Color(theme=4, tint=0.7999816888943144, type="theme")
ANS_PRODUCT_TAB_COLOR = Color(theme=7, tint=0.7999816888943144, type="theme")  # ANS (Product) + Int. only — an outlier vs. the other Product tabs
# "Olive Green, Accent 3, Lighter 80%" — used for the Corporate group (Brand,
# Finance, Thought Leadership person tabs) and ANS Open Standard.
OLIVE_GREEN_TAB_COLOR = Color(theme=6, tint=0.7999816888943144, type="theme")

TAB_COLOR_BY_TAB = {
    "GDSBRL + Int.": SBRL_TAB_COLOR,
    "Commerce (Product) + Int.": PRODUCT_TAB_COLOR,
    "AGI (Product) + Int.": PRODUCT_TAB_COLOR,
    "Airo (Product) + Int.": PRODUCT_TAB_COLOR,
    "ANS (Product) + Int.": ANS_PRODUCT_TAB_COLOR,
    "Other (Product) + Int.": PRODUCT_TAB_COLOR,
    "ANS Open Standard": OLIVE_GREEN_TAB_COLOR,
    "Brand + Int.": OLIVE_GREEN_TAB_COLOR,
    "Finance + Int.": OLIVE_GREEN_TAB_COLOR,
    **{tab: OLIVE_GREEN_TAB_COLOR for tab in PEOPLE_TABS.values()},
}


def build_legend_sheet(wb, index: int = 0):
    """Create the Legend tab explaining the workbook's color-coding conventions."""
    ws = wb.create_sheet(title="Legend", index=index)
    ws.column_dimensions["B"].width = 21.1
    ws.column_dimensions["C"].width = 22.1

    ws["B2"] = "Data Labels"

    ws["B3"] = "Red"
    ws["B3"].fill = PatternFill("solid", fgColor="FFFF0000")
    ws["C3"] = "Removed"

    ws["B4"] = "Orange"
    ws["B4"].fill = PatternFill("solid", fgColor="FFFBE2D5")
    ws["C4"] = "Keep - T1"

    ws["B5"] = "Pink"
    ws["B5"].fill = PatternFill("solid", fgColor="FFF2CEEF")
    ws["C5"] = "Keep - Non-T1"

    if ENABLE_YELLOW_HIGHLIGHTING:
        ws["B6"] = "Yellow"
        ws["B6"].fill = PatternFill("solid", fgColor=YELLOW_FILL_ARGB)
        ws["C6"] = 'Hit Sentence doesn\'t mention "GoDaddy"'

    ws["B8"] = "Tabs"

    ws["B9"] = "Pink"
    ws["B9"].fill = PatternFill("solid", fgColor="FFF2CFEF")
    ws["C9"] = "SBRL"

    ws["B10"] = "Blue"
    ws["B10"].fill = PatternFill("solid", fgColor="FFC0E5F4")
    ws["C10"] = "Product"

    ws["B11"] = "Green"
    ws["B11"].fill = PatternFill(fill_type="solid", fgColor=OLIVE_GREEN_TAB_COLOR)
    ws["C11"] = "Corporate"

    ws["B12"] = "White"
    ws["C12"] = "Source of Truth/General"

    return ws


def create_workbook(tab_data: dict[str, list[list]], output_path: str):
    """Create a new IC Data workbook with all tabs."""
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    build_legend_sheet(wb)

    for tab_name in TAB_ORDER:
        rows = tab_data.get(tab_name, [])
        ws = wb.create_sheet(title=tab_name[:31])  # Excel 31-char tab name limit
        if tab_name in TAB_COLOR_BY_TAB:
            ws.sheet_properties.tabColor = TAB_COLOR_BY_TAB[tab_name]
        write_tab(ws, rows)
        print(f"  📄 {tab_name}: {len(rows)} mentions")

    resort_all_tabs(wb)
    wb.save(output_path)
    print(f"\n✅ Saved to {output_path}")


def _row_sort_key(cells_info):
    """Group by data-label color (Orange -> Pink -> Yellow -> everything else),
    newest first within each group. cells_info is the list of (value, font, fill,
    border, alignment, number_format) tuples captured for one row; Date is column
    1, Time is column 2."""
    _, _, fill, *_ = cells_info[0]
    fg = fill.fgColor
    if fill.fill_type == "solid" and fg.type == "rgb" and fg.rgb == ORANGE_FILL_ARGB:
        group = 0
    elif fill.fill_type == "solid" and fg.type == "rgb" and fg.rgb == PINK_FILL_ARGB:
        group = 1
    elif fill.fill_type == "solid" and fg.type == "rgb" and fg.rgb == YELLOW_FILL_ARGB:
        group = 2
    else:
        group = 3

    date_val = cells_info[0][0]
    time_val = cells_info[1][0]
    date_ord = date_val.toordinal() if date_val else 0
    time_secs = (time_val.hour * 3600 + time_val.minute * 60 + time_val.second) if time_val else 0
    return (group, -date_ord, -time_secs)


def resort_tab(ws):
    """Re-lay-out a data tab's rows: new T1 (Orange) first, new Non-T1 (Pink)
    second, flagged new data (Yellow) third, then everything else — newest-to-oldest
    by Date/Time within each group."""
    if ws.max_row < 3:
        return  # header only (or a single data row) — nothing to reorder

    max_col = ws.max_column
    entries = [
        # cell.font/.fill/.border/.alignment return an immutable StyleProxy; copy()
        # unwraps it into a real, reassignable style object.
        [(c.value, copy(c.font), copy(c.fill), copy(c.border), copy(c.alignment), c.number_format) for c in row]
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=max_col)
    ]
    entries.sort(key=_row_sort_key)

    for row_idx, cells_info in enumerate(entries, start=2):
        for col_idx, (value, font, fill, border, alignment, number_format) in enumerate(cells_info, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font = font
            cell.fill = fill
            cell.border = border
            cell.alignment = alignment
            cell.number_format = number_format


def resort_all_tabs(wb):
    """Apply resort_tab to every tab this script manages that exists in the workbook."""
    for tab_name in TAB_ORDER:
        ws_name = tab_name[:31]
        if ws_name in wb.sheetnames:
            resort_tab(wb[ws_name])


def downgrade_reviewed_rows(wb):
    """Age out last run's "new data" highlighting: any row still filled Orange or Pink
    (Keep - T1 / Keep - Non-T1) is cleared to White now that it's no longer new.
    Yellow (flagged: "GoDaddy" not found in Hit Sentence) is a standing content-quality
    flag, not a review-recency one — like Red (Removed), it's left untouched until a
    human resolves it. Only scans tabs this script manages (not foreign/manual sheets
    in the workbook)."""
    downgraded = 0
    for tab_name in TAB_ORDER:
        ws_name = tab_name[:31]
        if ws_name not in wb.sheetnames:
            continue
        ws = wb[ws_name]
        if ws.max_row < 2:
            continue
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
            marker_fill = row[0].fill
            fg = marker_fill.fgColor
            if marker_fill.fill_type == "solid" and fg.type == "rgb" and fg.rgb in (ORANGE_FILL_ARGB, PINK_FILL_ARGB):
                for cell in row:
                    cell.fill = NO_FILL
                downgraded += 1
    if downgraded:
        print(f"  🎨 Downgraded {downgraded} previously-new row(s) from Orange/Pink to White")
    return downgraded


def append_to_workbook(tab_data: dict[str, list[list]], existing_path: str, output_path: str):
    """Append new mentions to an existing IC Data workbook, deduplicating by Document ID."""
    wb = openpyxl.load_workbook(existing_path)

    if "Legend" not in wb.sheetnames:
        build_legend_sheet(wb)

    downgrade_reviewed_rows(wb)

    for tab_name, new_rows in tab_data.items():
        if not new_rows:
            continue

        # Find or create the worksheet
        # Account for 31-char truncation
        ws_name = tab_name[:31]
        if ws_name in wb.sheetnames:
            ws = wb[ws_name]
            # Collect existing Document IDs (column C = col 3)
            existing_ids = set()
            for row in ws.iter_rows(min_row=2, min_col=3, max_col=3, values_only=True):
                if row[0] is not None:
                    existing_ids.add(row[0])

            # Filter out duplicates
            new_unique = [r for r in new_rows if r[2] not in existing_ids]

            # Append new rows at the bottom, tagged Orange/Pink/Yellow as new data to review
            start_row = ws.max_row + 1
            for offset, row_data in enumerate(new_unique):
                row_idx = start_row + offset
                fill = _row_fill(row_data)
                for col_idx, value in enumerate(row_data, start=1):
                    cell = ws.cell(row=row_idx, column=col_idx, value=value)
                    cell.font = CELL_FONT
                    cell.border = THIN_BORDER
                    cell.fill = fill

            print(f"  📄 {tab_name}: +{len(new_unique)} new ({len(new_rows) - len(new_unique)} dupes skipped)")
        else:
            ws = wb.create_sheet(title=ws_name)
            if tab_name in TAB_COLOR_BY_TAB:
                ws.sheet_properties.tabColor = TAB_COLOR_BY_TAB[tab_name]
            write_tab(ws, new_rows)
            print(f"  📄 {tab_name}: {len(new_rows)} mentions (new tab)")

    resort_all_tabs(wb)
    wb.save(output_path)
    print(f"\n✅ Saved to {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def fetch_all_streams(client: CisionOneClient, after: str, before: str, fetch_full_text: bool = False) -> dict[str, list[list]]:
    """Fetch mentions from all 18 streams and organize by tab."""
    tab_data: dict[str, list[list]] = defaultdict(list)
    total_mentions = 0
    unmatched_person_mentions = 0
    flagged_count = 0

    for i, stream in enumerate(STREAMS, start=1):
        sid = stream["id"]
        label = stream["label"]
        tier = stream["tier"]
        tab = stream["tab"]

        print(f"\n[{i}/{len(STREAMS)}] Fetching {tier} — {label} (ID {sid}) → {tab}")

        try:
            mentions = client.get_all_mentions_chunked(sid, after, before, chunk_days=1)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

        if stream.get("split_by_person"):
            matched = 0
            for m in mentions:
                person_tabs = {
                    PEOPLE_LOOKUP[kw.lower()]
                    for kw in (m.get("keywords") or [])
                    if kw.lower() in PEOPLE_LOOKUP
                }
                if not person_tabs:
                    unmatched_person_mentions += 1
                    continue
                row = mention_to_row(m, stream, fetch_full_text=fetch_full_text)
                if row.hit_sentence_flagged:
                    flagged_count += 1
                for person_tab in person_tabs:
                    tab_data[person_tab].append(row)
                matched += 1
            print(f"    ✓ {len(mentions)} mentions → routed {matched} to person tabs")
        else:
            rows = [mention_to_row(m, stream, fetch_full_text=fetch_full_text) for m in mentions]
            flagged_count += sum(1 for r in rows if r.hit_sentence_flagged)
            tab_data[tab].extend(rows)
            print(f"    ✓ {len(mentions)} mentions")

        total_mentions += len(mentions)

    if unmatched_person_mentions:
        print(f"\n⚠️  {unmatched_person_mentions} Thought Leadership mention(s) didn't match any known person and were skipped")
    if flagged_count:
        suffix = " even after checking the full article" if fetch_full_text else " (run with --fetch-full-text to double-check the full article)"
        print(f"\n🟡 {flagged_count} mention(s) flagged Yellow — \"GoDaddy\" not found{suffix}")

    # Deduplicate within each tab (same article can appear in T1 and Non-T1 with different Input Names)
    # We keep both since they have different Input Names — that's intentional per the grid design
    print(f"\n📊 Total mentions across all streams: {total_mentions}")
    return dict(tab_data)


def main():
    parser = argparse.ArgumentParser(description="Generate GoDaddy IC Data grid from Cision One API")
    parser.add_argument("--after", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--before", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Output file path (default: GoDaddy_IC_Data_<after>_to_<before>.xlsx)")
    parser.add_argument("--append-to", default=None, help="Path to existing workbook to append to")
    parser.add_argument("--token", default=None, help="Cision One API token (or set CISION_API_TOKEN env var)")
    parser.add_argument(
        "--fetch-full-text", action="store_true",
        help="For mentions flagged Yellow (\"GoDaddy\" not in Cision's excerpt), fetch the "
             "mention's own URL and re-check the full article text before giving up. Off by "
             "default — adds one HTTP request per flagged mention.",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("CISION_API_TOKEN")
    if not token:
        print("Error: No API token. Pass --token or set CISION_API_TOKEN.")
        sys.exit(1)

    after_iso = f"{args.after}T00:00:00.000Z"
    before_iso = f"{args.before}T23:59:59.000Z"

    print("=" * 60)
    print("  GoDaddy IC Data Grid Generator")
    print("=" * 60)
    print(f"  Date range: {args.after} → {args.before}")
    print(f"  Streams: {len(STREAMS)}")
    print(f"  Rate limit: ~10 req/min ({REQUEST_INTERVAL}s between requests)")
    print(f"  Mode: {'Append' if args.append_to else 'New workbook'}")
    print(f"  Full-text re-check for flagged mentions: {'On' if args.fetch_full_text else 'Off'}")
    print()

    client = CisionOneClient(api_token=token)
    tab_data = fetch_all_streams(client, after_iso, before_iso, fetch_full_text=args.fetch_full_text)

    if args.append_to:
        output_path = args.output or args.append_to
        append_to_workbook(tab_data, args.append_to, output_path)
    else:
        output_path = args.output or f"GoDaddy_IC_Data_{args.after}_to_{args.before}.xlsx"
        create_workbook(tab_data, output_path)


if __name__ == "__main__":
    main()
