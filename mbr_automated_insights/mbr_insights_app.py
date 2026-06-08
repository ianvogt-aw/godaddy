import streamlit as st
import pandas as pd
import json
import boto3
from io import BytesIO

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Media Coverage Insights",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Media Coverage Insights Generator")
st.markdown(
    "Upload the GoDaddy IC Data workbook and let Claude generate "
    "business-unit summaries, an executive summary, and strategic "
    "insights — all in one click."
)

st.info(
    "**⚠️ Data Preparation Required:** This application assumes you are uploading "
    "a version of the GoDaddy IC Data grid with only relevant coverage data. Save a "
    "copy of the grid with only coverage from the month of interest (delete scrubbed "
    "rows + old coverage — use sorting to make this easy)."
)

# ──────────────────────────────────────────────────────────────
# Sheet-name → internal-key mapping  (matched by substring)
#
# Each tuple is (substring_to_match, internal_key).
# Matching is case-insensitive and checks whether the substring
# appears anywhere in the actual Excel sheet name, so small
# naming variations across yearly files are handled automatically.
# ──────────────────────────────────────────────────────────────
SHEET_MAP = [
    ("gdsbrl",              "small_business_research_lab"),
    ("commerce",            "commerce"),
    ("agi",                 "agi"),
    ("airo",                "airo"),
    ("ans open standard",   "ans_open_standard"),   # must precede generic "ans"
    ("ans",                 "ans"),
    ("other",               "other"),
    ("brand identity",      "_skip_brand_identity"), # ignore Brand Identity (v2)
    ("brand",               "brand"),
    ("finance + int",       "finance"),
    ("finance",             "finance"),              # fallback label
    ("aman bhutani",        "aman_bhutani"),
    ("gourav pani",         "gourav_pani"),
    ("kasturi mudulodu",    "kasturi_mudulodu"),
    ("mark mccaffrey",      "mark_mccaffrey"),
    ("jared sine",          "jared_sine"),
    ("travis muhlestein",   "travis_muhlestein"),
    ("demetria",            "demetria"),
    ("berea schaffer",      "berea_schaffer"),
    ("general",             "general"),
]

COLUMNS_TO_KEEP = ["Date", "Title", "Hit Sentence"]

# ──────────────────────────────────────────────────────────────
# Business-unit definitions
#
# Each entry: (internal_key, display_label, list_of_source_keys)
# If source_keys is None the key is read directly from the
# parsed sheets dict; otherwise sources are concatenated.
# ──────────────────────────────────────────────────────────────
BUSINESS_UNITS = [
    ("small_business_research_lab", "🔬 Small Business Research Lab", None),
    ("product", "🏭 Product", ["commerce", "agi", "airo", "ans", "other"]),
    ("ans_open_standard", "🌐 ANS Open Standard", None),
    ("brand", "🎨 Brand", None),
    (
        "thought_leadership",
        "👔 Thought Leadership",
        [
            "aman_bhutani",
            "gourav_pani",
            "kasturi_mudulodu",
            "mark_mccaffrey",
            "jared_sine",
            "travis_muhlestein",
            "demetria",
            "berea_schaffer",
        ],
    ),
    ("financial", "💰 Financial", ["finance"]),
    (
        "corporate",
        "🏢 Corporate",
        [
            "brand",
            "aman_bhutani",
            "gourav_pani",
            "kasturi_mudulodu",
            "mark_mccaffrey",
            "jared_sine",
            "travis_muhlestein",
            "demetria",
            "berea_schaffer",
            "finance",
        ],
    ),
]


# ──────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────
def _match_sheet(sheet_name: str) -> str | None:
    """Return the internal key for a given Excel sheet name, or None."""
    lower = sheet_name.lower()
    for substring, key in SHEET_MAP:
        if substring in lower:
            return key
    return None


@st.cache_data(show_spinner="Loading Excel file …")
def load_and_process(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Read the uploaded Excel and build combined datasets."""
    buf = BytesIO(file_bytes)
    xls = pd.ExcelFile(buf)

    # ── Parse individual sheets by name ──
    raw: dict[str, pd.DataFrame] = {}
    matched_keys: set[str] = set()

    for sheet_name in xls.sheet_names:
        key = _match_sheet(sheet_name)
        if key is None or key.startswith("_skip"):
            continue
        if key in matched_keys:
            # Duplicate match — skip (first match wins)
            continue
        matched_keys.add(key)
        df = pd.read_excel(buf, sheet_name=sheet_name, header=0)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        raw[key] = df

    # Warn about any critical sheets that were not found
    expected = {
        "small_business_research_lab", "commerce", "agi", "airo", "ans",
        "ans_open_standard", "other", "brand", "finance",
        "aman_bhutani", "gourav_pani", "kasturi_mudulodu",
        "mark_mccaffrey", "jared_sine",
        "travis_muhlestein", "demetria", "berea_schaffer",
    }
    missing = expected - set(raw.keys())
    if missing:
        st.warning(
            f"The following expected sheets were not matched: "
            f"{', '.join(sorted(missing))}. They will be treated as empty."
        )

    # Helper — safely select columns and return a (possibly empty) DF
    def cols(key: str) -> pd.DataFrame:
        if key not in raw:
            return pd.DataFrame(columns=COLUMNS_TO_KEEP)
        df = raw[key]
        return df[[c for c in COLUMNS_TO_KEEP if c in df.columns]]

    # ── Build combined datasets for each business unit ──
    datasets: dict[str, pd.DataFrame] = {}
    for unit_key, _label, sources in BUSINESS_UNITS:
        if sources is None:
            datasets[unit_key] = cols(unit_key)
        else:
            datasets[unit_key] = pd.concat(
                [cols(s) for s in sources], ignore_index=True
            )

    # ── All-coverage union (for article counts / overview) ──
    datasets["all_coverage"] = (
        pd.concat(
            [cols(k) for k in raw if not k.startswith("_skip") and k != "general"],
            ignore_index=True,
        )
        .drop_duplicates()
    )

    return datasets


# ──────────────────────────────────────────────────────────────
# AWS Bedrock configuration
# ──────────────────────────────────────────────────────────────
AWS_REGION = st.secrets["BEDROCK_REGION"]
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
CLAUDE_MODEL_ID = st.secrets["BEDROCK_MODEL_ID"]


# ──────────────────────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────────────────────
def call_claude(bedrock_client, prompt: str, max_tokens: int = 400) -> str:
    """Invoke Claude via AWS Bedrock."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    })
    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def generate_coverage_summary(client, df: pd.DataFrame, unit_name: str) -> str:
    """Generate a BU-level coverage summary."""
    coverage_data = df[["Date", "Title", "Hit Sentence"]].to_string(index=False)
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = (
        f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        if pd.notna(min_date) else "N/A"
    )
    prompt = f"""You are analyzing media coverage data for the {unit_name} area of GoDaddy.

Dataset: {unit_name}
Total articles: {len(df)}
Date range: {date_range}

Here is the complete coverage data:
{coverage_data}

Please provide a brief, concise summary (3-5 bullet points, ~100-150 words) that describes:
- The main themes and topics covered
- Any notable trends or patterns
- Specific coverage examples generating and driving these insights

GUIDELINES:
- Keep the summary factual and focused on what the data shows.
- Do not make up ANY information, numbers or figures.
- Use only what is in the provided data to generate insights."""
    return call_claude(client, prompt, max_tokens=300)


def generate_executive_summary(client, summaries: dict[str, str]) -> str:
    """Synthesize individual BU summaries into an executive overview."""
    summaries_text = "\n\n".join(
        f"{name.replace('_', ' ').upper()}:\n{summary}"
        for name, summary in summaries.items()
    )
    prompt = f"""You are creating an executive summary of media coverage of GoDaddy across all business units and coverage areas.

Below are summaries of coverage from each area:

{summaries_text}

Please create a concise executive summary (4-6 bullet points, ~150-200 words) that:
- Synthesizes the key themes across all coverage areas
- Highlights the most significant trends or patterns
- Provides a holistic view of the organization's media presence
- Notes any notable differences or commonalities between the coverage areas
- Provides specific coverage pieces generating and driving these insights

Keep the summary factual and focused on what the data showed."""
    return call_claude(client, prompt, max_tokens=400)


def generate_overall_insights(client, summaries: dict[str, str]) -> str:
    """Generate high-level strategic insights from the full set of section summaries."""
    summaries_text = "\n\n".join(
        f"{name.replace('_', ' ').upper()}:\n{summary}"
        for name, summary in summaries.items()
    )
    prompt = f"""You are a senior media strategist reviewing ALL coverage insights generated for GoDaddy across every business unit and coverage area.

Below are the individual section summaries that were produced:

{summaries_text}

Please generate THREE to FIVE high-level strategic insights that synthesize
patterns across the entire grid. Format your response as bullet points
starting with asterisks (*).

Each insight should:
- Be one concise sentence or two short sentences
- Surface cross-cutting patterns, trends, or narrative arcs that span
  multiple coverage areas
- Reference specific themes, product launches, campaigns, or executive
  visibility where relevant
- Call out opportunities or risks visible only at the aggregate level
  (e.g. narrative concentration, coverage gaps, message alignment)

Do NOT simply repeat individual section summaries — your value is in
connecting the dots between them."""
    return call_claude(client, prompt, max_tokens=500)


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Reference")
    st.markdown("**Insight sections generated:**")
    for _key, label, _src in BUSINESS_UNITS:
        st.caption(label)
    st.divider()
    st.caption(
        "Sheets are matched by name (case-insensitive substring), "
        "so column order or extra tabs won't break parsing."
    )

# ──────────────────────────────────────────────────────────────
# Main area — file upload + run
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your GoDaddy IC Data workbook (.xlsx)", type=["xlsx", "xls"]
)

if uploaded_file:
    datasets = load_and_process(uploaded_file.read())

    # Quick stats
    st.subheader("📋 Dataset Overview")
    display_units = [(k, l) for k, l, _ in BUSINESS_UNITS]
    stat_cols = st.columns(len(display_units))
    for col, (key, label) in zip(stat_cols, display_units):
        n = len(datasets.get(key, []))
        col.metric(label.split(" ", 1)[1], f"{n:,} rows")

    st.divider()

    if st.button("🚀 Generate Insights", type="primary", use_container_width=True):
        # ── Bedrock client ──
        try:
            bedrock_runtime = boto3.client(
                "bedrock-runtime",
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
        except Exception as e:
            st.error(f"Failed to connect to AWS Bedrock: {e}")
            st.stop()

        summaries: dict[str, str] = {}
        total_steps = len(BUSINESS_UNITS) + 2  # +1 exec summary, +1 overall

        # ── Business-unit summaries ──
        st.subheader("📊 Coverage Area Summaries")
        progress = st.progress(0, text="Starting analysis …")

        for idx, (unit_key, unit_label, _sources) in enumerate(BUSINESS_UNITS):
            progress.progress(
                idx / total_steps,
                text=f"Summarizing {unit_label} …",
            )
            df = datasets[unit_key]
            if df.empty:
                summaries[unit_key] = "_No coverage data for this period._"
                with st.expander(unit_label, expanded=False):
                    st.markdown("_No coverage data for this period._")
                continue

            summary = generate_coverage_summary(
                bedrock_runtime,
                df,
                unit_label.split(" ", 1)[1],  # drop emoji prefix
            )
            summaries[unit_key] = summary
            with st.expander(unit_label, expanded=False):
                st.markdown(summary)

        # ── Executive summary ──
        progress.progress(
            len(BUSINESS_UNITS) / total_steps,
            text="Generating executive summary …",
        )
        st.subheader("📋 Executive Summary")
        exec_summary = generate_executive_summary(bedrock_runtime, summaries)
        st.markdown(exec_summary)

        # ── Overall insights (synthesized from section summaries) ──
        progress.progress(
            (len(BUSINESS_UNITS) + 1) / total_steps,
            text="Generating overall insights …",
        )
        st.subheader("💡 Overall Insights")
        insights = generate_overall_insights(bedrock_runtime, summaries)
        st.markdown(insights)

        progress.progress(1.0, text="✅ Analysis complete!")

else:
    st.info("Upload the GoDaddy IC Data workbook to get started.")
