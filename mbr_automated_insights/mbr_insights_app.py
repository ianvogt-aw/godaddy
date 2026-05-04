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
st.markdown("Upload your prepared Excel workbook and let Claude generate business-unit summaries, an executive summary, and strategic insights — all in one click.")

st.info(
    "**⚠️ Data Preparation Required:** This application assumes you are uploading the following: "
    "a version of the GoDaddy Grid with only relevant data. This means you must save a copy of "
    "the grid with only coverage data from the month of interest (delete scrubbed rows + old "
    "coverage, use sorting to make this easy)."
)

# ──────────────────────────────────────────────────────────────
# Hardcoded sheet → internal-name mapping (order matters)
# ──────────────────────────────────────────────────────────────
SHEET_NAMES = [
    "legend",
    "small_business_research_lab",
    "commerce",
    "agi",
    "airo",
    "ans",
    "other",
    "brand",
    "finance",
    "aman_bhutani",
    "gourav_pani",
    "kasturu_mudulodu",
    "mark_mccaffrey",
    "jared_sine",
    "general",
]

COLUMNS_TO_KEEP = ["Date", "Title", "Hit Sentence"]

BUSINESS_UNITS = [
    ("small_business_research_lab", "🔬 Small Business Research Lab"),
    ("product", "🏭 Product"),
    ("brand", "🎨 Brand"),
    ("executive", "👔 Executive"),
    ("financial", "💰 Financial"),
    ("corporate", "🏢 Corporate"),
]


# ──────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Excel file …")
def load_and_process(file_bytes):
    """Read the uploaded Excel and create combined datasets."""
    buf = BytesIO(file_bytes)
    xls = pd.ExcelFile(buf)
    actual_sheets = xls.sheet_names

    if len(actual_sheets) < len(SHEET_NAMES):
        st.error(
            f"Expected at least {len(SHEET_NAMES)} sheets but found {len(actual_sheets)}. "
            "Please upload the correct workbook."
        )
        st.stop()

    dataframes = {}
    for i, name in enumerate(SHEET_NAMES):
        dataframes[name] = pd.read_excel(buf, sheet_name=actual_sheets[i], header=0)

    # Convert Date columns
    for name, df in dataframes.items():
        if "Date" in df.columns:
            dataframes[name]["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Helper to safely select columns
    def cols(df):
        return df[[c for c in COLUMNS_TO_KEEP if c in df.columns]]

    # Build combined datasets
    small_business_research_lab = cols(dataframes["small_business_research_lab"])

    product = pd.concat(
        [cols(dataframes[n]) for n in ("commerce", "agi", "airo", "ans", "other")],
        ignore_index=True,
    )

    brand = cols(dataframes["brand"])

    executive = pd.concat(
        [cols(dataframes[n]) for n in ("aman_bhutani", "gourav_pani", "kasturu_mudulodu", "mark_mccaffrey", "jared_sine")],
        ignore_index=True,
    )

    financial = cols(dataframes["finance"])

    corporate = pd.concat(
        [cols(dataframes[n]) for n in ("brand", "aman_bhutani", "gourav_pani", "kasturu_mudulodu", "mark_mccaffrey", "finance", "general")],
        ignore_index=True,
    )

    all_coverage = pd.concat(
        [small_business_research_lab, product, brand, executive, financial, cols(dataframes["general"])],
        ignore_index=True,
    ).drop_duplicates()

    datasets = {
        "small_business_research_lab": small_business_research_lab,
        "product": product,
        "brand": brand,
        "executive": executive,
        "financial": financial,
        "corporate": corporate,
        "all_coverage": all_coverage,
    }

    return datasets


# ──────────────────────────────────────────────────────────────
# AWS Bedrock configuration  (read from .streamlit/secrets.toml)
# ──────────────────────────────────────────────────────────────
AWS_REGION = st.secrets["BEDROCK_REGION"]
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
CLAUDE_MODEL_ID = st.secrets["BEDROCK_MODEL_ID"]


# ──────────────────────────────────────────────────────────────
# LLM helpers  (matches original notebook's Bedrock calls)
# ──────────────────────────────────────────────────────────────
def call_claude(bedrock_client, prompt, max_tokens=400):
    """Invoke Claude via AWS Bedrock, matching the original notebook pattern."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
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


def generate_coverage_summary(client, df, unit_name):
    coverage_data = df[["Date", "Title", "Hit Sentence"]].to_string(index=False)
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}" if pd.notna(min_date) else "N/A"
    prompt = f"""You are analyzing media coverage data for the {unit_name} business unit of GoDaddy.

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


def generate_executive_summary(client, summaries):
    summaries_text = "\n\n".join(
        [f"{name.upper()}:\n{summary}" for name, summary in summaries.items()]
    )
    prompt = f"""You are creating an executive summary of media coverage of GoDaddy across all business units.

Below are summaries of coverage from each business unit:

{summaries_text}

Please create a concise executive summary (4-6 bullet points, ~150-200 words) that:
- Synthesizes the key themes across all business units
- Highlights the most significant trends or patterns
- Provides a holistic view of the organization's media presence
- Notes any notable differences or commonalities between the coverage areas
- Provides specific coverage pieces generating and driving these insights

Keep the summary factual and focused on what the data showed."""
    return call_claude(client, prompt, max_tokens=400)


def generate_overall_insights(client, all_coverage_df):
    coverage_data = all_coverage_df[["Date", "Title", "Hit Sentence"]].to_string(index=False)
    min_date = all_coverage_df["Date"].min()
    max_date = all_coverage_df["Date"].max()
    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}" if pd.notna(min_date) else "N/A"
    prompt = f"""You are analyzing ALL media coverage data for GoDaddy for the period {date_range}.

Total articles: {len(all_coverage_df)}

Here is the complete coverage data:
{coverage_data}

Please generate exactly THREE strategic insights about the overall media coverage. Format your response as three bullet points starting with asterisks (*).

Each insight should:
- Be one concise sentence or two short sentences
- Focus on high-level patterns, trends, or notable coverage drivers
- Include specific details like product launches, campaigns, research initiatives, or themes
- Reference specific coverage examples
- If relevant, mention volume changes, geographic reach, or coverage quality

Generate the three insights now, formatted as bullet points."""
    return call_claude(client, prompt, max_tokens=400)


# ──────────────────────────────────────────────────────────────
# Sidebar — reference info only
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Reference")
    st.caption("**Expected sheets (first 15, in order):** " + ", ".join(SHEET_NAMES))

# ──────────────────────────────────────────────────────────────
# Main area — file upload + run
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your Excel workbook (.xlsx)", type=["xlsx", "xls"])

if uploaded_file:
    datasets = load_and_process(uploaded_file.read())

    # Show quick stats
    st.subheader("📋 Dataset Overview")
    stat_cols = st.columns(len(datasets))
    for col, (name, df) in zip(stat_cols, datasets.items()):
        col.metric(name.replace("_", " ").title(), f"{len(df)} rows")

    st.divider()

    if st.button("🚀 Generate Insights", type="primary", use_container_width=True):
        # Build the Bedrock client with baked-in credentials
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

        summaries = {}

        # ── Business-unit summaries ──
        st.subheader("📊 Business Unit Summaries")
        progress = st.progress(0, text="Starting analysis …")

        for idx, (unit_key, unit_label) in enumerate(BUSINESS_UNITS):
            progress.progress(
                (idx) / (len(BUSINESS_UNITS) + 2),
                text=f"Summarizing {unit_label} …",
            )
            summary = generate_coverage_summary(
                bedrock_runtime,
                datasets[unit_key],
                unit_key.replace("_", " ").title(),
            )
            summaries[unit_key] = summary
            with st.expander(unit_label, expanded=False):
                st.markdown(summary)

        # ── Executive summary ──
        progress.progress(
            (len(BUSINESS_UNITS)) / (len(BUSINESS_UNITS) + 2),
            text="Generating executive summary …",
        )
        st.subheader("📋 Executive Summary")
        exec_summary = generate_executive_summary(bedrock_runtime, summaries)
        st.markdown(exec_summary)

        # ── Overall insights ──
        progress.progress(
            (len(BUSINESS_UNITS) + 1) / (len(BUSINESS_UNITS) + 2),
            text="Generating overall insights …",
        )
        st.subheader("💡 Overall Insights")
        insights = generate_overall_insights(bedrock_runtime, datasets["all_coverage"])
        st.markdown(insights)

        progress.progress(1.0, text="✅ Analysis complete!")

else:
    st.info("Upload an Excel workbook to get started.")
