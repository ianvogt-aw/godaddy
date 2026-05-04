import streamlit as st
import pandas as pd
import json
import boto3
from datetime import datetime, date

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Media Coverage Insights",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Media Coverage Insights Generator")
st.markdown("Upload your Excel workbook, pick a date range, and let Claude generate business-unit summaries, an executive summary, and strategic insights — all in one click.")

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
def load_and_process(file_bytes, start_date, end_date):
    """Read the uploaded Excel, filter by date range, return combined datasets."""
    xls = pd.ExcelFile(file_bytes)
    actual_sheets = xls.sheet_names

    if len(actual_sheets) < len(SHEET_NAMES):
        st.error(
            f"Expected at least {len(SHEET_NAMES)} sheets but found {len(actual_sheets)}. "
            "Please upload the correct workbook."
        )
        st.stop()

    dataframes = {}
    for i, name in enumerate(SHEET_NAMES):
        dataframes[name] = pd.read_excel(file_bytes, sheet_name=actual_sheets[i], header=0)

    # Convert Date columns
    for name, df in dataframes.items():
        if "Date" in df.columns:
            dataframes[name]["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filter by date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for name, df in dataframes.items():
        if "Date" in df.columns:
            dataframes[name] = df[(df["Date"] >= start) & (df["Date"] <= end)]

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
# AWS Bedrock configuration
# ──────────────────────────────────────────────────────────────
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


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


def generate_coverage_summary(client, df, unit_name, date_range):
    coverage_data = df[["Date", "Title", "Hit Sentence"]].to_string(index=False)
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


def generate_executive_summary(client, summaries, date_range):
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


def generate_overall_insights(client, all_coverage_df, date_range):
    coverage_data = all_coverage_df[["Date", "Title", "Hit Sentence"]].to_string(index=False)
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
# Sidebar — configuration
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("AWS Credentials")
    aws_region = st.text_input("AWS Region", value="us-east-2")
    aws_access_key = st.text_input("AWS Access Key ID", type="password")
    aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
    st.caption("Leave credentials blank to use your default AWS profile (~/.aws/credentials or env vars).")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date(2026, 3, 1))
    with col2:
        end_date = st.date_input("End date", value=date(2026, 3, 31))
    st.divider()
    st.caption("**Expected sheets (first 15, in order):** " + ", ".join(SHEET_NAMES))

# ──────────────────────────────────────────────────────────────
# Main area — file upload + run
# ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your Excel workbook (.xlsx)", type=["xlsx", "xls"])

# Determine if AWS credentials are available (explicit or default profile)
aws_ready = bool(aws_region)

if uploaded_file and aws_ready:
    datasets = load_and_process(uploaded_file.read(), start_date, end_date)

    # Show quick stats
    st.subheader("📋 Dataset Overview")
    stat_cols = st.columns(len(datasets))
    for col, (name, df) in zip(stat_cols, datasets.items()):
        col.metric(name.replace("_", " ").title(), f"{len(df)} rows")

    st.divider()

    if st.button("🚀 Generate Insights", type="primary", use_container_width=True):
        # Build the Bedrock client, matching the original notebook
        try:
            if aws_access_key and aws_secret_key:
                bedrock_runtime = boto3.client(
                    "bedrock-runtime",
                    region_name=aws_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                )
            else:
                bedrock_runtime = boto3.client(
                    "bedrock-runtime",
                    region_name=aws_region,
                )
        except Exception as e:
            st.error(f"Failed to connect to AWS Bedrock: {e}")
            st.stop()

        date_range_text = f"{start_date} to {end_date}"
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
                date_range_text,
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
        exec_summary = generate_executive_summary(bedrock_runtime, summaries, date_range_text)
        st.markdown(exec_summary)

        # ── Overall insights ──
        progress.progress(
            (len(BUSINESS_UNITS) + 1) / (len(BUSINESS_UNITS) + 2),
            text="Generating overall insights …",
        )
        st.subheader("💡 Overall Insights")
        insights = generate_overall_insights(bedrock_runtime, datasets["all_coverage"], date_range_text)
        st.markdown(insights)

        progress.progress(1.0, text="✅ Analysis complete!")

elif not uploaded_file:
    st.info("Upload an Excel workbook to get started.")
