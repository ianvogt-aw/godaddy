# 📊 Media Coverage Insights Generator

A Streamlit app that analyzes GoDaddy media coverage data from an Excel workbook and generates AI-powered summaries and insights using Claude on AWS Bedrock.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run mbr_insights_app.py
```

That's it. The app opens in your browser.

## Setup (one-time)

Create a `.streamlit/secrets.toml` file next to `mbr_insights_app.py` (a template is included) and fill in your AWS credentials:

```toml
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "wJal..."
BEDROCK_REGION = "us-east-2"
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
```

If deploying on Streamlit Community Cloud, add these same four keys in the app's Secrets settings instead.

## How to Use

1. **Prepare your data** — save a copy of the GoDaddy Grid with only coverage data from the month of interest (delete scrubbed rows + old coverage, use sorting to make this easy).
2. **Upload your Excel workbook** — it must have at least 15 sheets in this order:
   - legend, small_business_research_lab, commerce, agi, airo, ans, other, brand, finance, aman_bhutani, gourav_pani, kasturu_mudulodu, mark_mccaffrey, jared_sine, general
3. **Click "Generate Insights"** and wait ~30 seconds for all summaries to complete.

## What It Produces

- **6 business-unit summaries** (Small Business Research Lab, Product, Brand, Executive, Financial, Corporate)
- **1 executive summary** synthesizing all units
- **3 strategic insights** across the full dataset
