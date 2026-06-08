# 📊 Media Coverage Insights Generator

A Streamlit app that analyzes GoDaddy media coverage data from the IC Data workbook and generates AI-powered summaries and insights using Claude on AWS Bedrock.

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

1. **Upload the workbook** — no manual prep needed. Sheets are matched by name (case-insensitive substring), so exact ordering and extra tabs won't break parsing.
2. **Red-highlighted rows are auto-discarded** — any row whose Date cell has a red fill (`#FF0000`) is dropped before analysis, eliminating the need to manually delete scrubbed/old coverage.
3. **Filter to a date range** (optional) — after upload, use the date pickers to narrow coverage to a specific month or period. Stats update live.
4. **Click "Generate Insights"** and wait for all summaries to complete.

## What It Produces

**7 coverage-area summaries:**

| Section | Source sheets |
|---|---|
| 🔬 Small Business Research Lab | GDSBRL |
| 🏭 Product | Commerce, AGI, Airo, ANS, Other |
| 🌐 ANS Open Standard | ANS Open Standard |
| 🎨 Brand | Brand |
| 👔 Thought Leadership | Aman Bhutani, Gourav Pani, Kasturi Mudulodu, Mark McCaffrey, Jared Sine, Travis Muhlestein, Demetria, Berea Schaffer |
| 💰 Financial | Finance |
| 🏢 Corporate | Brand + all Thought Leadership names + Finance |

**1 executive summary** synthesizing all coverage areas

**3–5 overall strategic insights** synthesized from the section summaries (replaces the former General-tab-based insights)

## Sheet Matching

The app uses case-insensitive substring matching to identify sheets, so minor naming changes across yearly files (e.g. `ANS (Product) + Int.` vs `ANS + Int.`) are handled automatically. Unrecognized sheets are silently skipped; missing expected sheets produce a warning and are treated as empty.
