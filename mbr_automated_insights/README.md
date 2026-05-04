# 📊 Media Coverage Insights Generator

A Streamlit app that analyzes GoDaddy media coverage data from an Excel workbook and generates AI-powered summaries and insights using Claude.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run mbr_insights_app.py
```

That's it. The app opens in your browser.

## How to Use

1. **Paste your Anthropic API key** in the sidebar (it stays in-memory only).
2. **Set the date range** for the analysis period.
3. **Upload your Excel workbook** — it must have at least 15 sheets in this order:
   - legend, small_business_research_lab, commerce, agi, airo, ans, other, brand, finance, aman_bhutani, gourav_pani, kasturu_mudulodu, mark_mccaffrey, jared_sine, general
4. **Click "Generate Insights"** and wait ~30 seconds for all summaries to complete.

## What It Produces

- **6 business-unit summaries** (Small Business Research Lab, Product, Brand, Executive, Financial, Corporate)
- **1 executive summary** synthesizing all units
- **3 strategic insights** across the full dataset
