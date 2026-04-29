# GoDaddy Mention Extractor — Streamlit Edition

Extracts the sentence before, the GoDaddy mention itself, and the sentence after
from each article URL in a Meltwater (or similar) CSV export.

## Repo layout

```
.
├── app.py               # Streamlit app
├── requirements.txt     # Python dependencies
├── packages.txt         # System packages (for Playwright on Community Cloud)
└── .streamlit/
    └── secrets.toml     # Local dev only — DO NOT commit
```

## Local development

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium        # one-time browser download
streamlit run app.py
```

Open http://localhost:8501.

## Deploying to Streamlit Community Cloud (free, private)

1. Push this repo to **GitHub** (can be private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, paste:

```toml
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
BEDROCK_REGION = "us-east-2"
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
```

   Skip the AWS keys entirely if you won't use the translation feature.

5. Click **Deploy**.
6. Once live, go to **Share → Manage access** and add your team's email addresses.
   They authenticate with Google/GitHub — no passwords, no installs.

## CSV format

The input CSV must have:
- A column of article URLs (specify the header name or column letter in the UI)
- Optionally a `Hit Sentence` column (from Meltwater) — used to find the right mention
- Optionally a `Language` column — used to decide which rows to translate

## Translation

When **Translate to English** is checked, rows where the `Language` column is
anything other than `english` will be sent to Claude (via AWS Bedrock) for
translation after extraction. The `Before:` / `Mention:` / `After:` labels are
preserved.

Requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to be set with
Bedrock access in the configured region.

## Fetch strategy

The app tries three methods per URL, falling back on failure:

1. **curl_cffi** — Chrome TLS fingerprint; covers most sites
2. **Playwright** — full headless Chromium; handles JS challenges
3. **requests** — plain HTTP last resort

Rows that can't be fetched are marked `ERROR: <reason>` in the output column.

## Notes

- Requests to the same domain are spaced at least 5 seconds apart to avoid bans.
- The general inter-request delay (0.5–5 s) is configurable in the UI.
- The app processes rows synchronously — for 500+ rows expect it to run for
  several minutes. Community Cloud has a 1 GB RAM limit; the app is 
  designed to stream row-by-row so memory usage stays flat.
