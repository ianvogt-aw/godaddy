"""
T1 / Non-T1 unique story counts by tab and month  (flat CSV format, e.g. 2025 data)
==================================================================================
Companion to count_t1_by_month.py for the single-CSV "Intelligence Center" export,
where each "tab" is a flag column (non-blank = the story belongs to that tab) and a
story can sit on several tabs.

  * T1      = "Top Tier" column equals "Y" (--t1-values to change; "*" = any non-blank)
  * Non-T1  = everything else
  * Tab     = any non-blank value in that tab's column
  * Dates outside --year are ignored; rows with an unparseable Date are reported and skipped
  * Unique by URL (falls back to Document ID when URL is blank):
      - per tab: a repeated URL counts once, in the month of its earliest date;
        T1 if any copy is T1
      - GENERAL: every unique URL across the whole file (including rows that sit on no tab)

Usage:
    py count_t1_by_month_csv.py                      # 2025 tab set
    py count_t1_by_month_csv.py --year 2026          # 2026 tab set + ANS PAGE csv as "ANS Open Standard"
    py count_t1_by_month_csv.py "path\\to\\file.csv" --year 2025 --end-month 9
    py count_t1_by_month_csv.py file.csv --tabs Domain Partners Brand --output out.xlsx
"""

import argparse
import calendar
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

DEFAULT_CSV = Path(__file__).with_name("GoDaddy Intelligence Center - NEW Data.csv")
# Tabs are flag columns; "Column=Label" renames a column in the report.
TABS_2025 = ["Domain", "Partners", "Independents", "Commerce", "Brand", "Finance",
             "Aman Bhutani", "Gourav Pani", "Kasturi Mudulodu", "Fara Howard", "Mark McCaffrey"]
TABS_2026 = ["SBRL=GDSBRL", "Commerce", "AGI", "Airo", "ANS", "Other", "Brand", "Finance",
             "Aman Bhutani", "Gourav Pani", "Kasturi Mudulodu", "Mark McCaffrey", "Jared Sine",
             "Travis Mulestein"]
DEFAULT_ANS_CSV = Path(__file__).with_name("GoDaddy Intelligence Center - ANS PAGE.csv")


def nonblank(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace("\xa0", " ").str.strip().ne("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--end-month", type=int, default=None, help="Default: last month with data in --year")
    ap.add_argument("--tabs", nargs="*", default=None,
                    help='Flag columns to report as tabs ("Col=Label" to rename). Default depends on --year')
    ap.add_argument("--extra-csv", nargs="*", default=None, metavar="FILE=TAB",
                    help='Extra CSVs where every row belongs to one tab, e.g. "ans.csv=ANS Open Standard". '
                         "Default for 2026: the ANS PAGE csv as 'ANS Open Standard'")
    ap.add_argument("--tier-col", default="Top Tier")
    ap.add_argument("--t1-values", nargs="*", default=["Y"],
                    help='Top Tier values that count as T1 (default: Y, matches Looker). '
                         'Pass --t1-values "*" to treat any non-blank value as T1')
    ap.add_argument("--dedup", action="store_true",
                    help="Count unique URLs instead of raw rows (default: raw row counts)")
    ap.add_argument("--output")
    args = ap.parse_args()

    src = Path(args.csv)
    if not src.exists():
        sys.exit(f"File not found: {src}")
    out = Path(args.output) if args.output else src.with_name(f"{'unique' if args.dedup else 'raw'}_counts_{args.year}.xlsx")

    specs = args.tabs if args.tabs is not None else (TABS_2026 if args.year >= 2026 else TABS_2025)
    tab_cols = [(sp.split("=", 1)[0], sp.split("=", 1)[-1]) for sp in specs]   # (column, label)
    extra_specs = args.extra_csv
    if extra_specs is None:
        extra_specs = [f"{DEFAULT_ANS_CSV}=ANS Open Standard"] if args.year >= 2026 else []
    extras = []   # (path, label)
    for sp in extra_specs:
        f, label = sp.rsplit("=", 1)
        if not Path(f).exists():
            sys.exit(f"Extra CSV not found: {f}")
        extras.append((Path(f), label))

    def load(path):
        d = pd.read_csv(path, dtype=str, low_memory=False, encoding_errors="replace")
        miss = [c for c in ["Date", "URL", args.tier_col] if c not in d.columns]
        if miss:
            sys.exit(f"{path.name}: missing columns {miss}")
        return d

    df = load(src)
    missing = [c for c, _ in tab_cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}")
    df["_src"] = ""
    for path, label in extras:       # extra rows: every row belongs to that tab
        e = load(path)
        e["_src"] = label
        print(f"{path.name}: {len(e)} rows -> '{label}'")
        df = pd.concat([df, e], ignore_index=True)

    dates = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    bad = dates.isna().sum()
    if bad:
        print(f"Note: {bad} rows with unparseable/blank Date skipped: {df.loc[dates.isna(), 'Date'].head(5).tolist()}")
    df["_date"] = dates
    in_year = df["_date"].dt.year == args.year
    end_month = args.end_month or int(df.loc[in_year, "_date"].dt.month.max())
    months = list(range(args.start_month, end_month + 1))
    df = df[(df["_date"].dt.year == args.year) & df["_date"].dt.month.isin(months)].copy()
    print(f"{len(df)} rows in {args.year} months {months[0]}-{months[-1]}")

    # unique key: URL, else Document ID, else row-unique
    url = df["URL"].fillna("").astype(str).str.strip().str.lower().str.rstrip("/")
    did = df["Document ID"].fillna("").astype(str).str.strip().str.strip('"') if "Document ID" in df else ""
    key = url.where(url != "", "docid:" + did)
    key = key.where(key != "docid:", "__row_" + df.index.astype(str))
    df["_key"] = key
    if args.t1_values == ["*"]:
        df["_t1"] = nonblank(df[args.tier_col])
    else:
        keep = {v.strip().upper() for v in args.t1_values}
        df["_t1"] = df[args.tier_col].fillna("").astype(str).str.replace(" ", " ").str.strip().str.upper().isin(keep)
        odd = df[nonblank(df[args.tier_col]) & ~df["_t1"]]
        if len(odd):
            print(f"Note: {len(odd)} rows have a non-blank {args.tier_col} that is not {args.t1_values} "
                  f"(counted Non-T1): {odd[args.tier_col].value_counts().head(3).to_dict()}")
    df["_month"] = df["_date"].dt.month

    def tally(sub):
        if not args.dedup:   # raw rows
            c = {m: [0, 0] for m in months}
            for m, t1 in zip(sub["_month"], sub["_t1"]):
                c[m][0 if t1 else 1] += 1
            return c, len(sub)
        # one record per unique key: earliest date's month, T1 if any copy is T1
        g = sub.groupby("_key").agg(date=("_date", "min"), t1=("_t1", "any"))
        g["month"] = g["date"].dt.month
        c = {m: [0, 0] for m in months}
        for m, t1 in zip(g["month"], g["t1"]):
            c[m][0 if t1 else 1] += 1
        return c, len(g)

    labels = [calendar.month_abbr[m] for m in months]
    rows, summary = [], []
    tab_masks = [(label, nonblank(df[col])) for col, label in tab_cols]
    tab_masks += [(label, df["_src"] == label) for _, label in extras]
    for tab, mask in tab_masks:
        sub = df[mask]
        c, n_unique = tally(sub)
        summary.append((tab, len(sub), n_unique, len(sub) - n_unique))
        for ti, tier in enumerate(["T1", "Non-T1"]):
            rows.append((tab, tier, [c[m][ti] for m in months]))
    # General = the Intelligence Center rows; the extra CSVs' rows are already in it.
    gdf = df[df["_src"] == ""]
    gc, g_unique = tally(gdf)
    gen_name = "GENERAL (unique overall)" if args.dedup else "GENERAL (all rows)"
    gen_rows = [(gen_name, tier, [gc[m][ti] for m in months])
                for ti, tier in enumerate(["T1", "Non-T1"])]
    summary.append(("GENERAL (all rows)", len(gdf), g_unique, len(gdf) - g_unique))
    no_tab = int((~pd.concat([m for _, m in tab_masks], axis=1).any(axis=1)).sum())

    w = max(len(t) for t, _ in tab_masks + [("GENERAL (unique overall)", None)]) + 2
    hdr = f"{'Tab':<{w}}{'Tier':<8}" + "".join(f"{l:>7}" for l in labels) + f"{'Total':>8}"
    print(f"\n{"Unique stories by URL" if args.dedup else "Raw row counts"}, {args.year}\n\n{hdr}")
    for tab, tier, vals in rows:
        print(f"{tab:<{w}}{tier:<8}" + "".join(f"{v:>7}" for v in vals) + f"{sum(vals):>8}")
    print("-" * len(hdr))
    for tab, tier, vals in gen_rows:
        print(f"{tab:<{w}}{tier:<8}" + "".join(f"{v:>7}" for v in vals) + f"{sum(vals):>8}")
    print(f"\nRows on none of the listed tabs (General only): {no_tab}")

    wb = Workbook()
    ws = wb.active
    ws.title = "Unique by Month" if args.dedup else "Raw by Month"
    ws.append(["Tab", "Tier"] + labels + ["Total"])
    for tab, tier, vals in rows:
        ws.append([tab, tier] + vals + [sum(vals)])
    ws.append([])
    for tab, tier, vals in gen_rows:
        ws.append([tab, tier] + vals + [sum(vals)])
    n = ws.max_row
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F4E78")
        c.alignment = Alignment(horizontal="center")
    for r in (n - 1, n):
        for c in ws[r]:
            c.font = Font(bold=True)
    ws.column_dimensions["A"].width = w + 2
    ws.freeze_panes = "C2"
    ws.append([])
    ws.append([f"Source: {src.name}, {args.year}. T1 = '{args.tier_col}' in {args.t1_values}. "
               "per-tab rows dedupe within the tab, GENERAL dedupes across the whole file." if args.dedup
               else "Raw row counts, no de-duplication; a row counts on every tab flagged for it."])
    ws2 = wb.create_sheet("Dedup summary")
    ws2.append(["Tab", "Rows (pre-dedup)", "Unique URLs", "Duplicates removed"])
    for r in summary:
        ws2.append(list(r))
    for c in ws2[1]:
        c.font = Font(bold=True)
    ws2.column_dimensions["A"].width = w + 2
    for col in "BCD":
        ws2.column_dimensions[col].width = 22
    wb.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
