"""
T1 / Non-T1 unique story counts by tab and month
================================================
Reads one or more IC Data workbooks and counts UNIQUE stories (by URL) per tab,
per month, split into T1 and Non-T1:

  * T1      = Custom Categories column equals "Y"  (CISION grid), or holds a category tag
              such as "US T1 2026 Product" (older Meltwater-style grid) -> any non-blank value
  * Non-T1  = Custom Categories column is blank (whitespace / non-breaking space = blank)
  * Rows with a RED fill (Legend: "Removed") are discarded
  * Rows with no Date are ignored

De-duplication (by URL; falls back to Document ID if the URL is blank):
  * Per tab: a URL repeated within a tab (including across source files) counts once,
    in the month of its earliest date. It is T1 if any copy is marked Y.
  * GENERAL (overall): a URL that shows up on several tabs counts once, in its earliest
    month, and is T1 if any tab marks it Y. Tabs still each count it individually.

Sources: pass --source FILE:START-END for each workbook (month range inclusive).
Tabs are matched across files by name.

Usage:
    py count_t1_by_month.py ^
        --source "GoDaddy IC Data - 2026 NEW.xlsx:1-8" ^
        --source "GoDaddy_IC_Data_2026_CISION.xlsx:9-9" ^
        --output unique_counts_JanSep.xlsx

    # Single file, default Jan-Sep:
    py count_t1_by_month.py --source older_version.xlsx

Writes <first source>_T1_counts.xlsx next to the first source unless --output is given.
"""

import argparse
import calendar
import re
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

# Helper / non-story tabs (never counted); Hist_Data duplicates other tabs.
ALWAYS_SKIP = {"Legend", "Hist_Calc", "Hist_Stage", "Hist_Data"}


def is_red(cell) -> bool:
    f = cell.fill
    if f is None or f.fill_type != "solid":
        return False
    c = f.fgColor
    return c is not None and c.type == "rgb" and isinstance(c.rgb, str) and c.rgb.upper().endswith("FF0000")


def norm_key(url, doc_id, unique_counter):
    u = str(url).strip().lower() if url else ""
    if u:
        return u.rstrip("/")
    if doc_id:
        return "docid:" + str(doc_id).strip()
    unique_counter[0] += 1
    return f"__blank_{unique_counter[0]}"


def parse_source(s, default_range):
    m = re.match(r"^(.*):(\d{1,2})(?:-(\d{1,2}))?$", s)
    if m and not re.match(r"^[A-Za-z]$", m.group(1)):   # don't split the drive letter
        a = int(m.group(2))
        b = int(m.group(3) or a)
        return Path(m.group(1)), a, b
    return Path(s), *default_range


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="FILE[:START-END] (months inclusive). Repeat for multiple files.")
    ap.add_argument("--start-month", type=int, default=1, help="Overall report start (default 1)")
    ap.add_argument("--end-month", type=int, default=9, help="Overall report end (default 9)")
    ap.add_argument("--year", type=int, default=2026, help="Only count this year (0 = any)")
    ap.add_argument("--output", help="Output .xlsx path")
    ap.add_argument("--alias", nargs="*", default=[], metavar="SRC=DEST",
                    help='Merge a tab into another name, e.g. --alias "Brand Identity=Brand Identity (v2)"')
    ap.add_argument("--exclude", nargs="*", default=[], help="Extra tab names to skip")
    args = ap.parse_args()

    months = list(range(args.start_month, args.end_month + 1))
    year = args.year or None
    sources = [parse_source(s, (args.start_month, args.end_month)) for s in args.source]
    for p, a, b in sources:
        if not p.exists():
            sys.exit(f"File not found: {p}")
    out = Path(args.output) if args.output else sources[0][0].with_name(sources[0][0].stem + "_T1_counts.xlsx")
    aliases = dict(a.split("=", 1) for a in args.alias)
    skip = ALWAYS_SKIP | set(args.exclude)

    blank_ctr = [0]
    tabs = {}       # tab -> {key: [earliest_date, is_t1]}
    general = {}    # key -> [earliest_date, is_t1]
    raw_rows = {}   # tab -> kept (non-red, in-range) rows before dedup
    red_rows = 0
    odd_values = {}  # non-blank Custom Categories that are neither "Y" nor contain "T1"

    for path, m_from, m_to in sources:
        print(f"Loading {path.name} (months {m_from}-{m_to}) ...")
        wb = openpyxl.load_workbook(path, read_only=True)
        for ws in wb:
            if ws.title in skip:
                continue
            rows = ws.iter_rows()
            header = [c.value for c in next(rows, [])]
            if "Date" not in header or "Custom Categories" not in header:
                print(f"  (skipping '{ws.title}': no Date / Custom Categories columns)")
                continue
            di, ci = header.index("Date"), header.index("Custom Categories")
            ui = header.index("URL") if "URL" in header else None
            ii = header.index("Document ID") if "Document ID" in header else None
            tname = aliases.get(ws.title, ws.title)
            tab = tabs.setdefault(tname, {})
            n_raw = 0
            for row in rows:
                if len(row) <= max(di, ci):
                    continue
                d = row[di].value
                if not hasattr(d, "month"):
                    continue
                if (year and d.year != year) or not (m_from <= d.month <= m_to):
                    continue
                if is_red(row[0]):
                    red_rows += 1
                    continue
                n_raw += 1
                cat = row[ci].value
                cat = str(cat).replace(" ", " ").strip() if cat is not None else ""
                t1 = cat != ""
                if t1 and cat.upper() != "Y" and "T1" not in cat.upper():
                    odd_values[cat] = odd_values.get(cat, 0) + 1
                key = norm_key(row[ui].value if ui is not None and ui < len(row) else None,
                               row[ii].value if ii is not None and ii < len(row) else None, blank_ctr)
                for store in (tab, general):
                    cur = store.get(key)
                    if cur is None:
                        store[key] = [d, t1]
                    else:
                        cur[0] = min(cur[0], d)
                        cur[1] = cur[1] or t1
            raw_rows[tname] = raw_rows.get(tname, 0) + n_raw
        wb.close()

    def tally(store):
        c = {m: [0, 0] for m in months}
        for d, t1 in store.values():
            if d.month in c:
                c[d.month][0 if t1 else 1] += 1
        return c

    labels = [calendar.month_abbr[m] for m in months]
    rows = []
    for tab, store in tabs.items():
        c = tally(store)
        for ti, tier in enumerate(["T1", "Non-T1"]):
            rows.append((tab, tier, [c[m][ti] for m in months]))
    gc = tally(general)
    gen_rows = [("GENERAL (unique overall)", tier, [gc[m][ti] for m in months])
                for ti, tier in enumerate(["T1", "Non-T1"])]

    w = max([len(t) for t in tabs] + [26]) + 2
    print(f"\nUnique stories by URL (red rows discarded), {labels[0]}-{labels[-1]} {year or ''}\n")
    hdr = f"{'Tab':<{w}}{'Tier':<8}" + "".join(f"{l:>7}" for l in labels) + f"{'Total':>8}"
    print(hdr)
    for tab, tier, vals in rows + [None] * 0:
        print(f"{tab:<{w}}{tier:<8}" + "".join(f"{v:>7}" for v in vals) + f"{sum(vals):>8}")
    print("-" * len(hdr))
    for tab, tier, vals in gen_rows:
        print(f"{tab:<{w}}{tier:<8}" + "".join(f"{v:>7}" for v in vals) + f"{sum(vals):>8}")
    print(f"\nRed rows discarded: {red_rows}")

    # Excel output
    out_wb = openpyxl.Workbook()
    ws = out_wb.active
    ws.title = "Unique by Month"
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
    ws.append(["Sources: " + "; ".join(f"{p.name} (months {a}-{b})" for p, a, b in sources)])
    ws.append(["Unique by URL. T1 = Custom Categories 'Y'; Non-T1 = blank. Red rows discarded. "
               "Per-tab rows dedupe within the tab; GENERAL dedupes across all tabs."])

    ws2 = out_wb.create_sheet("Dedup summary")
    ws2.append(["Tab", "Rows kept (non-red, pre-dedup)", "Unique URLs", "Duplicates removed"])
    for tab, store in tabs.items():
        ws2.append([tab, raw_rows.get(tab, 0), len(store), raw_rows.get(tab, 0) - len(store)])
    for c in ws2[1]:
        c.font = Font(bold=True)
    ws2.column_dimensions["A"].width = w + 2
    for col in "BCD":
        ws2.column_dimensions[col].width = 28
    out_wb.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
