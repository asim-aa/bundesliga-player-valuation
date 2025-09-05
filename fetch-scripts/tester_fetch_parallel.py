"""
Fetch player dates of birth from Wikipedia in parallel and save to CSV.

What it does:
- Reads player names from a CSV (column 'name').
- Queries Wikipedia (via wptools) for each player's infobox.
- Extracts and normalizes birth date strings to DD-MM-YYYY when possible.
- Writes results to 'birthdays_output.csv' alongside this script.

Usage:
  python fetch-scripts/tester_fetch_parallel.py

Notes:
- Requires the 'wptools' package and internet access.
- Update CSV_INPUT below if your processed players CSV is located elsewhere.
"""
import os
import sys
import io
import re
import pandas as pd
import wptools
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr

# ───── CONFIG ────────────────────────────────────────────────────────────────
# Absolute path to the input CSV containing a 'name' column.
CSV_INPUT    = "/Users/asim/bundesliga-player-valuation/data/processed/players_clean.csv"
# Column name in CSV that holds player names.
NAME_COLUMN  = "name"
# Number of threads to use for concurrent requests.
MAX_WORKERS  = 8
# Directory for writing the output CSV (next to this script).
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, "birthdays_output.csv")

# Month name → number map for parsing textual dates.
_MONTHS = {
    'January': 1, 'Jan': 1,
    'February': 2, 'Feb': 2,
    'March': 3, 'Mar': 3,
    'April': 4, 'Apr': 4,
    'May': 5,
    'June': 6, 'Jun': 6,
    'July': 7, 'Jul': 7,
    'August': 8, 'Aug': 8,
    'September': 9, 'Sep': 9, 'Sept': 9,
    'October': 10, 'Oct': 10,
    'November': 11, 'Nov': 11,
    'December': 12, 'Dec': 12,
}


def _normalize_dob(raw: str) -> str:
    """
    Take raw infobox birth_date/born string and convert to 'DD-MM-YYYY'.
    If parsing fails, return raw string.
    """
    if not raw:
        return raw

    raw = raw.strip()

    # 1) Template format: {{birth date|YYYY|MM|DD}} or {{birth date and age|YYYY|MM|DD|...}}
    m = re.search(r'\{\{\s*birth date(?: and age)?\|(\d{4})\|(\d{1,2})\|(\d{1,2})', raw, re.IGNORECASE)
    if m:
        year, mon, day = m.group(1), m.group(2), m.group(3)
        return f"{int(day):02d}-{int(mon):02d}-{year}"

    # 2) ISO-like YYYY-MM-DD
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', raw)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{int(d):02d}-{int(mo):02d}-{y}"

    # 3) Textual format: 'DD Month YYYY' (possibly with '(age …)')
    #    Strip off any parenthetical age.
    raw = raw.split('(')[0].strip()
    m = re.match(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', raw)
    if m:
        d_str, mo_str, y_str = m.group(1), m.group(2), m.group(3)
        mo_num = _MONTHS.get(mo_str)
        if mo_num:
            return f"{int(d_str):02d}-{mo_num:02d}-{y_str}"

    # fallback
    return raw


def fetch_dob(name: str) -> tuple[str, str]:
    """
    Fetches raw 'birth_date' or 'born' from the Wikipedia infobox,
    normalizes it to DD-MM-YYYY, and returns (name, dob).
    Silences wptools internal prints.
    """
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            page = wptools.page(name, silent=True).get_parse()
        infobox = page.data.get("infobox") or {}
        # Prefer 'birth_date', then 'born', then the template variant.
        raw = (
            infobox.get("birth_date")
            or infobox.get("born")
            or infobox.get("birth_date_and_age")
            or ""
        )
        dob = _normalize_dob(raw)
    except Exception as e:
        dob = f"❌ error ({e})"
    return name, dob


def main():
    """Load names, fetch DOBs in parallel, and write the output CSV."""
    # 1) load names
    try:
        df = pd.read_csv(CSV_INPUT)
    except Exception as e:
        print(f"❌ Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if NAME_COLUMN not in df.columns:
        print(f"❌ Column '{NAME_COLUMN}' not found in {CSV_INPUT}", file=sys.stderr)
        sys.exit(1)

    names = df[NAME_COLUMN].dropna().astype(str).tolist()
    if not names:
        print(f"❌ No names found in column '{NAME_COLUMN}'", file=sys.stderr)
        sys.exit(1)

    # 2) Parallel fetch (preserve original order with index mapping).
    results = [None] * len(names)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(fetch_dob, nm): idx
            for idx, nm in enumerate(names)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                name, dob = fut.result()
            except Exception as e:
                name = names[idx]
                dob = f"❌ exception ({e})"
            results[idx] = (name, dob)

    # 3) Write to CSV
    out_df = pd.DataFrame(results, columns=["name", "date_of_birth"])
    try:
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Wrote {len(results)} entries to: {OUTPUT_CSV}")
    except Exception as e:
        print(f"❌ Failed to write output CSV: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
