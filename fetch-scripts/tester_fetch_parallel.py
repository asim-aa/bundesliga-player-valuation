#!/usr/bin/env python3
"""
tester_fetch_parallel.py

- Reads /Users/asim/bundesliga-player-valuation/data/processed/players_clean.csv
- Parallelizes Wikipedia queries for each name (max 8 threads)
- Suppresses all internal API-error printouts
- Writes a CSV 'birthdays_output.csv' next to this script
"""

import os
import sys
import io
import pandas as pd
import wptools
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr

# CONFIG
CSV_INPUT    = "/Users/asim/bundesliga-player-valuation/data/processed/players_clean.csv"
NAME_COLUMN  = "name"
MAX_WORKERS  = 8
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV   = os.path.join(SCRIPT_DIR, "birthdays_output.csv")

def fetch_dob(name):
    """
    Fetches 'birth_date' (or 'born') from the player's Wikipedia infobox.
    Silences all internal wptools prints by redirecting stdout/stderr.
    Returns: (name, dob_string)
    """
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            page = wptools.page(name, silent=True).get_parse()
        infobox = page.data.get("infobox") or {}
        dob = (
            infobox.get("birth_date")
            or infobox.get("born")
            or infobox.get("birth_date_and_age")
            or "❌ not found"
        )
    except Exception as e:
        dob = f"❌ error ({e})"
    return name, dob

def main():
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

    # 2) parallel fetch (and preserve order)
    results = [None] * len(names)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        future_to_idx = {
            exe.submit(fetch_dob, nm): idx
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

    # 3) write CSV
    out_df = pd.DataFrame(results, columns=["name", "date_of_birth"])
    try:
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Wrote {len(results)} entries to: {OUTPUT_CSV}")
    except Exception as e:
        print(f"❌ Failed to write output CSV: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
