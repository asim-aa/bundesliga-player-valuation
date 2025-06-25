#!/usr/bin/env python3
"""
Script: fetch_birthdays_from_csv.py

Hard-coded to read:
  /Users/asim/bundesliga-player-valuation/data/processed/players_clean.csv
Extracts the 'name' column, looks each up on Wikipedia, and prints:
  Name: Date of Birth
"""

import pandas as pd
import wptools
import time
import sys
import os

# 1) Hard-coded CSV path and column
CSV_PATH    = "/Users/asim/bundesliga-player-valuation/data/processed/players_clean.csv"
NAME_COLUMN = "name"

def fetch_dob(name):
    """
    Fetches the first matching birth-date field from the player's infobox.
    Returns either a date string or an error message.
    """
    try:
        # silent=True suppresses wptools debug prints
        page = wptools.page(name, silent=True).get_parse()
    except Exception as e:
        return f"❌ error fetching page ({e})"
    # Ensure infobox is at least a dict
    infobox = page.data.get('infobox') or {}
    # Look for the usual keys
    return (
        infobox.get('birth_date')
        or infobox.get('born')
        or infobox.get('birth_date_and_age')
        or "❌ not found"
    )

def main():
    # 2) Load the CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Failed to read '{CSV_PATH}': {e}", file=sys.stderr)
        sys.exit(1)

    if NAME_COLUMN not in df.columns:
        print(f"❌ Column '{NAME_COLUMN}' not found in {CSV_PATH}", file=sys.stderr)
        print("Available columns:", ", ".join(df.columns), file=sys.stderr)
        sys.exit(1)

    # 3) Extract player names
    names = df[NAME_COLUMN].dropna().astype(str).tolist()
    if not names:
        print(f"❌ No names found in column '{NAME_COLUMN}'", file=sys.stderr)
        sys.exit(1)

    # 4) Fetch & print
    for name in names:
        dob = fetch_dob(name)
        print(f"{name}: {dob}")

if __name__ == "__main__":
    main()
