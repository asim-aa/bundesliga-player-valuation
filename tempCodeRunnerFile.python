#!/usr/bin/env python3
"""
Script: clean_data.py

Cleans raw Bundesliga players data and writes out a processed CSV.
"""
import pandas as pd
import numpy as np
import os
import sys

def parse_monetary(val):
    """Convert strings like "€12 M" or "€500 K" to a float (in euros)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Remove currency symbols
    s = s.replace('€', '').replace(',', '')
    multiplier = 1.0
    if s.endswith('M'):
        multiplier = 1e6
        s = s[:-1]
    elif s.endswith('K'):
        multiplier = 1e3
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan

def main():
    # 1. Load raw CSV (drop the auto-index column)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    raw_path = os.path.join(repo_root, 'data', 'raw', 'bundesliga_player.csv')
    if not os.path.exists(raw_path):
        sys.exit(f"Error: raw data not found at {raw_path}")
    df = pd.read_csv(raw_path, index_col=0)  

    # 2. Parse monetary strings
    df['price']     = df['price'].apply(parse_monetary)
    df['max_price'] = df['max_price'].apply(parse_monetary)

    # 3. Convert date columns to datetime
    for col in ['joined_club', 'contract_expires']:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # 4. Handle missing values
    #   a) Drop rows missing any core features
    df = df.dropna(subset=['price', 'max_price', 'age', 'position'])

    #   b) Impute optional numeric: height → median by position
    df['height'] = df.groupby('position')['height'] \
                     .transform(lambda series: series.fillna(series.median()))

    #   c) Fill categorical missing
    df['nationality']  = df['nationality'].fillna('Unknown')
    df['player_agent'] = df['player_agent'].fillna('Unknown')
    df['outfitter']    = df['outfitter'].fillna('Unknown')
    df['foot']         = df['foot'].fillna('Unknown')

    # 5. Standardize text fields (lowercase & strip)
    text_cols = ['club', 'position', 'nationality',
                 'player_agent', 'outfitter', 'foot']
    for col in text_cols:
        df[col] = df[col].str.strip().str.lower()

    # 6. Save cleaned CSV
    proc_dir = os.path.join(repo_root, 'data', 'processed')
    os.makedirs(proc_dir, exist_ok=True)
    out_path = os.path.join(proc_dir, 'players_clean.csv')
    df.to_csv(out_path, index=False)

    print(f"Cleaned data written to {out_path}")

if __name__ == '__main__':
    main()
