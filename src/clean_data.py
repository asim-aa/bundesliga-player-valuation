#!/usr/bin/env python3
"""
Script: clean_data.py

Cleans the raw Bundesliga players dataset:
 - Parses monetary strings to numeric values
 - Converts date columns to datetime
 - Handles missing values
 - Standardizes text fields
 - Saves cleaned CSV to data/processed/players_clean.csv
"""
import os
import re
import pandas as pd

def parse_money(value):
    """Convert strings like '€12 M', '€650k' to numeric (float, in euros)."""
    if pd.isna(value):
        return None
    s = str(value).replace('€', '').replace(' ', '')
    match = re.match(r"([0-9,.]+)([MKmk]?)", s)
    if not match:
        try:
            return float(s)
        except ValueError:
            return None
    amount, unit = match.groups()
    amount = amount.replace(',', '')
    num = float(amount)
    if unit.upper() == 'M':
        num *= 1_000_000
    elif unit.lower() == 'k':
        num *= 1_000
    return num

def clean_dataframe(df):
    """Perform all cleaning steps on the DataFrame."""
    # Avoid SettingWithCopyWarning by working on a copy
    df = df.copy()

    # Parse monetary columns
    df['price_eur'] = df['price'].apply(parse_money)
    df['max_price_eur'] = df['max_price'].apply(parse_money)

    # Convert date columns
    for col in ['joined_club', 'contract_expires']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Handle missing values
    df = df.dropna(subset=['price_eur', 'max_price_eur'], how='all')
    median_price = df['price_eur'].median()
    median_max = df['max_price_eur'].median()
    df.loc[:, 'price_eur'] = df['price_eur'].fillna(median_price)
    df.loc[:, 'max_price_eur'] = df['max_price_eur'].fillna(median_max)

    # Standardize text fields
    text_cols = ['player_agent', 'club', 'nationality', 'place_of_birth', 'foot', 'outfitter']
    for col in text_cols:
        df.loc[:, col] = df[col].fillna('Unknown')
        df.loc[:, col] = df[col].astype(str).str.strip().str.title()

    return df

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    raw_path = os.path.join(repo_root, 'data', 'raw', 'bundesliga_player.csv')
    out_dir = os.path.join(repo_root, 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'players_clean.csv')

    df_raw = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    df_clean = clean_dataframe(df_raw)
    print(f"Cleaned data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    df_clean.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path}")

if __name__ == '__main__':
    main()
