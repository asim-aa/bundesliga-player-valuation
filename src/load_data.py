#!/usr/bin/env python3
"""
Script: load_data.py

Loads the raw Bundesliga players CSV and prints basic summaries:
 - First 5 rows
 - DataFrame info
 - Descriptive statistics
"""
import pandas as pd
import os
import sys

def load_data(path):
    """Read CSV into a DataFrame."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        sys.exit(f"Error: File not found at {path}")

def main():
    # Build path relative to this script
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = os.path.join(repo_root, 'data', 'raw', 'bundesliga_player.csv')

    print(f"Loading data from: {csv_path}\n")
    df = load_data(csv_path)

    print("--- First 5 rows ---")
    print(df.head(), end="\n\n")

    print("--- DataFrame Info ---")
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all'))

if __name__ == '__main__':
    main()
