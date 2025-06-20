#!/usr/bin/env python3
"""
Script: build_features.py

Loads the cleaned Bundesliga players dataset and creates new features:
 - Tenure in years
 - Price-to-max ratio
 - Age group buckets
 - Height category
Saves enhanced dataset to data/processed/players_features.csv
"""
import os
import pandas as pd

def main():
    # Paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    in_path = os.path.join(root, 'data', 'processed', 'players_clean.csv')
    out_path = os.path.join(root, 'data', 'processed', 'players_features.csv')

    # Load cleaned data
    df = pd.read_csv(in_path, parse_dates=['joined_club', 'contract_expires'])

    # 1. Tenure in years
    df['tenure_years'] = (pd.Timestamp.today() - df['joined_club']).dt.days / 365

    # 2. Price-to-max ratio
    df['price_to_max'] = df['price_eur'] / df['max_price_eur']

    # 3. Age group buckets
    bins = [0, 20, 24, 29, 100]
    labels = ['Under 21', '21-24', '25-29', '30+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    # 4. Height category
    df['height_cat'] = pd.cut(
        df['height'],
        bins=[0, 175, 185, 300],
        labels=['Short', 'Average', 'Tall']
    )

    # Save features
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path}")

if __name__ == '__main__':
    main()
