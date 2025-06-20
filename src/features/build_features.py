import os
import pandas as pd

def main():
    # Paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    in_path = os.path.join(root, 'data', 'processed', 'players_clean.csv')
    out_path = os.path.join(root, 'data', 'processed', 'players_features.csv')

    df = pd.read_csv(in_path, parse_dates=['joined_club', 'contract_expires'])
    # ... build features here ...

    df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path}")

if __name__ == '__main__':
    main()
