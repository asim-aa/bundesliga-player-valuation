#!/usr/bin/env python3
import argparse
from datetime import datetime
import sys
import matplotlib
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
print("Matplotlib backend:", matplotlib.get_backend())
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict future market‐value progression for a Bundesliga player."
    )
    parser.add_argument(
        "player_name",
        help="Full name of the player (e.g. 'Jamal Musiala')."
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        default=None,
        help="Projection start date (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=12,
        help="Number of periods to project (default: 12)."
    )
    parser.add_argument(
        "--freq",
        default="ME",
        help="Date frequency string for pd.date_range (e.g. 'ME', 'M', 'W'; default: 'ME')."
    )
    return parser.parse_args()
def main():
    args = parse_args()

    # 1) Load WITHOUT parse_dates
    df = pd.read_csv("data/processed/players_clean.csv")
    for date_col in ["contract_expires", "joined_club"]:
        if date_col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
            else:
                df[date_col] = df[date_col].astype(str)

    pipeline = joblib.load("models/best_pipeline.pkl")

    # 3) Lookup player
    mask = df["name"].str.lower() == args.player_name.lower()
    if not mask.any():
        print(f"Player '{args.player_name}' not found. Sample names:")
        print("\n".join(df["name"].unique()[:10]))
        sys.exit(1)
    static = df.loc[mask].iloc[0]

    # 4) Start date & index
    if args.start_date:
        start = pd.to_datetime(args.start_date)
    else:
        start = pd.Timestamp(datetime.today().date())
    dates = pd.date_range(start=start, periods=args.periods, freq=args.freq)

    # 5) Build snapshots
    rows = []
    for d in dates:
        rows.append({
            "age": static["age"],
            "days_remaining": (pd.to_datetime(static["contract_expires"]) - d).days,
            "height": static["height"],
            "position": static["position"],
        })
    X_future = pd.DataFrame(rows, index=dates)

    # 6) Fill & reorder features
    required = list(pipeline.feature_names_in_)
    for col in required:
        if col not in X_future.columns:
            X_future[col] = static[col]
    X_future = X_future[required]

    # 7) Cast categoricals, predict, plot…
    for col in X_future.select_dtypes(include="object"):
        X_future[col] = X_future[col].astype("category")

    preds = pipeline.predict(X_future)
    plt.figure(figsize=(8, 4))
    plt.plot(dates, preds, marker="o")
    plt.title(f"Predicted Market Value for {static['name']}")
    plt.xlabel("Date")
    plt.ylabel("Value (€)")
    plt.xticks(rotation=45)
    plt.tight_layout()
   # plt.show()
