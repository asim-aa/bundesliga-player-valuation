# src/predict_player_progression.py

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib


def load_pipeline(path="models/best_pipeline.pkl"):
    """Loads the pretrained sklearn Pipeline (preprocessing + model)."""
    return joblib.load(path)


def get_player_static(df, player_name):
    """
    Extracts the static profile for one player by name.
    """
    subset = df.loc[df["name"] == player_name]
    if subset.empty:
        sample = df["name"].unique()[:10].tolist()
        print(f"Error: no player named '{player_name}'\n"
              f"Here are 10 valid examples:\n  {sample}")
        sys.exit(1)

    player = subset.iloc[0]
    return {
        "birth_date": pd.to_datetime(player["birth_date"]),
        "contract_expires": pd.to_datetime(player["contract_expires"]),
        "height": player["height"],
        "position": player["position"],
        # optional fields; will be ignored if not present
        "preferred_foot": player.get("preferred_foot", None),
        "club": player.get("club", None)
    }


def build_time_grid(start_date, periods=12, freq="M"):
    """Returns a DatetimeIndex of future snapshot dates."""
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def assemble_snapshots(static, dates):
    rows = []
    for snapshot in dates:
        age = (snapshot - static["birth_date"]).days / 365.25
        days_remaining = (static["contract_expires"] - snapshot).days
        row = {
            "snapshot_date": snapshot,
            "age": age,
            "days_remaining": days_remaining,
            "height": static["height"],
            "position": static["position"],
        }
        if static["preferred_foot"] is not None:
            row["preferred_foot"] = static["preferred_foot"]
        if static["club"] is not None:
            row["club"] = static["club"]
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_features(df_snapshots):
    df = df_snapshots.copy()
    if "position" in df:
        df["position"] = df["position"].astype("category")
    if "preferred_foot" in df:
        df["preferred_foot"] = df["preferred_foot"].astype("category")
    return df.drop(columns=["snapshot_date"])


def plot_progression(dates, values, player_name="Player"):
    plt.figure(figsize=(8, 4))
    plt.plot(dates, values, marker="o")
    plt.title(f"{player_name} Value Projection")
    plt.xlabel("Date")
    plt.ylabel("Predicted Market Value (M€)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Predict a player’s market value progression over time"
    )
    parser.add_argument(
        "player_name",
        help="Exact player name as in data/processed/players_clean.csv"
    )
    parser.add_argument(
        "--start-date",
        default="2025-06-23",
        help="Start date for projection (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=12,
        help="Number of future points to project"
    )
    parser.add_argument(
        "--freq",
        default="M",
        help="Frequency for date grid (e.g. M=month, W=week)"
    )
    args = parser.parse_args()

    df = pd.read_csv("data/processed/players_clean.csv")
    pipeline = load_pipeline()

    # show a few valid names
    print("Sample player names:", df["name"].unique()[:10].tolist())

    static = get_player_static(df, args.player_name)
    future_dates = build_time_grid(
        start_date=pd.to_datetime(args.start_date),
        periods=args.periods,
        freq=args.freq
    )
    snapshots = assemble_snapshots(static, future_dates)
    X_player = prepare_features(snapshots)
    price_preds = pipeline.predict(X_player)
    plot_progression(future_dates, price_preds, args.player_name)

def main():
    # load data & pipeline
    df = pd.read_csv("data/processed/players_clean.csv")
    pipeline = load_pipeline()

    # show a few valid names
    print("Sample player names:", df["name"].unique()[:10].tolist())

    # interactive input
    player_name = input("Enter exact player name from the list above: ").strip()

    static = get_player_static(df, player_name)
    # … rest of code …


if __name__ == "__main__":
    main()
