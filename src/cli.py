#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from model_pipeline import load_model, load_data
from prediction import predict_value_progression

# project_root/src/predict_player_progression.py ⇒ project_root
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.pkl"
DATA_PATH  = BASE_DIR / "data"   / "processed" / "players_features.csv"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict future market-value progression for a Bundesliga player."
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

    # load model & data
    model = load_model(str(MODEL_PATH))
    df    = load_data(str(DATA_PATH))

    # generate projection
    dates, values = predict_value_progression(
        model,
        df,
        player_name=args.player_name,
        start_date=args.start_date,
        periods=args.periods,
        freq=args.freq
    )

    # plot results
    plt.figure(figsize=(8, 4))
    plt.plot(dates, values, marker="o")
    plt.title(f"Predicted Value Progression for {args.player_name}")
    plt.xlabel("Date")
    plt.ylabel("Market Value (€)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

#can also do: python src/cli.py --start-date 2025-01-01 --periods 60 --freq M "Jamal Musiala"