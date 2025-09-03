#!/usr/bin/env python3
# Predicts Bundesliga player market-value progression.
# - Loads a trained pipeline and processed data
# - Projects future market values for a specified player
# - Plots the projected values over time
# Usage examples:
#   python src/cli.py "Jamal Musiala" --years 5 --freq Y
#   python src/cli.py "Jamal Musiala" --start-date 2025-01-01 --periods 60 --freq M
import argparse
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt

from model_pipeline import load_model, load_data
from prediction import predict_value_progression

# Resolve project root relative to this file
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_pipeline.pkl"
DATA_PATH  = BASE_DIR / "data"   / "processed" / "players_features.csv"

# Parse CLI arguments for progression prediction.
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
        default=None,
        help="Number of periods to project (default: 12 if --years not set)."
    )
    parser.add_argument(
        "--freq",
        default="ME",
        help="Date frequency string for pd.date_range (e.g. 'ME', 'M', 'W', 'Y'; default: 'ME')."
    )
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Project over N years (overrides --periods; M/ME -> 12 per year, Y/A -> 1 per year)."
    )
    return parser.parse_args()


# Orchestrate loading, projecting, and plotting results.
def main():
    args = parse_args()

    # Determine start date (default to today if not provided)
    start = args.start_date or date.today().isoformat()

    # Determine number of periods
    if args.years is not None:
        freq_upper = (args.freq or "").upper()
        if freq_upper in ("M", "ME"):
            periods = args.years * 12
        elif freq_upper in ("Y", "A"):
            periods = args.years
        else:
            periods = args.years
    else:
        periods = args.periods if args.periods is not None else 12

    # Load model & data
    model = load_model(str(MODEL_PATH))
    df    = load_data(str(DATA_PATH))

    # Generate projection
    dates, values = predict_value_progression(
        model,
        df,
        player_name=args.player_name,
        start_date=start,
        periods=periods,
        freq=args.freq
    )

    # Plot results
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
