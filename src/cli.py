#!/usr/bin/env python3
# Predicts Bundesliga player market-value progression.
# - Loads a trained pipeline and processed data
# - Projects future market values for a specified player
# - Plots the projected values over time
# Usage examples:
#   python src/cli.py "Jamal Musiala" --years 5 --freq Y
#   python src/cli.py "Jamal Musiala" --start-date 2025-01-01 --periods 60 --freq M
import argparse
import sys
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
        nargs="?",
        default=None,
        help="Full name of the player (e.g. 'Jamal Musiala')."
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode: prompt for player, frequency, and horizon."
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
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the plot (e.g. 'outputs/musiala_projection.png')."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window; useful for headless environments."
    )
    return parser.parse_args()


# Orchestrate loading, projecting, and plotting results.
def main():
    args = parse_args()

    def interactive_inputs():
        print("Interactive mode: answer the prompts below.")
        # Player name
        while True:
            name = input("Player name: ").strip()
            if name:
                break
            print("Please enter a non-empty name.")

        # Frequency selection
        freq_map = {"M": "M", "ME": "ME", "Y": "Y", "A": "Y"}
        while True:
            freq_in = input("Project by months or years? [M/Y] (default M): ").strip().upper()
            if not freq_in:
                freq = "ME"  # month-end by default for better plots
                break
            if freq_in in freq_map:
                freq = freq_map[freq_in]
                break
            print("Please enter 'M' for monthly or 'Y' for yearly.")

        # Horizon
        unit = "months" if freq in ("M", "ME") else "years"
        while True:
            horizon_str = input(f"How many {unit}? (e.g., 12): ").strip()
            try:
                horizon = int(horizon_str)
                if horizon <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Please enter a positive whole number.")

        # Start date
        start_in = input("Start date YYYY-MM-DD (leave blank for today): ").strip()
        start = start_in or date.today().isoformat()

        # Periods determination
        periods = horizon  # months if monthly freq, years if yearly freq
        return name, start, periods, freq

    if args.interactive:
        player_name, start, periods, freq = interactive_inputs()
    else:
        if not args.player_name:
            raise SystemExit("error: player_name is required unless --interactive is used")
        player_name = args.player_name
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
            freq = args.freq
        else:
            periods = args.periods if args.periods is not None else 12
            freq = args.freq

    # Load model & data
    model = load_model(str(MODEL_PATH))
    df    = load_data(str(DATA_PATH))

    # Generate projection
    try:
        dates, values = predict_value_progression(
            model,
            df,
            player_name=player_name,
            start_date=start,
            periods=periods,
            freq=freq
        )
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)

    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(dates, values, marker="o")
    plt.title(f"Predicted Value Progression for {player_name}")
    plt.xlabel("Date")
    plt.ylabel("Market Value (€)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save if requested
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"✔ Saved plot to {out_path}")

    # Show unless suppressed (useful for headless CI/demos)
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
