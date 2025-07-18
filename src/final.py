#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt

from model_pipeline import load_model, load_data
from prediction import predict_value_progression

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict market-value progression for a Bundesliga player."
    )
    parser.add_argument(
        'player_name',
        help="Player’s name as in the dataset ('name' column)."
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help="Start date for projection (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        '--periods',
        type=int,
        default=None,
        help="Number of future periods to project (default: 12). Ignored if --years is set."
    )
    parser.add_argument(
        '--freq',
        default='ME',
        help="Frequency string for pd.date_range (e.g. 'ME', 'M', 'W', 'Y'; default: 'ME')."
    )
    parser.add_argument(
        '--years',
        type=int,
        default=None,
        help="Shortcut to project over N years (will override --periods)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # if no start-date, use today
    start = args.start_date or date.today().isoformat()

    # determine number of periods
    if args.years is not None:
        freq_upper = args.freq.upper()
        if freq_upper in ('M', 'ME'):
            periods = args.years * 12
        elif freq_upper in ('Y', 'A'):
            periods = args.years
        else:
            # fallback: treat years as periods count
            periods = args.years
    else:
        # either user passed --periods or we default to 12
        periods = args.periods if args.periods is not None else 12

    # Resolve paths relative to project root
    BASE       = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE / 'models' / 'best_pipeline.pkl'
    DATA_PATH  = BASE / 'data'   / 'processed' / 'players_features.csv'

    # Load model and data
    model = load_model(str(MODEL_PATH))
    df    = load_data(str(DATA_PATH))

    # Generate projections
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
    plt.plot(dates, values, marker='o')
    plt.title(f"Predicted Value Progression for {args.player_name}")
    plt.xlabel('Date')
    plt.ylabel('Market Value (€)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
