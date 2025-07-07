import argparse
from model_pipeline import load_model, load_data
from prediction import predict_value_progression
import matplotlib.pyplot as plt
from pathlib import Path

# project_root/src/final.py ⇒ project_root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH  = BASE_DIR / "data"   / "processed" / "players_features.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Predict player market-value progression. Only the player's name is required."  
    )
    # Only a single positional argument for player name
    parser.add_argument(
        'player_name', 
        help='Full name of the player as in the dataset.'
    )
    # Optional defaults hidden from help by setting argparse.SUPPRESS
    parser.add_argument(
        '--start-date', 
        default='2025-01-01', 
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--periods', 
        type=int, 
        default=12, 
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--freq', 
        default='M', 
        help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    # Load your pre-trained model and necessary data
    model = load_model(str(MODEL_PATH))
    df    = load_data(str(DATA_PATH))

    dates, values = predict_value_progression(
        model,
        df,
        player_name=args.player_name,
        start_date=args.start_date,
        periods=args.periods,
        freq=args.freq
    )

    # Plot results
    plt.figure(figsize=(8,4))
    plt.plot(dates, values)
    plt.title(f"Predicted Value Progression for {args.player_name}")
    plt.xlabel("Date")
    plt.ylabel("Market Value")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
