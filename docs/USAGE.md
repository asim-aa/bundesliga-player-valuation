# Usage

This guide covers how to run projections, train models, and compare/tune them using the CLI and scripts in this repo.

## CLI: Project Player Value Progression

Entry points:
- `src/cli.py`: canonical CLI with interactive and non-interactive modes.
- `src/final.py`: thin wrapper; if run with no args, defaults to interactive mode.

### Interactive Mode
- Prompts for player name, monthly/yearly choice, horizon, and optional start date.
- Command:
  - `python src/cli.py --interactive`
  - or: `python src/final.py` (no args)

Prompts look like:
```
Player name: Jamal Musiala
Project by months or years? [M/Y] (default M): Y
How many years? (e.g., 12): 5
Start date YYYY-MM-DD (leave blank for today):
```
This opens a matplotlib plot of predicted value over time.

### Non-Interactive Mode
- Directly specify inputs as flags.
```
python src/cli.py "Jamal Musiala" --years 5 --freq Y
python src/cli.py "Jamal Musiala" --start-date 2025-09-01 --periods 60 --freq M
```

Flags:
- `player_name` (positional; optional if `--interactive`): full player name as it appears in the dataset (`name` column).
- `-i, --interactive`: prompt for inputs.
- `--start-date`: ISO date `YYYY-MM-DD`; defaults to today if omitted.
- `--periods`: number of future periods to project (used when `--years` is not set). Default: 12.
- `--years`: convenience for yearly horizons; translates to periods based on `--freq` (M/ME → 12 per year; Y/A → 1 per year).
- `--freq`: frequency string for `pandas.date_range`; common values: `ME` (month end, default), `M`, `Y`.

Paths used by the CLI:
- Model: `models/best_pipeline.pkl`
- Data:  `data/processed/players_features.csv`

## Train the Pipeline

Use `src/model_pipeline.py` to train preprocessing + model and save a reusable pipeline.

```
python src/model_pipeline.py \
  data/processed/players_features.csv \
  models/best_pipeline.pkl \
  --test-size 0.2
```

This prints test RMSE/R² and writes the joblib artifact to `models/best_pipeline.pkl` (used by the CLI).

## Tune and Compare Models

Run grid search across several models with consistent preprocessing and a fixed train/test split.

```
python src/model_tuning_and_comparison.py
```

Outputs:
- Printed best params and test metrics per model.
- A sorted comparison table by RMSE.
- Two bar plots (RMSE and R²).

To capture the headline metric for the README:
- Note the best model and its Test RMSE/R² from the console output.

## Troubleshooting

- If the CLI cannot find the model or data files, ensure the expected paths exist:
  - `models/best_pipeline.pkl`
  - `data/processed/players_features.csv`
- If running headless (no display), consider adding a flag to save plots instead of showing; I can add `--save-path` on request.

