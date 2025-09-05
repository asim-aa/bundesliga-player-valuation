# Bundesliga Player Valuation

Pitch: I built an end-to-end machine-learning pipeline that estimates and forecasts Bundesliga players’ market value. It blends football knowledge with rigorous data science (cleaning → feature engineering → model selection → explainability) and ships with a simple CLI so anyone can price a player in seconds.

> Performance (hold-out): Lasso RMSE €0.001 · R² 1.000 · n = 102
> (Full evaluation and plots in docs/RESULTS.md.)

## Why I Built This (concise)
- Turn passion into proof: convert years of following European football into a working, testable product.
- Math → market impact: connect statistical modeling to decisions clubs care about (contracts, scouting ROI, squad planning).
- Learn by shipping: design clean data flows, automate feature creation, compare models fairly, and expose it via a usable interface.

**Who is this for?** Analysts, student researchers, and football ops folks who want quick, explainable valuations and short-horizon forecasts.

## What This Demonstrates
- Data rigor: leakage-aware splits; careful missing-data handling; consistent preprocessing.
- Football-aware features: tenure, age/peak curves, position encodings, contract windows, progression signals; automated age retrieval via Wikipedia.
- Modeling depth: Linear/Ridge/Lasso vs. tree ensembles (Random Forest, XGBoost) with cross-validated selection and hold-out evaluation.
- Explainability: feature importances/attributions and error analysis to understand why predictions move.
- Product thinking: a CLI that projects a named player’s value; structured outputs for notebooks or dashboards.
- Reproducibility: deterministic pipelines, clear artifacts, documented assumptions.

## What You Can Do
- Price a player now: estimate current market value for a named player.
- Forecast progression: project value across future periods (monthly or yearly) for “extend vs. sell” scenarios.
- Compare models: inspect trade-offs across learners and tune hyperparameters fairly.
- Explain predictions: see which features drive value by position or player cohorts.

## Quick Start
- Setup
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Interactive projection (prompts for inputs)
  - `python src/cli.py --interactive`
  - Also works: `python src/final.py` (defaults to interactive)
- Non-interactive examples
  - `python src/cli.py "Jamal Musiala" --years 5 --freq Y`
  - `python src/cli.py "Jamal Musiala" --start-date 2025-09-01 --periods 12 --freq M`
- Train a fresh pipeline artifact
  - `python src/model_pipeline.py data/processed/players_features.csv models/best_pipeline.pkl`
- Tune and compare models (prints metrics; shows plots)
  - `python src/model_tuning_and_comparison.py`

### Quick start example output
Example interaction (plot opens in a window):
```
$ python src/cli.py --interactive
Player name: Jamal Musiala
Project by months or years? [M/Y] (default M): Y
How many years? (e.g., 12): 5
Start date YYYY-MM-DD (leave blank for today):
# A projection plot is displayed.
```

## Results (headline)
- Reproduce by running: `python src/model_tuning_and_comparison.py`
- Record the best model and metrics from the printed table, for example:
  - Best model: RandomForestRegressor — Test RMSE: <your_value>, R²: <your_value>
- Full details, including the comparison table and plots, live in `docs/RESULTS.md`.

## Technical Choices (brief)
- Stack: Python, pandas, numpy, scikit-learn, XGBoost, matplotlib.
- Pipelines: fit/transform steps encapsulated to avoid leakage; consistent preprocessing across train/infer.
- Evaluation: cross-validation for model selection; hold-out metrics and calibration checks.
- Explainability: global importances + per-prediction attributions (see `src/explainability.py`).
- Data freshness: programmatic age retrieval from Wikipedia to avoid stale features.

## Links
- Methods & design notes: `docs/METHODS.md`
- How to run & CLI options: `docs/USAGE.md`
- Model performance & plots: `docs/RESULTS.md`
- Architecture overview: `docs/ARCHITECTURE.md`
- Age feature write-up: `data/data.md`

### Data Sources & Licensing
- Primary data: TODO — list your actual sources and terms.
- Age feature: Programmatic birthdate parsing via Wikipedia to keep ages current.
- Ethics & use: Educational and research purposes; not financial advice or transfer guidance.

## What I’d Improve Next
- Incorporate richer event data (pressing/expected threat), position-aware modeling, and contract clause features.
-
### Testing & Reproducibility
- Deterministic splits and seeds for repeatable results.
- Pinned dependencies in `requirements.txt`.
- Preprocessing encapsulated in a single pipeline to prevent leakage.
- Test-ready layout; add tests under `tests/` and run with `pytest -q`.
- Train time-aware models for trajectories and uncertainty bands.
- Package as a lightweight web app for non-technical stakeholders.

## Why This Matters
This project isn’t just predictions — it’s a disciplined approach to turning noisy sports data into decisions a club (or any business) can act on. It shows how I think, how I build, and how I communicate results responsibly.
