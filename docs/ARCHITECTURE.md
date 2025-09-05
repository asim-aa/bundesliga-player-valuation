# Architecture Overview

High-level view of the data flow and main components.

## Data Flow

1. Data Preparation
   - Source data is cleaned and transformed into `data/processed/players_features.csv`.
   - Age enrichment via Wikipedia can be performed separately (see `fetch-scripts/`).

2. Training
   - `src/model_pipeline.py` builds a preprocessing + model `Pipeline` (impute/scale/encode → regressor).
   - Trains on processed data and saves `models/best_pipeline.pkl`.

3. Tuning & Comparison
   - `src/model_tuning_and_comparison.py` constructs a shared preprocessor and runs GridSearchCV over multiple estimators.
   - Reports Test RMSE/R² and best params; plots comparisons.

4. Inference / Projection
   - `src/cli.py` loads `models/best_pipeline.pkl` and `data/processed/players_features.csv`.
   - Calls `prediction.predict_value_progression(...)` to produce a time series for a given player.
   - Displays a matplotlib plot of projected values.

## Components & Responsibilities

- `src/model_pipeline.py`:
  - Defines final feature lists and preprocessing.
  - Trains RandomForest inside a `Pipeline` and saves the artifact.

- `src/model_tuning_and_comparison.py`:
  - Infers numeric/categorical columns from train split; builds `ColumnTransformer`.
  - Tunes Ridge, Lasso, RandomForest, and XGBoost with GridSearchCV.

- `src/cli.py` and `src/final.py`:
  - CLI for interactive/non-interactive usage.
  - `final.py` is a wrapper; when run without args it defaults to interactive mode.

- `src/prediction.py`:
  - Contains progression logic used by the CLI.

- `src/explainability.py`:
  - Provides tools for feature importances / attributions.

## Assumptions

- Processed dataset includes required features referenced by the pipeline.
- The trained pipeline is saved at `models/best_pipeline.pkl`.
- The `name` column in the dataset matches the player names used at inference.

