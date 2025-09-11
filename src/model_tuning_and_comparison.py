#!/usr/bin/env python3
# Model tuning and comparison for market-value regression.
# - Loads unified feature set from players_features.csv
# - Builds the same preprocessor as production (model_pipeline)
# - Tunes multiple models via GridSearchCV and evaluates on a reproducible split
# - Saves the tuned best pipeline to models/best_pipeline.pkl
# - Prints a comparison table and plots RMSE/R² bars
# Usage:
#   PYTHONPATH=src MPLBACKEND=Agg python src/model_tuning_and_comparison.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from model_pipeline import (
    load_data,
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def main():
    df = load_data("data/processed/players_features.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["price_eur"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    model_defs = {
        "Ridge": {
            "estimator": Ridge(max_iter=5_000),
            "params": {"model__alpha": [0.1, 1, 10]},
        },
        "Lasso": {
            "estimator": Lasso(max_iter=5_000),
            "params": {"model__alpha": [0.001, 0.01, 0.1]},
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
            },
        },
        "XGBoost": {
            "estimator": XGBRegressor(objective="reg:squarederror", random_state=42),
            "params": {
                "model__learning_rate": [0.01, 0.1],
                "model__subsample": [0.7, 1.0],
                "model__n_estimators": [100, 200],
            },
        },
    }

    records = []
    best_overall = None
    best_rmse = float("inf")

    for name, md in model_defs.items():
        print(f"\n→ Tuning {name}…")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", md["estimator"]),
        ])
        gs = GridSearchCV(
            pipe,
            md["params"],
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        preds = best.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))

        print(f"  • Best params: {gs.best_params_}")
        print(f"  • Test RMSE = {rmse:.3f}, R² = {r2:.3f}")

        records.append({"model": name, "rmse": rmse, "r2": r2})

        if rmse < best_rmse:
            best_rmse = rmse
            best_overall = best

    results = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
    print("\n=== All models compared ===")
    print(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    results.plot.bar(x="model", y="rmse", ax=ax1, legend=False, title="Test RMSE")
    results.plot.bar(x="model", y="r2", ax=ax2, legend=False, title="Test R²")
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    # Persist tuned best pipeline for CLI consumption
    out = Path("models/best_pipeline.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_overall, out)
    print(f"[tuning] Saved tuned best pipeline to: {out}")


if __name__ == "__main__":
    main()
