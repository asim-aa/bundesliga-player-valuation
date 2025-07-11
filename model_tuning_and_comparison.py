#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_splits():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    # ——— DEBUG: make sure no leakage of target
    for name, X in [("train", X_train), ("test", X_test)]:
        if "price" in X.columns:
            print(f"⚠️  Dropping leaked 'price' column from X_{name}")
            X.drop(columns="price", inplace=True)

    return X_train, X_test, y_train, y_test

def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object","category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",   StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

def main():
    X_train, X_test, y_train, y_test = load_splits()
    preprocessor = build_preprocessor(X_train)

    model_defs = {
        "Ridge": {
            "estimator": Ridge(max_iter=5_000),
            "params": {"model__alpha": [0.1, 1, 10]}
        },
        "Lasso": {
            "estimator": Lasso(max_iter=5_000),
            "params": {"model__alpha": [0.001, 0.01, 0.1]}
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth":   [None, 5, 10]
            }
        },
        "XGBoost": {
            "estimator": XGBRegressor(objective="reg:squarederror", random_state=42),
            "params": {
                "model__learning_rate": [0.01, 0.1],
                "model__subsample":     [0.7, 1.0],
                "model__n_estimators":  [100, 200]
            }
        }
    }

    records = []
    for name, md in model_defs.items():
        print(f"\n→ Tuning {name}…")
        pipe = Pipeline([
            ("preproc", preprocessor),
            ("model",   md["estimator"])
        ])
        gs = GridSearchCV(pipe,
                          md["params"],
                          cv=5,
                          scoring="neg_mean_squared_error",
                          n_jobs=-1,
                          verbose=1)
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        preds = best.predict(X_test)

        # correct RMSE
        mse  = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, preds)

        print(f"  • Best params: {gs.best_params_}")
        print(f"  • Test RMSE = {rmse:.3f}, R² = {r2:.3f}")

        records.append({"model": name, "rmse": rmse, "r2": r2})

    # comparison table + plots
    results = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
    print("\n=== All models compared ===")
    print(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    results.plot.bar(x="model", y="rmse", ax=ax1, legend=False, title="Test RMSE")
    results.plot.bar(x="model", y="r2",   ax=ax2, legend=False, title="Test R²")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
