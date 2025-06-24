#!/usr/bin/env python3
"""
Script: baseline_model.py
Preprocesses (encodes/scales) then trains & evaluates a linear baseline.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_eval_linear(X_train, X_test, y_train, y_test):
    # 1. Identify columns by dtype
    num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object","category"]).columns.tolist()

    # 2. Build a ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    # 3. Create the full pipeline
    pipe = Pipeline([
        ("preproc", preprocessor),
        ("model", LinearRegression())
    ])

    # 4. Fit and predict
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 5. Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"LinearRegression RMSE: {rmse:.2f}")
    print(f"LinearRegression R²:   {r2:.3f}")

    return pipe, rmse, r2

if __name__ == "__main__":
    # Load your train/test splits
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    train_and_eval_linear(X_train, X_test, y_train, y_test)
