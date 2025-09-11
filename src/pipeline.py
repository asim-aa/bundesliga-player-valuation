# src/pipeline.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

def main():
    # 1. Load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    # 2. Preprocessor
    num_cols = X_train.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object","category"]).columns
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    # 3. Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 4. Fit and evaluate
    pipeline.fit(X_train, y_train)

    # 5. Save the trained pipeline for future predictions
    joblib.dump(pipeline, "models/best_pipeline.pkl")
    print("✔ Saved pipeline to models/best_pipeline.pkl")

    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Pipeline RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
