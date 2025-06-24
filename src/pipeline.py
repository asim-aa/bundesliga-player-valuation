#!/usr/bin/env python3
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def build_pipeline(X):
    numeric_cols   = X.select_dtypes(include=["int64","float64"]).columns
    categorical_cols = X.select_dtypes(include=["object","category"]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    model_pipe = Pipeline([
        ("preproc", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return model_pipe

if __name__ == "__main__":
    # Example usage
    from data_split import load_and_split
    X_train, X_test, y_train, y_test = load_and_split("data/processed/players_clean.csv")
    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)
    print("Pipeline training complete.")
