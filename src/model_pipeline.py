# Bundesliga player market value regression pipeline
# - Loads processed player data from CSV
# - Builds preprocessing (impute/scale/encode) + RandomForest regressor
# - Trains, evaluates, and saves a scikit-learn Pipeline via joblib
# Usage example:
#   python -m src.model_pipeline data/processed_players.csv models/pipeline.joblib --test-size 0.2

import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1) FINAL FEATURE LISTS
NUMERIC_FEATURES = [
    'age',
    'max_price_eur',    # peak market value
    'price_to_max',     # relative value: price_eur / max_price_eur
    'tenure_years'
]
CATEGORICAL_FEATURES = [
    'position',
    'club',
    'nationality'
]


# Load CSV and compute engineered features.
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV into a DataFrame and compute any engineered features.
    """
    df = pd.read_csv(csv_path)
    if 'price_to_max' not in df.columns:
        df['price_to_max'] = df['price_eur'] / df['max_price_eur']
    df = df.dropna(subset=['price_eur'])
    return df


# Build ColumnTransformer for numeric/categorical preprocessing.
def build_preprocessor() -> ColumnTransformer:
    """
    Create a ColumnTransformer: imputes + scales numerics, imputes + encodes categoricals.
    """
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, NUMERIC_FEATURES),
            ('cat', cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder='drop'
    )
    return preprocessor


# Assemble preprocessing + RandomForest into a Pipeline.
def build_model_pipeline() -> Pipeline:
    """
    Assemble preprocessing + RandomForest regressor.
    """
    preprocessor = build_preprocessor()
    # Best from recent tuning: Lasso with alpha=0.001
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso(alpha=0.001, max_iter=5000, random_state=42))
    ])
    return pipeline


# Train pipeline, evaluate on holdout, and save to disk.
def train_and_save_model(
    csv_path: str,
    model_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load data, train pipeline, evaluate, and save.
    """
    df = load_data(csv_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['price_eur']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"[train] Test RMSE: {rmse:.2f}, R2: {r2:.3f}")

    # Save
    joblib.dump(pipeline, model_path)
    print(f"[train] Saved pipeline to: {model_path}")


# Load a saved scikit-learn Pipeline from disk.
def load_model(model_path: str):
    """
    Load a saved sklearn Pipeline via joblib.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print(f"[load_model] loaded object of type: {type(model)} from {path}")
    return model


if __name__ == '__main__':
    import argparse

    # CLI entry point: parse args, train, evaluate, and save the model.
    parser = argparse.ArgumentParser(
        description='Train & save the market-value regression pipeline.'
    )
    parser.add_argument('csv_path', help='Path to processed CSV')
    parser.add_argument('model_path', help='Output path for joblib file')
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction of data to hold out for testing'
    )
    args = parser.parse_args()

    train_and_save_model(
        args.csv_path,
        args.model_path,
        test_size=args.test_size
    )
