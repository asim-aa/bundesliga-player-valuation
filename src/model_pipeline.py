import joblib
import pandas as pd
from pathlib import Path


def load_model(model_path: str):
    """
    Load a scikit-learn Pipeline or estimator via joblib.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print(f"[load_model] loaded object of type: {type(model)} from {path}")
    return model


def load_data(csv_path: str):
    """
    Load processed player-features CSV into a DataFrame,
    parsing date columns as needed.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=['joined_club', 'contract_expires'],
        infer_datetime_format=True
    )
    return df
