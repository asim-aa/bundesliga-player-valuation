# src/prediction.py
import pandas as pd

def predict_value_progression(model, df, player_name, start_date, periods, freq):
    """
    Returns (dates, predictions) for the given player.
    """
    # 1) Locate the player row
    player_rows = df[df["full_name"] == player_name]
    if player_rows.empty:
        raise ValueError(f"Player '{player_name}' not found in data")
    # Drop any non-feature cols (adjust names as needed)
    X_current = player_rows.drop(columns=["full_name", "price"], errors="ignore")
    
    # 2) Build a date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # 3) Replicate the player’s features for each future date
    X_future = pd.concat(
        [X_current.assign(pred_date=date) for date in dates],
        ignore_index=True
    )
    # (If you need to extract date features—month, year, etc.—do it here.)
    
    # 4) Predict
    preds = model.predict(X_future)
    
    return dates, preds
