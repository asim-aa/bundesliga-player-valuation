import pandas as pd
import difflib


def predict_value_progression(model, df, player_name, start_date, periods, freq):
    """
    Returns (dates, predictions) for the given player.
    """
    # Use 'name' column for lookup
    name_col = "name"
    # Case-insensitive exact match
    mask = df[name_col].str.lower() == player_name.lower()
    player_rows = df[mask]

    # Fuzzy suggestions if no exact match
    if player_rows.empty:
        suggestions = difflib.get_close_matches(
            player_name,
            df[name_col].tolist(),
            n=5,
            cutoff=0.6
        )
        if suggestions:
            raise ValueError(
                f"Player '{player_name}' not found. Did you mean one of: {suggestions}?"
            )
        else:
            raise ValueError(
                f"Player '{player_name}' not found in column '{name_col}'."
            )

    # Drop only the target column; keep 'name' for preprocessing
    X_current = player_rows.drop(columns=["price"], errors="ignore")

    # Build date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Replicate features for each future date
    X_future = pd.concat(
        [X_current.assign(pred_date=date) for date in dates],
        ignore_index=True
    )

    # Predict
    preds = model.predict(X_future)
    return dates, preds
