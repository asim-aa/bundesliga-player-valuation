# src/prediction.py
import pandas as pd
import difflib

def predict_value_progression(model, df, player_name, start_date, periods, freq):
    """
    Returns (dates, predictions) for the given player.
    """

    # — look up the player name (case‐insensitive) —
    mask = df["name"].str.lower() == player_name.lower()
    player_rows = df[mask]
    if player_rows.empty:
        suggestions = difflib.get_close_matches(player_name, df["name"], n=5, cutoff=0.6)
        if suggestions:
            raise ValueError(f"Player '{player_name}' not found. Did you mean: {suggestions}")
        else:
            raise ValueError(f"Player '{player_name}' not found in column 'name'.")

    # — figure out exactly which columns your pipeline was trained on —
    preproc = model.named_steps['preproc']  # your ColumnTransformer
    # transformers_ is a list of (name, transformer, columns)
    num_cols = cat_cols = []
    for nm, transformer, cols in preproc.transformers_:
        if nm == 'num':
            num_cols = cols
        elif nm == 'cat':
            cat_cols = cols
    feature_cols = list(num_cols) + list(cat_cols)

    # — slice down to only those columns —
    X_current = player_rows[feature_cols].reset_index(drop=True)

    # — build your future DataFrame by repeating that row —
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    X_future = pd.concat([X_current.copy() for _ in dates], ignore_index=True)

    # — predict —
    preds = model.predict(X_future)
    return dates, preds
