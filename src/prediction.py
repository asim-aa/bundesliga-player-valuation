# src/prediction.py
import pandas as pd
import difflib

def predict_value_progression(model, df, player_name, start_date, periods, freq):
    """
    Returns (dates, predictions) for the given player,
    injecting a growing 'age' feature so the curve can slope up or down.
    """

    # — look up the player name (case-insensitive) —
    mask = df["name"].str.lower() == player_name.lower()
    player_rows = df[mask]
    if player_rows.empty:
        suggestions = difflib.get_close_matches(player_name, df["name"], n=5, cutoff=0.6)
        if suggestions:
            raise ValueError(f"Player '{player_name}' not found. Did you mean: {suggestions}")
        else:
            raise ValueError(f"Player '{player_name}' not found in column 'name'.")

    # — figure out exactly which columns your pipeline was trained on —
    preproc = model.named_steps['preproc']  # ColumnTransformer
    num_cols = cat_cols = []
    for nm, transformer, cols in preproc.transformers_:
        if nm == 'num':
            num_cols = cols
        elif nm == 'cat':
            cat_cols = cols
    feature_cols = list(num_cols) + list(cat_cols)

    # — get the static feature row (first matching player) —
    X_base = player_rows[feature_cols].iloc[[0]].reset_index(drop=True)

    # original age (in years) from your data
    orig_age = float(player_rows.loc[player_rows.index[0], "age"])

    # — build your date index and a list of age-bumped rows —
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    rows = []

    for d in dates:
        row = X_base.copy()

        # compute fractional months ahead
        months_ahead = (d - pd.to_datetime(start_date)) / pd.Timedelta(days=30.44)
        years_ahead = months_ahead / 12.0

        # bump the age
        row["age"] = orig_age + years_ahead

        rows.append(row)

    # — concatenate into a full DataFrame —
    X_future = pd.concat(rows, ignore_index=True)

    # — predict on each aged-up instance —
    preds = model.predict(X_future)
    return dates, preds
