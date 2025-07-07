# src/prediction.py
import pandas as pd
import numpy as np
import difflib

def predict_value_progression(model, df, player_name, start_date, periods, freq):
    """
    Returns (dates, adjusted_predictions) for the given player,
    applying a piecewise decay so that value peaks ~26–29, then
    declines, then drops more sharply after 32.
    """

    # — look up the player name (case-insensitive) —
    mask = df["name"].str.lower() == player_name.lower()
    player_rows = df[mask]
    if player_rows.empty:
        suggestions = difflib.get_close_matches(player_name, df["name"], n=5, cutoff=0.6)
        raise ValueError(
            f"Player '{player_name}' not found. "
            + (f"Did you mean: {suggestions}" if suggestions else "")
        )

    # — grab the pipeline’s feature list —
    preproc = model.named_steps['preproc']
    num_cols = cat_cols = []
    for nm, _, cols in preproc.transformers_:
        if nm == 'num':
            num_cols = cols
        elif nm == 'cat':
            cat_cols = cols
    feature_cols = list(num_cols) + list(cat_cols)

    # — base feature row & original age —
    X_base   = player_rows[feature_cols].iloc[[0]].reset_index(drop=True)
    orig_age = float(player_rows.iloc[0]["age"])

    # — build date index & replicate the base row —
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    rows  = []
    for d in dates:
        row = X_base.copy()
        # compute fractional years ahead
        months_ahead = (d - pd.to_datetime(start_date)) / pd.Timedelta(days=30.44)
        row["age"]   = orig_age + (months_ahead / 12.0)
        rows.append(row)
    X_future = pd.concat(rows, ignore_index=True)

    # — raw model predictions —
    raw_preds = model.predict(X_future)

    # — piecewise age decay multiplier —
    ages = X_future["age"].to_numpy()
    # default multiplier = 1
    mult = np.ones_like(ages)

    # after age 29, start gradual decline up to 32: 0%→30% drop
    mask_29_32 = (ages > 29) & (ages <= 32)
    mult[mask_29_32] = 1 - ((ages[mask_29_32] - 29) / 3) * 0.30

    # after age 32, steeper exponential decay
    mask_gt32 = ages > 32
    # continue from whatever mult at age=32, then apply exp drop
    base_at_32 = 1 - (3 / 3) * 0.30  # = 0.70
    mult[mask_gt32] = base_at_32 * np.exp(-(ages[mask_gt32] - 32) / 5)

    # combine
    adjusted_preds = raw_preds * mult

    return dates, adjusted_preds
