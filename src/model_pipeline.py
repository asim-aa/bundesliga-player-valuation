#!/usr/bin/env python3
"""
Script: model_pipeline.py

– Loads cleaned data
– Fits preprocessing on full dataset
– Splits into train/test
– Runs baseline LinearRegression (on log-price)
– Compares Ridge, Lasso, RF, XGBoost via grid search
– Persists best model
– Prints feature importances
– Builds SHAP explanations
– Computes all metrics on original price scale
"""
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import shap

def main():
    # 1) Load cleaned data
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(repo_root, 'data', 'processed', 'players_clean.csv')
    df = pd.read_csv(data_path)

    # 2) Define X and log-transformed target y_log
    X     = df.drop('price', axis=1)
    y_log = np.log1p(df['price'])

    # 3) Preprocessor definition
    numeric_cols = ['age', 'height', 'max_price']
    cat_cols     = ['position', 'nationality', 'foot', 'club', 'player_agent', 'outfitter']

    full_pre = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ])

    # 4) Fit on the full X so it learns all categories
    full_pre.fit(X)

    # 5) Wrap the fitted transformer so Pipeline won't re-fit it
    wrapped_pre = FunctionTransformer(full_pre.transform, validate=False)

    # 6) Train/test split on X and log-target
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    # Recover the original-price test target for metric computation
    y_test_orig = np.expm1(y_test_log)

    # 7) Baseline pipeline (LinearRegression on log-price)
    baseline = Pipeline([
        ('pre', wrapped_pre),
        ('lr',  LinearRegression()),
    ])
    baseline.fit(X_train, y_train_log)

    # Predict log-price
    y_pred_log = baseline.predict(X_test)
    # Log-scale metrics
    mse_log = mean_squared_error(y_test_log, y_pred_log)
    print("Baseline LinearRegression (log-target)")
    print(f"  RMSE (log-scale): {np.sqrt(mse_log):.3f}")
    print(f"  R²  (log-scale): {r2_score(y_test_log, y_pred_log):.3f}")

    # Back-transform and compute original-scale metrics
    y_pred = np.expm1(y_pred_log)
    print("Baseline LinearRegression (original-scale)")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred)):.2f}")
    print(f"  R² : {r2_score(y_test_orig, y_pred):.3f}")

    # 8) Advanced models + grid search
    models = {
        'ridge': {
            'model': Ridge(),
            'params': {'model__alpha': [0.1, 1, 10]}
        },
        'lasso': {
            'model': Lasso(max_iter=5000),
            'params': {'model__alpha': [0.01, 0.1, 1]}
        },
        'rf': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth':    [None, 10, 20]
            }
        },
        'xgb': {
            'model': XGBRegressor(objective='reg:squarederror', random_state=42),
            'params': {
                'model__n_estimators':    [100, 200],
                'model__max_depth':       [3, 6],
                'model__learning_rate':   [0.01, 0.1]
            }
        }
    }

    best_rmse  = np.inf
    best_model = None

    for name, cfg in models.items():
        pipe = Pipeline([
            ('pre',   wrapped_pre),
            ('model', cfg['model'])
        ])
        gs = GridSearchCV(
            pipe, cfg['params'],
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        gs.fit(X_train, y_train_log)

        # back-transform predictions
        y_hat_log = gs.predict(X_test)
        y_hat     = np.expm1(y_hat_log)

        rmse = np.sqrt(mean_squared_error(y_test_orig, y_hat))
        r2   = r2_score(y_test_orig, y_hat)
        print(f"{name.upper():<5} RMSE={rmse:.2f}, R2={r2:.3f}, best_params={gs.best_params_}")

        if rmse < best_rmse:
            best_rmse  = rmse
            best_model = gs.best_estimator_

    # 9) Persist best model
    os.makedirs(os.path.join(repo_root, 'models'), exist_ok=True)
    joblib.dump(best_model, os.path.join(repo_root, 'models', 'best_model.pkl'))

    # 10) Feature importances
    tree = best_model.named_steps['model']
    if hasattr(tree, 'feature_importances_'):
        imps = tree.feature_importances_
        names = full_pre.get_feature_names_out()
        imp_df = pd.DataFrame({'feature': names, 'importance': imps})
        imp_df = imp_df.sort_values('importance', ascending=False).head(10)
        print("\nTop 10 Feature Importances:")
        print(imp_df.to_string(index=False))

    # 11) SHAP on the tree model (disable strict additivity check)
    X_train_enc = full_pre.transform(X_train)
    X_test_enc  = full_pre.transform(X_test)
    if hasattr(X_train_enc, 'toarray'):
        X_train_enc = X_train_enc.toarray()
        X_test_enc  = X_test_enc.toarray()

    explainer = shap.TreeExplainer(tree, data=X_train_enc)
    shap_vals  = explainer.shap_values(
        X_test_enc,
        check_additivity=False
    )

    shap.summary_plot(
        shap_vals,
        X_test_enc,
        feature_names=full_pre.get_feature_names_out()
    )

if __name__ == '__main__':
    main()
