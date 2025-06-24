#!/usr/bin/env python3
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def tune_ridge_lasso(X_train, y_train, alphas=[0.01,0.1,1,10,100], cv=5):
    ridge_cv = GridSearchCV(Ridge(), {"alpha": alphas},
                            scoring="neg_root_mean_squared_error", cv=cv)
    ridge_cv.fit(X_train, y_train)

    lasso_cv = GridSearchCV(Lasso(max_iter=5000), {"alpha": alphas},
                            scoring="neg_root_mean_squared_error", cv=cv)
    lasso_cv.fit(X_train, y_train)

    print("Best Ridge α:", ridge_cv.best_params_)
    print("Best Lasso α:", lasso_cv.best_params_)
    return ridge_cv.best_estimator_, lasso_cv.best_estimator_

def fit_tree_models(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                               random_state=42)
    xgb_reg.fit(X_train, y_train)

    return rf, xgb_reg
