#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_eval_linear(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"LinearRegression RMSE: {rmse:.2f}")
    print(f"LinearRegression R²:   {r2:.3f}")
    return lr, rmse, r2
