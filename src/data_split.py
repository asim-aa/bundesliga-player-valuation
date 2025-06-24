#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split(path, target_col="price", test_size=0.2, random_state=42):
    """
    Loads a CSV into df, splits into X_train, X_test, y_train, y_test.
    """
    df = pd.read_csv(path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split("data/processed/players_clean.csv")
    print("Train/Test sizes:", X_train.shape, X_test.shape)
