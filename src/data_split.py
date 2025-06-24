# src/split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your cleaned data
df = pd.read_csv("data/processed/players_clean.csv")

# 2. Split into features and target
X = df.drop("price", axis=1)
y = df["price"]


# 3. Hold out 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Save splits
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
