````markdown
# `model_pipeline.py` Documentation

This document walks through each part of `src/model_pipeline.py`, explains what the code does, why we produce each output, and how to interpret the final results.

---

## 1. Project Setup & Data Loading

```python
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_path = os.path.join(repo_root, 'data', 'processed', 'players_clean.csv')
df = pd.read_csv(data_path)
````

* **What it does**

  1. Locates the project root directory.
  2. Builds a portable path to `players_clean.csv`.
  3. Loads the cleaned DataFrame `df`.

* **Why**
  Ensures the script can run from any working directory and always uses the same pre-cleaned data.

---

## 2. Train/Test Split on Raw Prices

```python
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

* **What it does**
  Splits 80% for training, 20% for testing, with a fixed random seed for reproducibility.

* **Why**
  Prevents data leakage and simulates evaluating on truly unseen players.

---

## 3. Preprocessing: Scaling & Encoding

```python
numeric_cols = ['age', 'height', 'max_price']
cat_cols     = ['position','nationality','foot','club','player_agent','outfitter']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
])
```

* **What it does**

  * **StandardScaler**: zero-mean, unit-variance scaling for numeric features.
  * **OneHotEncoder**: binary encoding of categories, dropping the first level to avoid collinearity, ignoring unseen categories in test.

* **Why**
  Prepares features for models that assume normalized inputs or cannot natively handle strings.

---

## 4. Baseline Linear Regression

```python
baseline = Pipeline([
    ('pre', preprocessor),
    ('lr',  LinearRegression()),
])
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)
```

* **What it does**
  Chains preprocessing + a plain `LinearRegression`, trains on the train set, predicts on test.

* **Why**
  Serves as a simple benchmark: if your complex models can’t beat this, revisit your pipeline.

---

## 5. Baseline Metrics

```python
print("Baseline LinearRegression")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.0f}")
print(f"  R² : {r2_score(y_test, y_pred):.3f}")
```

**Output:**

```
Baseline LinearRegression
  RMSE: 10
  R² : 0.436
```

* **RMSE (Root Mean Squared Error)** ≈ 10 M € — average absolute error.
* **R²** ≈ 0.436 — explains 43.6% of variance in player prices.

---

## 6. Advanced Models & Hyperparameter Search

```python
models = {
  'ridge': { 'model': Ridge(), 'params': {'model__alpha': [0.1,1,10]} },
  'lasso': { 'model': Lasso(max_iter=5000), 'params': {'model__alpha': [0.01,0.1,1]} },
  'rf':    { 'model': RandomForestRegressor(random_state=42),
             'params': {'model__n_estimators':[100,200],'model__max_depth':[None,10,20]} },
  'xgb':   { 'model': XGBRegressor(objective='reg:squarederror',random_state=42),
             'params': {'model__n_estimators':[100,200],'model__max_depth':[3,6],'model__learning_rate':[0.01,0.1]} }
}

for name,cfg in models.items():
    pipe = Pipeline([('pre', preprocessor),('model', cfg['model'])])
    gs = GridSearchCV(pipe, cfg['params'], cv=5,
                      scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs.fit(X_train, y_train)
    y_hat = gs.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, y_hat))
    r2    = r2_score(y_test, y_hat)
    print(f"{name.upper():<5} RMSE={rmse:.0f}, R2={r2:.3f}, best_params={gs.best_params_}")
```

**Output:**

```
RIDGE RMSE=6, R2=0.803, best_params={'model__alpha': 10}
LASSO RMSE=6, R2=0.817, best_params={'model__alpha': 0.1}
RF    RMSE=6, R2=0.826, best_params={'model__max_depth':10,'model__n_estimators':100}
XGB   RMSE=4, R2=0.908, best_params={'model__learning_rate':0.1,'model__max_depth':3,'model__n_estimators':100}
```

* **Ridge/Lasso**: regularized linear models.
* **Random Forest**: bagged decision trees.
* **XGBoost**: gradient-boosted trees.
* **GridSearchCV** finds optimal hyperparameters via 5-fold CV.

---

## 7. Combined Performance Table

| Model                  | RMSE (M €) |    R² |
| ---------------------- | ---------: | ----: |
| **Baseline (Linear)**  |         10 | 0.436 |
| **Ridge**              |          6 | 0.803 |
| **Lasso**              |          6 | 0.817 |
| **Random Forest (RF)** |          6 | 0.826 |
| **XGBoost (XGB)**      |          4 | 0.908 |

* **RMSE**: average prediction error in millions of euros (lower is better).
* **R²**: fraction of variance explained (higher is better).

---

## 8. Persisting the Best Model

```python
os.makedirs(os.path.join(repo_root,'models'), exist_ok=True)
joblib.dump(best_model, os.path.join(repo_root,'models','best_model.pkl'))
```

* **Saves** the pipeline (preprocessor + estimator) for later use in scoring or deployment.

---

## 9. Feature Importances

```python
tree = best_model.named_steps['model']
imps  = tree.feature_importances_
names = preprocessor.get_feature_names_out()
top10 = pd.DataFrame({'feature':names,'importance':imps})\
           .nlargest(10,'importance')
print(top10.to_string(index=False))
```

**Output:**

```
                           feature  importance
                    num__max_price    0.491740
            cat__player_agent_Roof    0.089389
  cat__player_agent_Sports360 Gmbh    0.082507
                        num__age    0.060587
           cat__outfitter_Unknown    0.034035
          cat__club_Bayern Munich    0.021392
             cat__club_Rb Leipzig    0.020178
                      num__height    0.015207
cat__position_midfield - Central    0.015092
         cat__club_Bor. Dortmund    0.014523
```

* **`max_price`** dominates (\~49% of importance).
* **Agent**, **age**, **club**, **height**, **position** follow.

---

## 10. SHAP Explanations

```python
X_train_enc = preprocessor.transform(X_train)
X_test_enc  = preprocessor.transform(X_test)
if hasattr(X_train_enc,'toarray'):
    X_train_enc = X_train_enc.toarray()
    X_test_enc  = X_test_enc.toarray()

explainer = shap.TreeExplainer(tree, data=X_train_enc)
shap_vals  = explainer.shap_values(X_test_enc)
shap.summary_plot(shap_vals, X_test_enc, feature_names=names)
```

* **SHAP** assigns each feature a “push” on each prediction for interpretability.
* **Summary plot** shows global and individual effects.

---

## 11. Interpreting the Metrics

* **Baseline LinearRegression**

  * RMSE = 10 M €, R² = 0.436 → linear model explains \~43.6% of price variance.

* **Ridge & Lasso**

  * RMSE ≈ 6 M €, R² ≈ 0.80 → regularization cuts error by \~40%.

* **Random Forest & XGBoost**

  * RF: RMSE ≈ 6 M €, R² ≈ 0.826
  * XGB: **RMSE ≈ 4 M €, R² ≈ 0.908** ← **best performer**

---

### Why These Outputs?

* **RMSE** (absolute error) speaks directly to business impact: “On average, our model is off by €4 M.”
* **R²** (relative fit) tells data scientists how much of the predictable signal in the data has been captured.
* **Feature importances** guide feature engineering priorities.
* **SHAP** provides transparency for model-driven decisions.

---
