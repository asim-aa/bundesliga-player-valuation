# Methods

**Purpose:** Deep dive on data ingestion, exploratory data analysis (EDA), feature engineering, preprocessing, and modeling choices.
---

## 1. Data Ingestion & Exploratory Analysis

### Concept  
Before modeling, we must load raw data, clean it, and explore its key patterns and quirks. EDA helps us spot outliers, understand distributions, and decide how to handle missing values.

### Implementation in This Project  
1. **Loading & Cleaning**  
   - **File:** `src/load_data.py`  
   - Read raw CSV:  
     ```python
     df = pd.read_csv("data/raw/bundesliga_player.csv")
     ```  
   - Parse dates and normalize price strings (“€”, “M”, “K”) to floats.  
   - Impute missing categorical fields (e.g. agent, outfitter) with `"Unknown"`.  
   - Write cleaned output to `data/processed/players_clean.csv`.

2. **EDA Notebooks**  
   - **File:** `notebooks/04_exploratory_analysis.ipynb`  
   - **Univariate analysis:** histograms of `age`, `market_value` distributions.  
   - **Bivariate analysis:**  
     - Boxplots of value by `position`.  
     - Scatterplots of `age` vs. `market_value`.  
   - **Key insight:** players peak around ages 26–29, with long tails for exceptional youth talents.

---

## 2. Feature Engineering & Preprocessing

### Concept  
Transform raw columns into model-ready inputs: impute, scale, and encode. Augment features with domain knowledge (e.g. age-decay multipliers).

### Implementation in This Project  
1. **Pipeline Definition**  
   - **File:** `src/model_pipeline.py`  
   - Numeric pipeline: median imputation → standard scaling.  
   - Categorical pipeline: most-frequent imputation → one-hot encoding.  
   - Final feature lists and `ColumnTransformer` inside a sklearn `Pipeline` named `preprocessor`:  
     ```python
     NUMERIC_FEATURES = ['age', 'max_price_eur', 'price_to_max', 'tenure_years']
     CATEGORICAL_FEATURES = ['position', 'club', 'nationality']
     # numeric: median impute → scale; categorical: constant impute → one-hot
     ```

2. **Domain-Driven Features**  
   - **File:** `src/prediction.py`  
   - **Age-decay multiplier:**  
     - Ages 18–26: multiplier = 1.0 (growth plateau)  
     - Ages 26–32: linear decline to multiplier = 0.7  
     - >32: exponential decay: `mult *= exp(-(age-32)/5)`

3. **Enriching Age Data Using Wikipedia**  
   - **Folder:** `fetch-scripts/`  
   - Scripts like `fetch_birthdays_from_csv.py` can query Wikipedia to extract dates of birth and compute current ages. These utilities were used during data preparation as needed.

---

## 3. Modeling Approaches

### Concept  
Compare a simple baseline to more complex models. Baseline linear regression offers interpretability; regularized and ensemble methods capture non-linear patterns and control overfitting.

### Implementation in This Project  
1. **Baseline Linear Regression**  
   - **File:** `model_tuning_and_comparison.py`  
   - Pipeline: `preprocessor` → `LinearRegression()`  
   - Trained on 80/20 train/test split.  
   - Logged RMSE and R² on test set.

2. **Regularized Linear Models**  
   - Ridge and Lasso via `GridSearchCV`.  
   - Parameter grid: `{"model__alpha": [0.01, 0.1, 1, 10]}`.  
   - 5-fold cross-validation optimizing negative RMSE.

3. **Tree-Based Ensembles**  
   - **RandomForestRegressor:** tune `n_estimators`, `max_depth`.  
   - **XGBRegressor:** tune `learning_rate`, `subsample`, `colsample_bytree`.  
   - Each wrapped in the same preprocessing pipeline for fair comparison.

---

## 4. Hyperparameter Tuning & Model Selection

### Concept  
Use cross-validation to select hyperparameters that generalize best. Compare models on uniform metrics to choose the final estimator.

### Implementation in This Project  
1. **Grid Search Loop**  
   - **File:** `model_tuning_and_comparison.py`  
   - Iterate over model definitions and parameter grids:  
     ```python
     for name, spec in model_defs.items():
         gs = GridSearchCV(Pipeline([("preprocessor", preprocessor), ("model", spec["estimator"])]),
                           spec["params"], cv=5, scoring="neg_root_mean_squared_error")
         gs.fit(X_train, y_train)
         record_result(name, gs.best_params_, gs.best_score_, evaluate_on_test(gs.best_estimator_))
     ```
2. **Persisting the Best Model**  
   - `src/model_pipeline.py` trains and saves an end‑to‑end artifact at `models/best_pipeline.pkl` used by the CLI.

---

## 5. Evaluation Metrics

### Concept  
- **RMSE (Root Mean Squared Error):** average magnitude of prediction errors, in the same units as market value.  
- **R² (Coefficient of Determination):** proportion of variance explained by the model.

### Implementation in This Project  
- Printed after each training run.  
- Comparison tables and plots in `docs/RESULTS.md`.

---

## 6. Deployment & Prediction Pipeline

### Concept  
Bundle preprocessing and prediction logic into one serializable object and expose a simple CLI.

### Implementation in This Project  
1. **Prediction Script**  
   - **File:** `src/cli.py`  
   - Loads `models/best_pipeline.pkl` and `data/processed/players_features.csv`.  
   - Parses arguments via `argparse` (`--interactive`, `--years/--periods`, `--freq`, `--save`, `--no-show`).  
   - Displays a plot and can save it for headless runs.

---

## 7. Key Learnings & Reflections

- **Feature importance:** age and past price trends dominated model decisions.  
- **Regularization impact:** Ridge with α=0.1 balanced bias-variance best.  
- **Domain integration:** age-decay multiplier improved early/player peak modeling by ~5% RMSE.  
- **Wikipedia API usage:** dynamic enrichment of player metadata via public APIs taught me to blend data science with lightweight web scraping.  
- **Next steps:** incorporate in-game performance metrics (e.g. xG, defensive actions) to refine value estimates.

---
