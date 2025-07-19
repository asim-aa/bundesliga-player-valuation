# Methods

**Purpose:** Deep dive on data ingestion, exploratory data analysis (EDA), feature engineering, preprocessing, and modeling choices.

**Why:** Readers interested in “how it works” can skip the rest.

---

## 1. Data Ingestion & Exploratory Analysis

### Concept  
Before modeling, we must load raw data, clean it, and explore its key patterns and quirks. EDA helps us spot outliers, understand distributions, and decide how to handle missing values.

### Implementation in This Project  
1. **Loading & Cleaning**  
   - **File:** `src/load_data.py`  
   - Read raw CSV:  
     ```python
     df = pd.read_csv("data/raw/bundesliga_players.csv")
     ```  
   - Parse dates and normalize price strings (“€”, “M”, “K”) to floats.  
   - Impute missing categorical fields (e.g. agent, outfitter) with `"Unknown"`.  
   - Write cleaned output to `data/processed/players_clean.csv`.

2. **EDA Notebooks**  
   - **File:** `notebooks/eda.ipynb`  
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
   - **File:** `src/pipeline.py`  
   - Numeric pipeline: median imputation → standard scaling.  
   - Categorical pipeline: most-frequent imputation → one-hot encoding.  
   - Assembled into a `ColumnTransformer`:  
     ```python
     numeric_feats = ["age", "days_remaining", "height", "past_price_mean"]
     categorical_feats = ["position", "club", "foot"]
     preprocessor = ColumnTransformer([
       ("num", Pipeline([("imp", SimpleImputer("median")), ("sc", StandardScaler())]), numeric_feats),
       ("cat", Pipeline([("imp", SimpleImputer("most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_feats)
     ])
     ```

2. **Domain-Driven Features**  
   - **File:** `src/prediction.py`  
   - **Age-decay multiplier:**  
     - Ages 18–26: multiplier = 1.0 (growth plateau)  
     - Ages 26–32: linear decline to multiplier = 0.7  
     - >32: exponential decay: `mult *= exp(-(age-32)/5)`

3. **Enriching Age Data Using Wikipedia API**  
   - **File:** `src/fetch_player_ages.py`  
   - To ensure accurate and consistent age values, we used the **Wikipedia API** to fetch each player's birthdate and calculate their current age:
     - Queried Wikipedia for each player's full name using the `wikipedia` Python library or `requests` with `wikipediaapi`.
     - Parsed the first sentence of the page summary to extract the date of birth using regular expressions.
     - Computed age as the difference between today’s date and the extracted birthdate.
     - Example:  
       For `"Jamal Musiala"` → summary:  
       _"Jamal Musiala (born 26 February 2003)..."_  
       → extracted DOB: `2003-02-26`  
       → calculated age: `22`
     - The resulting age was added to the dataset for modeling.
   - This process improved the quality of our age feature, which is a key variable in player valuation.

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
         gs = GridSearchCV(Pipeline([("pre", preprocessor), ("model", spec["estimator"])]),
                           spec["params"], cv=5, scoring="neg_root_mean_squared_error")
         gs.fit(X_train, y_train)
         record_result(name, gs.best_params_, gs.best_score_, evaluate_on_test(gs.best_estimator_))
     ```
2. **Persisting the Best Model**  
   - Save winner with `joblib.dump(...)` to `models/best_model.pkl`.

---

## 5. Evaluation Metrics

### Concept  
- **RMSE (Root Mean Squared Error):** average magnitude of prediction errors, in the same units as market value.  
- **R² (Coefficient of Determination):** proportion of variance explained by the model.

### Implementation in This Project  
- Printed after each training run.  
- Final comparison chart in `notebooks/final_report.ipynb`.

---

## 6. Deployment & Prediction Pipeline

### Concept  
Bundle preprocessing and prediction logic into one serializable object and expose a simple CLI.

### Implementation in This Project  
1. **Prediction Script**  
   - **File:** `src/predict_player_progression.py`  
   - Loads `models/best_model.pkl` and `data/processed/players_clean.csv`.  
   - Parses arguments via `argparse` (`--start-date`, `--periods`, `--freq`, `player_name`).  
   - Outputs a table or plot of projected market values.

---

## 7. Key Learnings & Reflections

- **Feature importance:** age and past price trends dominated model decisions.  
- **Regularization impact:** Ridge with α=0.1 balanced bias-variance best.  
- **Domain integration:** age-decay multiplier improved early/player peak modeling by ~5% RMSE.  
- **Wikipedia API usage:** dynamic enrichment of player metadata via public APIs taught me to blend data science with lightweight web scraping.  
- **Next steps:** incorporate in-game performance metrics (e.g. xG, defensive actions) to refine value estimates.

---

*End of methods deep dive.*
