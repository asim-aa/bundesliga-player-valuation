
## 1. Data Ingestion & Cleaning

### Raw Data
- **File:** `data/raw/bundesliga_player.csv`
- **Source:** scraped/downloaded Bundesliga player market value dataset.
- **Issues Observed:**
  - Inconsistent player names (spelling/formatting variations).
  - Duplicate rows across seasons.
  - Missing or outdated fields (especially age).
  - Market value recorded as strings with currency symbols.

### Cleaning Process
- **Script:** `src/clean_data.py`
- **Output:** `data/processed/players_clean.csv`
- **Steps Applied:**
  - Dropped irrelevant columns.
  - Normalized player names and created stable IDs for joins.
  - Converted market value strings → numeric (in € millions).
  - Parsed date fields into standard ISO format.
  - Imputed or dropped missing records based on completeness.
  - Standardized categorical fields (positions, clubs).

---

## 2. Feature Engineering & Enrichment

### Engineered Features
- **Script:** `src/pipeline.py` + `src/features/`
- **Output:** `data/processed/players_features.csv`
- **Key Features Added:**
  - **Age Group Buckets**: U21, 21–24, 25–29, 30+.  
  - **Minutes Played Categories**: low, medium, high buckets.  
  - **Domain Feature**: “Age decay” multiplier to capture value drop after peak age.

---

### 🧠 Player Age Retrieval Using Wikipedia API
- **Script:** `src/fetch_birthdays_from_csv.py`  
- **Output:** `fetch-scriptsbirthdays_output.csv`

#### Why This Was Necessary
The raw dataset did not consistently include up-to-date player ages. Since age is one of the strongest predictors of market value, we needed a reliable, automated source of truth.

#### How It Works
1. **Construct Query:** For each player’s name, build a request to the Wikipedia API (via the `wikipedia` package or `requests` + `wikipediaapi`).  
2. **Fetch Page:** Retrieve the player’s page and extract the introductory sentence.  
3. **Parse Birthdate:** Use regex/NLP to detect birthdate strings in the format `(born DD Month YYYY)`.  
4. **Compute Age:** Calculate `today.year - birth.year`, adjusting if the birthday hasn’t occurred yet this year.  
5. **Integrate:** Append the calculated age to the player’s record and store in `players_features.csv`.

#### Example
- Input: `"Jamal Musiala"`  
- Wikipedia first line: *“Jamal Musiala (born 26 February 2003) is a German professional footballer...”*  
- Extracted DOB: `26 February 2003`  
- Computed Age (2025): **22**  
- Stored as `age = 22` in `players_features.csv`

#### Lessons Learned
- How to interact with public APIs for **data enrichment**.  
- Parsing semi-structured text with regex.  
- Importance of **validation and fallbacks** when API responses differ or pages are missing.  

**Tools Used:** `requests`, `wikipedia` / `wikipediaapi`, `datetime`, `re`  
**Skills Displayed:** API consumption, text parsing, feature engineering, error handling.  

---

## 3. Train/Test Split
- **Script:** `src/data_split.py`  
- **Outputs:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`  
- **Method:** reproducible `train_test_split` with stratification on target values to maintain distribution balance.  
- **Purpose:** ensure robust evaluation and prevent leakage.

---

## 4. Observations
- Raw data had skewed distribution of market values, with a long tail of outliers (superstars).  
- Missing or inconsistent ages would have biased younger/less documented players without enrichment.  
- Encoding positions and minutes provided useful signals for distinguishing player valuations.

---

## 5. Limitations
- Transfermarkt valuations are subjective and can shift due to non-performance factors (injuries, media hype).  
- Wikipedia parsing depends on page formatting; fallback handling was needed for missing or non-standard entries.  
- Lower-tier players often had incomplete data, reducing reliability for out-of-sample predictions.

