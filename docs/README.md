# Bundesliga Player Valuation

This project brings together my lifelong passion for soccer and my training in mathematics–computer science.  
The goal: **quantify and forecast Bundesliga player market value** using a reproducible data-science pipeline.  

---

## Why I Built This
Growing up, I lived and breathed the game — but as my math background deepened, I started asking sharper questions:  

- What *really* makes one player more valuable than another?  
- Can numbers reveal hidden strengths or undervalued talent?  
- How do factors like age, position, or career trajectory shape future value?  

This project is my attempt to answer those questions — not just as a fan, but as a data scientist turning raw match sheets into actionable insights.  

---

## Project Goals
- **Data ingestion & cleaning**: turn messy player data into a structured dataset.  
- **Feature engineering**: capture tenure, contract details, performance ratios, age progression, and more.  
- **Modeling & prediction**: compare regressors (linear, regularized, tree-based) to forecast market value.  
- **Explainability**: highlight which features truly drive valuation.  
- **Interface**: provide a simple CLI to generate forecasts for any player.  

---

## Repo Structure
```bash
bundesliga-player-valuation/
├── data/               # raw + processed player data
├── notebooks/          # EDA & visualization
├── src/                # all Python source code
│   ├── load_data.py
│   ├── clean_data.py
│   ├── features/...
│   ├── baseline_model.py
│   ├── advanced_models.py
│   ├── final.py        # end-to-end pipeline
│   └── cli.py          # command line interface
└── docs/               # METHODS.md, RESULTS.md, USAGE.md, etc.
