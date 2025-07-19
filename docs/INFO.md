# Bundesliga Player Valuation

Welcome to the Bundesliga Player Valuation project! This is where my two greatest passions collide: the beautiful game of soccer and the elegant world of mathematics.

## My Passion for Soccer
Soccer has been my lifelong love. I grew up cheering in packed stadiums, celebrating thrilling victories, and sharing the excitement of every goal with friends and family. The passion, the drama, and the skill on display in each match have always inspired me.

## Curiosity for the Numbers
As I watched the game, I began to wonder: what makes some players stand out statistically? Can we use data to uncover which defender consistently thwarts attacks, or which midfielder quietly controls possession? This curiosity drove me to explore the hidden numbers behind every pass, tackle, and goal.

## The Intersection of Math and Soccer
With a strong background in mathematics and data analysis, I saw a natural opportunity to blend these interests. In this project, I use statistical techniques and programming to translate raw player data into meaningful insights about player value. By combining soccer knowledge and quantitative methods, I aim to reveal the metrics that truly matter on the pitch.

---

## Project Workflow
Here is a simple, step-by-step overview of how the project is organized, explained in clear terms for anyone to follow:

### 1. Setting Up the Project
I started by creating a GitHub repository called **`bundesliga-player-valuation`**. Inside this repository, I organized folders for raw data, cleaned data, exploratory work, scripts, and outputs. This structure keeps everything neat and makes it easy to collaborate or share with hiring managers.

### 2. Bringing in the Raw Data
Next, I downloaded the full Bundesliga player dataset and placed the original CSV file into a folder named **`data/raw/`**. Then, I wrote a small Python script (`src/load_data.py`) that reads this file and prints a quick preview. This script shows the first few rows, the data types, and basic statistics, ensuring that the data loaded correctly.

### 3. Cleaning and Preparing the Data
Raw data is rarely perfect, so the third step is all about cleaning. I created a script (`src/clean_data.py`) that:

- Converts price tags like “€12 M” into actual numbers.
- Turns date fields (such as contract start and end dates) into date objects the computer can understand.
- Deals with missing values by either filling in sensible defaults or removing incomplete records.
- Standardizes text fields (like agent names and club names) so they all follow the same format.

This process outputs a tidy file in **`data/processed/players_clean.csv`**, ready for deeper analysis.

### 4. Creating New Features
With clean data in hand, I moved on to feature engineering. This means deriving new, useful columns that capture important insights. For example:

- **Tenure Years:** How long each player has been with their current club.
- **Price Ratio:** A comparison of a player’s current value against their maximum observed value.
- **Age Group:** Simple categories such as “Under 21” or “30+” to separate players by generation.
- **Height Category:** Labels like “Short,” “Average,” or “Tall” to quickly assess physical profiles.

I put this logic into `src/features/build_features.py` and saved the enhanced dataset to **`data/processed/players_features.csv`**.

---

With these steps complete, the project is ready for the next phases—which include building predictive models, creating visualizations, and drawing compelling insights about the most valuable players in the Bundesliga. Feel free to explore the code, review the data files, and see how analytics can bring a fresh perspective to the beautiful game!
