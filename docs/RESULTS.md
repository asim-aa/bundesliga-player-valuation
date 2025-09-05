# Results

This page summarizes model selection results and provides instructions to reproduce metrics and plots.

## Reproduce

Run the tuning script, which performs GridSearchCV over several estimators with a shared preprocessing pipeline and evaluates on a held-out test set.

```
python src/model_tuning_and_comparison.py
```

You will see, for each model:
- Best hyperparameters (from cross-validation)
- Test RMSE and R²

At the end, a sorted comparison table is printed and two bar plots (RMSE, R²) are displayed.

## Headline Metrics (latest run)

- Best model: Lasso
- Test RMSE: 0.000960
- Test R²:   1.000000
- Test samples (n): 102
- Best params: `{'model__alpha': 0.001}`

## Example Console Snippet

```
→ Tuning Ridge…
  • Best params: {'model__alpha': 0.1}
  • Test RMSE = 0.290, R² = 1.000

→ Tuning Lasso…
  • Best params: {'model__alpha': 0.001}
  • Test RMSE = 0.001, R² = 1.000

→ Tuning RandomForest…
  • Best params: {'model__max_depth': 5, 'model__n_estimators': 100}
  • Test RMSE = 0.684, R² = 0.998

→ Tuning XGBoost…
  • Best params: {'model__learning_rate': 0.1, 'model__n_estimators': 200, 'model__subsample': 1.0}
  • Test RMSE = 0.312, R² = 0.999

=== All models compared ===
          model      rmse        r2
0         Lasso  0.000960  1.000000
1         Ridge  0.289710  0.999557
2       XGBoost  0.312350  0.999485
3  RandomForest  0.683996  0.997529
```

## Plots

The script displays two bar plots:
- Test RMSE by model
- Test R² by model


