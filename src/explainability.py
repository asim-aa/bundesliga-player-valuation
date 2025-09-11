#!/usr/bin/env python3
"""
Explainability utilities for the Bundesliga Player Valuation project.

Features:
- Global feature importance for tree-based models (feature_importances_)
- Global coefficients for linear models (|coef|)
- Optional SHAP summary (if `shap` is installed):
  - LinearExplainer for linear models
  - TreeExplainer for tree-based models

Usage examples:
  python src/explainability.py --save outputs/feature_importance.png
  python src/explainability.py --shap --save-shap outputs/shap_summary.png
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap  # optional
except Exception:  # pragma: no cover
    shap = None

from model_pipeline import (
    load_model,
    load_data,
    build_preprocessor,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def get_feature_names(preprocessor) -> np.ndarray:
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        # Fallback: numeric + OHE categories
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                ohe = trans.named_steps.get("onehot")
                if hasattr(ohe, "get_feature_names_out"):
                    names.extend(ohe.get_feature_names_out(cols))
                else:
                    names.extend(cols)
        return np.array(names)


def plot_feature_importance(model, feature_names, top=20, save_path=None):
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        title = "Feature Importances (tree-based)"
    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        importances = np.abs(coef)
        title = "Feature Importances (|coef|)"
    else:
        raise ValueError("Regressor has neither feature_importances_ nor coef_.")

    order = np.argsort(importances)[::-1][:top]
    fn, vals = feature_names[order], importances[order]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(fn)), vals[::-1])
    plt.yticks(range(len(fn)), fn[::-1])
    plt.title(title)
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        print(f"✔ Saved feature importance to {out}")
    else:
        plt.show()


def shap_summary(model, X_trans, feature_names, save_path=None):
    if shap is None:
        print("shap not installed; skipping SHAP summary.")
        return
    try:
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_trans, feature_dependence="independent")
            sv = explainer.shap_values(X_trans)
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_trans)
        shap.summary_plot(sv, X_trans, feature_names=feature_names, show=False)
        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"✔ Saved SHAP summary to {out}")
        else:
            plt.show()
    except Exception as e:  # pragma: no cover
        print(f"SHAP failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explain trained pipeline.")
    parser.add_argument("--csv", default="data/processed/players_features.csv")
    parser.add_argument("--model", default="models/best_pipeline.pkl")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--save", default=None, help="Path to save feature-importance plot")
    parser.add_argument("--shap", action="store_true", help="Compute SHAP summary plot")
    parser.add_argument("--save-shap", default=None, help="Path to save SHAP summary plot")
    args = parser.parse_args()

    pipe = load_model(args.model)
    pre = pipe.named_steps["preprocessor"]
    reg = pipe.named_steps.get("regressor") or pipe.named_steps.get("model")

    df = load_data(args.csv)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_trans = pre.fit_transform(X) if not hasattr(pre, "transformers_") else pre.transform(X)
    feature_names = get_feature_names(pre)

    plot_feature_importance(reg, feature_names=np.array(feature_names), top=args.top, save_path=args.save)

    if args.shap:
        # Use a subset to keep plots readable
        n = min(500, X_trans.shape[0])
        shap_summary(reg, X_trans[:n], feature_names=np.array(feature_names), save_path=args.save_shap)


if __name__ == "__main__":
    main()
