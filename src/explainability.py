#!/usr/bin/env python3
import shap
import numpy as np

def tree_feature_importance(pipeline):
    rf = pipeline.named_steps["rf"]
    pre = pipeline.named_steps["preproc"]
    numeric_cols   = pre.transformers_[0][2]
    cat_steps      = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_cols       = cat_steps.get_feature_names_out(pre.transformers_[1][2])
    feature_names  = list(numeric_cols) + list(cat_cols)

    importances = rf.feature_importances_
    sorted_feats = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_feats[:10]:
        print(f"{name}: {imp:.3f}")

def shap_explain(pipeline, X_train, X_test, idx=0):
    rf = pipeline.named_steps["rf"]
    pre = pipeline.named_steps["preproc"]
    X_train_t = pre.transform(X_train)
    X_test_t  = pre.transform(X_test)

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_train_t)

    # summary
    shap.summary_plot(sv, X_train_t, feature_names=pre.get_feature_names_out())

    # single-instance force plot
    shap.force_plot(explainer.expected_value, sv[idx], X_test_t[idx],
                    feature_names=pre.get_feature_names_out(), matplotlib=True)
