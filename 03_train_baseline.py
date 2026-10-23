"""
03_train_baseline.py  -  PHASE 5: first, simple models on the region table.

Three models are compared, all tested with Leave-One-Patient-Out (LOPO) cross-validation:
train on 29 patients, test on the 1 patient left out, repeat 30 times. A patient is never
in training and test at the same time - this is the single most important rule.

  region_prior   : "how often was this region epileptogenic in the other patients?"
                   (uses NO imaging - any real model has to beat this)
  logreg         : logistic regression on the MRI features
  random_forest  : random forest on the MRI features

Outputs:
  outputs/baseline_predictions.csv   (regions with electrodes, out-of-fold probabilities)
  outputs/baseline_all_regions.csv   (every region of every patient, for brain heat-maps)
  outputs/baseline_metrics.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
import config
import runlog


def make_models():
    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                              class_weight="balanced_subsample",
                                                              random_state=config.SEED, n_jobs=-1)),
    }


def region_prior(train, names, alpha=2.0):
    """Smoothed positive rate of each region name in the training patients."""
    overall = train["target"].mean()
    stats = train.groupby("region_name")["target"].agg(["sum", "count"])
    rate = (stats["sum"] + alpha * overall) / (stats["count"] + alpha)
    return names.map(rate).fillna(overall).values


def main():
    table = pd.read_csv(os.path.join(config.OUT_DIR, "region_table.csv"))
    feat_cols = [c for c in table.columns if c.startswith("f_")]
    lab = table[table["sampled"]].copy()
    lab["target"] = lab["target"].astype(int)
    subjects = sorted(lab["subject"].unique())
    print(f"LOPO over {len(subjects)} patients, {len(lab)} labelled regions, {len(feat_cols)} features")

    model_names = ["region_prior"] + list(make_models())
    for m in model_names:
        lab["p_" + m] = np.nan
        table["p_" + m] = np.nan

    for test_sub in subjects:
        train = lab[lab["subject"] != test_sub]
        test_idx = lab.index[lab["subject"] == test_sub]
        all_idx = table.index[table["subject"] == test_sub]
        if train["target"].nunique() < 2:
            continue
        lab.loc[test_idx, "p_region_prior"] = region_prior(train, lab.loc[test_idx, "region_name"])
        table.loc[all_idx, "p_region_prior"] = region_prior(train, table.loc[all_idx, "region_name"])
        for name, model in make_models().items():
            model.fit(train[feat_cols], train["target"])
            lab.loc[test_idx, "p_" + name] = model.predict_proba(lab.loc[test_idx, feat_cols])[:, 1]
            table.loc[all_idx, "p_" + name] = model.predict_proba(table.loc[all_idx, feat_cols])[:, 1]

    metrics = pd.DataFrame([{"model": m, **common.summarize(lab, "p_" + m)} for m in model_names])
    print("\nOut-of-fold results (regions with electrodes):")
    print(metrics.to_string(index=False))
    print("\nReading the numbers:")
    print(" * AUROC 0.5 = coin flip, 1.0 = perfect.")
    print(f" * AUPRC must be compared with the prevalence ({lab['target'].mean():.2f}) - that is its 'coin flip'.")
    print(" * top3_hit = in how many patients a true positive region is among the model's 3 highest-ranked regions.")
    print(" * If the MRI models do not beat 'region_prior', the imaging features are not adding information yet.")

    # which features matter? (forest fitted on all labelled data, for interpretation only)
    rf = make_models()["random_forest"].fit(lab[feat_cols], lab["target"])
    imp = pd.Series(rf[-1].feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\nMost useful features (random forest importance):")
    print(imp.head(10).round(3).to_string())

    out = config.OUT_DIR
    lab.to_csv(os.path.join(out, "baseline_predictions.csv"), index=False)
    table.to_csv(os.path.join(out, "baseline_all_regions.csv"), index=False)
    metrics.to_csv(os.path.join(out, "baseline_metrics.csv"), index=False)
    print(f"\nSaved predictions and metrics to {out}/")


if __name__ == "__main__":
    with runlog.Log("03_train_baseline"):
        main()
