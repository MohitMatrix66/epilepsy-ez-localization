"""
07_lateralization.py  -  EXTRA: from which SIDE of the brain do the seizures come? (all patients with a clinical answer)

Why: the iEEG-derived SOZ labels exist for only ~10 patients. The clinical table (metadata_3T_7T.xlsx)
gives the seizure-onset side (Left / Right) for about 25 of the 30 patients, decided by the epilepsy team.
This script asks a simpler question with more patients: can the MRI alone tell the side?

Features (one row per patient): for every pair of matching left / right regions, the asymmetry
    (left - right) / ((left + right) / 2)
of the region's size and its T1 brightness (T2 too if every patient has it) -> about 80 numbers.
Models, all tested leave-one-patient-out (the tested patient is never used for training):
    majority          always guess the more common side of all patients (the "coin flip" to beat; it uses no MRI.
                      Not re-fitted per left-out patient: that would always guess against the left-out patient)
    hippocampus rule  the side with the SMALLER hippocampus (a classic sign of mesial temporal sclerosis); no training
    logreg            logistic regression on all asymmetries (strongly regularised)
    side_model        VERSION 9 - the side model, built like the published hippocampal-sclerosis models (AID-HS,
                      Ann Neurol 2025; Caldairou et al., Neurology 2021): a small logistic regression on FIVE left-right
                      measures chosen in advance (common.SIDE_FEATURES): hippocampus, amygdala and thalamus volume,
                      hippocampus T2 brightness (relative to the white matter of the same side; 25 of 30 patients have
                      a T2 - the others get 'no difference'), temporal-cortex thickness (FreeSurfer surfaces, when the
                      dataset has them). It says 'not sure' when its probability is between 0.4 and 0.6, like a
                      radiologist writing 'non-lateralizing'. Results are also shown separately for the patients whose
                      MRI report names hippocampal sclerosis and for the others.

Run:   python 07_lateralization.py
Output: outputs/lateralization_predictions.csv (one row per patient), outputs/lateralization_metrics.csv,
        outputs/side_model.json (the side model trained on every patient with a clinical side, for 08_predict_mri.py)
Downloads FreeSurfer files of every patient (~2-3 MB each, plus the surface thickness files) the first time.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import aws_data
import common
import config
import runlog

EPS = 1e-6
SIDE_C = 0.3          # regularisation of the side model (fixed in advance, not tuned)
NOT_SURE = 0.1        # 'not sure' when |p(left) - 0.5| < NOT_SURE


def clinical_sides():
    """{subject: 'L' / 'R'} from the clinical table ('Left only' / 'Left > Right' = L, and the same for R)."""
    df = aws_data.metadata_table()
    if df is None:
        raise SystemExit("metadata_3T_7T.xlsx not found - run SOZ\\01_scan_dataset.py first.")
    cols = {str(c).strip().lower().rstrip(":").strip(): c for c in df.columns}
    sub_col = next((c for k, c in cols.items() if "subject" in k or "participant" in k), None)
    lat_col = next((c for k, c in cols.items() if "onset lateral" in k), None)
    if sub_col is None or lat_col is None:
        raise SystemExit("metadata_3T_7T.xlsx has no subject / seizure-onset-lateralization column.")
    out = {}
    for sid, lat in zip(df[sub_col], df[lat_col]):
        sid, low = str(sid).strip(), str(lat).strip().lower()
        if not sid or sid.lower() == "nan":
            continue
        sid = sid if sid.startswith("sub-") else "sub-" + sid
        side = "L" if low.startswith(("left only", "left >")) else "R" if low.startswith(("right only", "right >")) else ""
        if side:
            out[sid] = side
    return out


def asymmetry_features(sub, use_t2, lab=None, t2=None):
    lab = np.asarray(common.load_mgz(sub, "aparc+aseg.mgz").dataobj).astype(np.int32) if lab is None else lab
    vols = {"t1": np.asarray(common.load_mgz(sub, "norm.mgz").dataobj).astype(np.float64)}
    if use_t2:
        vols["t2"] = (np.asarray(common.load_mgz(sub, "T2.mgz").dataobj) if t2 is None else t2).astype(np.float64)
    brain = lab > 0
    n = int(lab.max()) + 1
    cnt = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    means = {k: np.bincount(lab.ravel(), weights=v.ravel(), minlength=n) / np.maximum(cnt, 1) / v[brain].mean()
             for k, v in vols.items()}
    f = {}
    for rid in common.REGION_IDS:
        other = common.homolog(int(rid))
        if common.hemisphere(int(rid)) != "L" or other is None or rid >= n or other >= n:
            continue
        name = common.REGIONS[int(rid)].replace("ctx-lh-", "").replace("Left-", "")
        pairs = {"volume": (cnt[rid], cnt[other])}
        pairs.update({k: (m[rid], m[other]) for k, m in means.items()})
        for kind, (a, b) in pairs.items():
            f[f"asym_{kind}_{name}"] = (a - b) / ((abs(a) + abs(b)) / 2 + EPS) if (a > 0 and b > 0) else np.nan
    return f


def side_model_fit(X, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=SIDE_C, class_weight="balanced", max_iter=5000))
    return model.fit(X, y)


def side_call(p):
    return "not sure" if abs(p - 0.5) < NOT_SURE else ("L" if p >= 0.5 else "R")


def main():
    sides = clinical_sides()
    mri = common.mri_record()
    everyone = [s for s in aws_data.subjects() if common.has_mri(s, "aparc+aseg.mgz") and common.has_mri(s, "norm.mgz")]
    have = [s for s in everyone if s in sides]
    no_t2 = [s for s in have if not common.has_mri(s, "T2.mgz")] if config.USE_T2 else have
    use_t2 = config.USE_T2 and not no_t2
    print(f"{len(have)} patients with a clinical seizure-onset side and FreeSurfer files "
          f"(L {sum(sides[s] == 'L' for s in have)}, R {sum(sides[s] == 'R' for s in have)});  "
          f"{len(everyone) - len(have)} more without a clinical side;  T2 features (all-region model): "
          f"{'yes' if use_t2 else 'no'}")
    rows = []
    for sub in everyone:
        lab, t2, surf, label = common.side_inputs(sub)
        sf = common.side_features(lab, t2, surf, label)
        res, lesion, lesion_side = mri.get(sub, ("", "", ""))
        rows.append({"subject": sub, "side": sides.get(sub, ""), "mri_result": res, "mri_lesion": lesion,
                     "mri_lesion_side": lesion_side, "mts": common.is_mts(lesion),
                     **asymmetry_features(sub, use_t2, lab, t2), **{"side_" + k: v for k, v in sf.items()}})
        got = [k for k, v in sf.items() if np.isfinite(v)]
        print(f"  {sub}: done (side-model measures available: {len(got)} of {len(sf)})")
    full = pd.DataFrame(rows).reset_index(drop=True)
    df = full[full["side"] != ""].reset_index(drop=True)
    feat = [c for c in df.columns if c.startswith("asym_")]
    X = df[feat].fillna(0.0).to_numpy()
    y = (df["side"] == "L").astype(int).to_numpy()   # 1 = left
    side_cols = ["side_" + k for k in common.SIDE_FEATURES]
    XS = df[side_cols].fillna(0.0).to_numpy()            # missing measure = 'no difference'

    hip = "asym_volume_Hippocampus"
    df["p_left_hippocampus_rule"] = np.where(df[hip] < 0, 1.0, 0.0) if hip in df else np.nan   # smaller left -> left
    df["p_left_majority"] = float(y.mean() >= 0.5)   # a constant guess: balanced accuracy 0.5 by construction
    df["p_left_logreg"] = np.nan
    df["p_left_side_model"] = np.nan
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))
        model.fit(X[tr], y[tr])
        df.loc[i, "p_left_logreg"] = model.predict_proba(X[i:i + 1])[0, 1]
        df.loc[i, "p_left_side_model"] = side_model_fit(XS[tr], y[tr]).predict_proba(XS[i:i + 1])[0, 1]
    df["side_model_call"] = [side_call(p) if np.isfinite(p) else "" for p in df["p_left_side_model"]]

    def score(name, p, yy):
        ok = np.isfinite(p)
        pred = (p[ok] >= 0.5).astype(int)
        acc = float((pred == yy[ok]).mean()) if ok.any() else np.nan
        bal = float(np.mean([(pred[yy[ok] == c] == c).mean() for c in (0, 1) if (yy[ok] == c).any()])) if ok.any() \
            else np.nan
        auc = roc_auc_score(yy[ok], p[ok]) if ok.any() and len(np.unique(yy[ok])) == 2 and len(np.unique(p[ok])) > 2 \
            else np.nan
        return {"model": name, "patients": int(ok.sum()), "accuracy": round(acc, 3), "balanced_accuracy": round(bal, 3),
                "AUROC": round(auc, 3) if np.isfinite(auc) else np.nan}

    rows = [score(m, df[f"p_left_{m}"].to_numpy(dtype=float), y)
            for m in ("majority", "hippocampus_rule", "logreg", "side_model")]
    ps = df["p_left_side_model"].to_numpy(dtype=float)
    for label, sel in (("side_model: MRI report = hippocampal sclerosis", df["mts"].to_numpy(bool)),
                       ("side_model: all other patients", ~df["mts"].to_numpy(bool))):
        if sel.any():
            rows.append(score(label, ps[sel], y[sel]))
    sure = np.array([c in ("L", "R") for c in df["side_model_call"]])
    if sure.any():
        r = score("side_model: only when sure", ps[sure], y[sure])
        r["model"] += f" ({int(sure.sum())} of {len(df)} patients)"
        rows.append(r)
    metrics = pd.DataFrame(rows)

    # the side model trained on every patient with a clinical side (for 08_predict_mri.py and the other patients)
    final = side_model_fit(XS, y)
    scaler, lr = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
    card = {"version": "9", "features": common.SIDE_FEATURES, "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "not_sure": NOT_SURE, "C": SIDE_C, "trained_on": df["subject"].tolist()}
    with open(os.path.join(config.OUT_DIR, "side_model.json"), "w") as fh:
        json.dump(card, fh, indent=1)
    rest = full[full["side"] == ""].copy()
    if len(rest):
        rest["p_left_side_model"] = final.predict_proba(rest[side_cols].fillna(0.0).to_numpy())[:, 1]
        rest["side_model_call"] = [side_call(p) for p in rest["p_left_side_model"]]
    out = pd.concat([df, rest], ignore_index=True)
    out.to_csv(os.path.join(config.OUT_DIR, "lateralization_predictions.csv"), index=False)
    metrics.to_csv(os.path.join(config.OUT_DIR, "lateralization_metrics.csv"), index=False)
    print("\nLeave-one-patient-out results (predicting the clinical seizure-onset side from the MRI):")
    print(metrics.to_string(index=False))
    show = df[["subject", "side", "mri_lesion", "side_model_call", "p_left_side_model"]].copy()
    show["p_left_side_model"] = show["p_left_side_model"].round(2)
    show["mri_lesion"] = show["mri_lesion"].str.slice(0, 30)
    print("\nSide model, every patient (tested by a model that never saw him/her):")
    print(show.to_string(index=False))
    if len(rest):
        print("Patients without a clinical side (final side model): "
              + ", ".join(f"{r.subject} {r.side_model_call}" for r in rest.itertuples()))
    print("Side model weights (+ = points to the left when the left value is larger): "
          + ", ".join(f"{k.replace('asym_', '')} {c:+.2f}" for k, c in zip(common.SIDE_FEATURES, card["coef"])))
    print("\nHow to read this:")
    print(" * 'majority' always says the more common side: any useful model must beat its balanced accuracy (0.5).")
    print(" * balanced accuracy = average of 'left patients correct' and 'right patients correct'; AUROC = the fair")
    print("   score that does not depend on a model's lean to one side (0.5 = guessing).")
    print(" * With ~25 patients one patient = 4%, so differences of a few percent are not meaningful.")
    print(f"Saved lateralization_predictions.csv, lateralization_metrics.csv and side_model.json in {config.OUT_DIR}")


if __name__ == "__main__":
    with runlog.Log("07_lateralization"):
        main()
