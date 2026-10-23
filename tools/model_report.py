"""
tools/model_report.py  -  how good is the model, patient by patient, and has it learned more than ANATOMY?
(version 3: fair region scores, an anatomy baseline for every site, and the left-right tests)

Run from the project folder AFTER training: 03_train_baseline.py on the laptop, and 05_train_cnn.py on Colab. First copy
the Colab results from Google Drive into the laptop's outputs folder: cnn_predictions_contacts.csv,
cnn_predictions_regions.csv, cnn_metrics.csv and logs\\05_train_cnn.txt. Then:
    python tools\\model_report.py
It prints the report and saves outputs\\model_report.txt and outputs\\for_claude\\model_results.zip (the small result
tables, the training labels and the logs - no MRI, no patches, no model weights)  <- send that zip to Claude.
Nothing is changed or re-trained.

With about 10 patients one average number can be luck, so every number is per patient, with a 95% interval from
re-sampling the patients (bootstrap). AUROC within one patient = how well a score ranks that patient's SOZ sites above
the patient's other sites (1 = all SOZ sites first, 0.5 = random). Three questions, three tests:
  1. Does the model rank SOZ sites above the other sites?        -> against chance (0.5)
  2. Has it learned more than anatomy ("hippocampus is often SOZ")? -> against the ANATOMY PRIOR: for every site, how
     often sites in the same brain region were SOZ in the OTHER patients. Regions are scored with the MEAN of their
     sites, because the maximum rewards regions with many electrodes - and clinicians put more electrodes where they
     already suspect the seizures start, so the number of electrodes alone 'predicts' the SOZ without any MRI.
  3. Does it know WHICH SIDE is epileptic in this patient?        -> every SOZ site against the SAME place on the
     other side (its mirror, version 8: 05_train_cnn.py writes cnn_predictions_controls.csv); and, in patients with
     electrodes on both sides, are the model's highest sites on the SOZ side? A model that only knows anatomy cannot
     tell the two sides apart.
"""
import json
import math
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import config  # noqa: E402
import runlog  # noqa: E402

N_BOOT = 2000
REGION_MODELS = (("p_region_prior", "region prior (anatomy only)"), ("p_logreg", "logistic regression (MRI features)"),
                 ("p_random_forest", "random forest (MRI features)"), ("p_cnn_mean", "3D CNN (mean of its sites)"))


# =========================================================================== small helpers
def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def top_hit(y, p, k):
    """1 if a positive is among the k highest scores, 0 if not; NaN without a positive."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.sum() == 0 or not np.isfinite(p).any():
        return np.nan
    order = np.argsort(-np.nan_to_num(p, nan=-np.inf), kind="stable")[:k]
    return float(y[order].max())


def random_top_hit(n, s, k):
    """Chance that a random order puts at least one of s positives (out of n) in the top k."""
    if s == 0 or n == 0:
        return np.nan
    k = min(k, n)
    return 1.0 - (math.comb(n - s, k) / math.comb(n, k) if n - s >= k else 0.0)


def boot_ci(values, seed=0):
    """Mean of the per-patient values and its 95% interval from re-sampling the patients."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_ci(m, lo, hi, digits=2, sign=False):
    if not np.isfinite(m):
        return "-"
    s = "+" if sign else ""
    return f"{m:{s}.{digits}f} [{lo:{s}.{digits}f} to {hi:{s}.{digits}f}]"


def verdict(lo, hi, ref, better, worse, unclear):
    if not np.isfinite(lo):
        return "cannot be tested"
    return better if lo > ref else worse if hi < ref else unclear


def read(name):
    path = os.path.join(config.OUT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def side_of(region_name):
    r = str(region_name)
    return "L" if r.startswith(("Left-", "ctx-lh-")) else "R" if r.startswith(("Right-", "ctx-rh-")) else ""


def twin(region_name):
    """The same region on the other side ('Left-Hippocampus' <-> 'Right-Hippocampus')."""
    r = str(region_name)
    for a, b in (("Left-", "Right-"), ("ctx-lh-", "ctx-rh-")):
        if r.startswith(a):
            return b + r[len(a):]
        if r.startswith(b):
            return a + r[len(b):]
    return None


def anatomy_prior(sites, alpha=2.0):
    """For every site: the SOZ rate of sites in the same region in the OTHER patients (smoothed towards the overall
    rate; sites outside any region form the group 'none'). Leave-one-patient-out, like the models."""
    out = pd.Series(np.nan, index=sites.index)
    names = sites["region_name"].fillna("none").replace("", "none")
    for sub in sites["subject"].unique():
        train = sites["subject"] != sub
        overall = sites.loc[train, "target"].mean()
        st = sites[train].groupby(names[train])["target"].agg(["sum", "count"])
        rate = (st["sum"] + alpha * overall) / (st["count"] + alpha)
        out[~train] = names[~train].map(rate).fillna(overall).to_numpy()
    return out


def pack(lines):
    """Small result tables, the training labels, the reports and the logs -> outputs/for_claude/model_results.zip."""
    out_dir = os.path.join(config.OUT_DIR, "for_claude")
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, "model_results.zip")
    files = []
    for f in sorted(os.listdir(config.OUT_DIR)):
        p = os.path.join(config.OUT_DIR, f)
        if os.path.isfile(p) and f.endswith((".csv", ".txt", ".json")) and f != "model_report.txt" \
                and os.path.getsize(p) < 20e6:
            files.append((p, f"outputs/{f}"))
    pred_dir = os.path.join(config.OUT_DIR, "predictions")          # 08_predict_mri.py: tables, summaries, pictures
    if os.path.isdir(pred_dir):
        for root, _, names in os.walk(pred_dir):
            for f in sorted(names):
                if f.endswith((".csv", ".txt", ".png")):
                    p = os.path.join(root, f)
                    files.append((p, "outputs/" + os.path.relpath(p, config.OUT_DIR).replace(os.sep, "/")))
    for sub in ("logs", "heatmaps"):
        d = os.path.join(config.OUT_DIR, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith((".txt", ".png")) and f != "model_report.txt":   # (this run's own log is not finished)
                    files.append((os.path.join(d, f), f"outputs/{sub}/{f}"))
    for p in (config.CONTACTS_CSV, os.path.join(config.SOZ_RESULTS_DIR, "step3_report.txt"),
              os.path.join(config.SOZ_RESULTS_DIR, "step2_report.txt")):
        if os.path.exists(p):
            files.append((p, f"SOZ_results/{os.path.basename(p)}"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model_report.txt", "\n".join(lines) + "\n")
        for p, arc in files:
            z.write(p, arc)
    return zpath, len(files) + 1


# =========================================================================== the report
def main():
    lines, summary = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 104)
    say("MODEL REPORT (version 3) - every patient separately, 95% intervals from re-sampling the patients")
    say("=" * 104)
    card_path = os.path.join(config.OUT_DIR, "cnn_model.json")
    if os.path.exists(card_path):
        try:
            with open(card_path) as fh:
                card = json.load(fh)
            say(f"CNN model: version {card.get('version')}, channels {card.get('channels')}, cubes "
                f"{card.get('patch_size')} samples x {card.get('patch_mm', 'voxel')} mm, "
                f"{len(card.get('training_subjects', []))} training patients, final models: "
                f"{'yes' if (card.get('final') or {}).get('models') else 'no'}")
        except (ValueError, OSError):
            say("CNN model: cnn_model.json could not be read")
    cnn = read("cnn_predictions_contacts.csv")
    base = read("baseline_predictions.csv")
    lab = read("contacts_labeled.csv")
    if cnn is None and base is None:
        raise SystemExit("No results found in outputs - run 03_train_baseline.py (and 05_train_cnn.py on Colab, then "
                         "copy its result files into outputs) first.")

    sites = None
    if cnn is not None:
        sites = cnn.copy()
        if lab is not None:
            keep = [c for c in ("region_name", "mri_side") if c in lab.columns]
            sites = sites.merge(lab[["subject", "contact"] + keep].drop_duplicates(["subject", "contact"]),
                                on=["subject", "contact"], how="left")
        if "region_name" not in sites.columns:
            sites["region_name"] = "none"
        if "mri_side" not in sites.columns or sites["mri_side"].isna().all():
            sites["mri_side"] = sites["region_name"].map(side_of)
        sites["mri_side"] = sites["mri_side"].fillna("").astype(str)
        sites["p_anatomy"] = anatomy_prior(sites)
        missing = sites["p_cnn"].isna()
        if missing.any():
            subs = ", ".join(sorted(sites.loc[missing, "subject"].unique()))
            say(f"\nWARNING: {int(missing.sum())} sites have no CNN prediction ({subs}) - a training fold did not finish."
                " Run the Colab cell again (it continues where it stopped), then copy the files again.")

    # ------------------------------------------------------------------ 1. sites: chance and anatomy
    if sites is not None:
        per = []
        for sub, g in sites.groupby("subject"):
            n, s = len(g), int(g["target"].sum())
            per.append({"patient": sub.replace("sub-", ""), "sites": n, "SOZ": s,
                        "CNN AUROC": auroc(g["target"], g["p_cnn"]),
                        "anatomy AUROC": auroc(g["target"], g["p_anatomy"]),
                        "SOZ in CNN top 3": top_hit(g["target"].to_numpy(), g["p_cnn"].to_numpy(), 3),
                        "random top 3": random_top_hit(n, s, 3)})
        per = pd.DataFrame(per)
        say(f"\n1. SITES (contacts / pairs) - every patient tested by the model that never saw him/her "
            f"({len(per)} patients, {int(per['sites'].sum())} sites, {int(per['SOZ'].sum())} SOZ)")
        show = per.copy()
        for c in ("CNN AUROC", "anatomy AUROC"):
            show[c] = show[c].map(lambda v: "-" if not np.isfinite(v) else f"{v:.2f}")
        show["SOZ in CNN top 3"] = show["SOZ in CNN top 3"].map(
            lambda v: "-" if not np.isfinite(v) else ("yes" if v else "no"))
        show["random top 3"] = show["random top 3"].map(lambda v: "-" if not np.isfinite(v) else f"{v:.0%}")
        say(show.to_string(index=False))
        m, lo, hi = boot_ci(per["CNN AUROC"])
        ma, loa, hia = boot_ci(per["anatomy AUROC"])
        diff = (per["CNN AUROC"] - per["anatomy AUROC"]).to_numpy()
        md, lod, hid = boot_ci(diff)
        say(f"\n   CNN AUROC within patients      {fmt_ci(m, lo, hi)}   (chance = 0.50)")
        say(f"   anatomy prior AUROC            {fmt_ci(ma, loa, hia)}   (no MRI at all)")
        say(f"   CNN minus anatomy              {fmt_ci(md, lod, hid, sign=True)}   better in "
            f"{int((diff > 0).sum())} of {int(np.isfinite(diff).sum())} patients")
        hit = per["SOZ in CNN top 3"].dropna()
        say(f"   a SOZ site among the CNN's 3 highest sites: {int(hit.sum())} of {len(hit)} patients "
            f"(a random order: {per.loc[hit.index, 'random top 3'].mean():.0%})")
        v1 = verdict(lo, hi, 0.5, "YES", "NO (worse than chance)", "not clear")
        v2 = verdict(lod, hid, 0.0, "YES", "NO (worse than anatomy)", "NOT CLEAR")
        summary += [f"Ranks SOZ sites above chance:              {v1}",
                    f"Learned more than anatomy (sites):         {v2}"]

    # ------------------------------------------------------------------ 2. regions
    if base is not None:
        reg = base.copy()
        if sites is not None:
            mean_pool = (sites[sites["region_id"] > 0].groupby(["subject", "region_id"])["p_cnn"].mean()
                         .rename("p_cnn_mean").reset_index())
            reg = reg.merge(mean_pool, on=["subject", "region_id"], how="left")
        models = [(c, t) for c, t in REGION_MODELS if c in reg.columns and reg[c].notna().any()]
        if "n_contacts" in reg.columns:
            models.append(("n_contacts", "number of electrodes in it (no MRI)"))
        rows, by_model = [], {}
        for col, text in models:
            vals = {sub: auroc(g["target"], g[col]) for sub, g in reg.groupby("subject")}
            t1 = [top_hit(g["target"].to_numpy(), g[col].to_numpy(), 1) for _, g in reg.groupby("subject")]
            t1 = [x for x in t1 if np.isfinite(x)]
            by_model[col] = vals
            m, lo, hi = boot_ci(list(vals.values()))
            rows.append({"score": text, "AUROC within patients [95%]": fmt_ci(m, lo, hi),
                         "better than random": f"{sum(v > 0.5 for v in vals.values() if np.isfinite(v))} of "
                                               f"{sum(np.isfinite(v) for v in vals.values())}",
                         "SOZ region ranked 1st": f"{int(sum(t1))} of {len(t1)}"})
        say(f"\n2. BRAIN REGIONS WITH ELECTRODES - the same {len(reg)} regions of {reg['subject'].nunique()} patients "
            f"({int(reg['target'].sum())} SOZ regions)")
        say(pd.DataFrame(rows).to_string(index=False))
        if "n_contacts" in by_model:
            say("   (the last row uses no MRI: clinicians put more electrodes where they suspect the SOZ. That is why the "
                "CNN's\n   region score is the MEAN of its sites here - the maximum would reward regions with many "
                "electrodes.)")
        if "p_region_prior" in by_model and "p_cnn_mean" in by_model:
            prior, cnnm = by_model["p_region_prior"], by_model["p_cnn_mean"]
            d = [cnnm[s] - prior[s] for s in prior if np.isfinite(prior[s]) and np.isfinite(cnnm.get(s, np.nan))]
            m, lo, hi = boot_ci(d)
            say(f"   CNN minus region prior: {fmt_ci(m, lo, hi, sign=True)}, better in {sum(x > 0 for x in d)} of "
                f"{len(d)} patients")
            summary.append(f"Learned more than anatomy (regions):       "
                           f"{verdict(lo, hi, 0.0, 'YES', 'NO (worse than anatomy)', 'NOT CLEAR')}")

    # ------------------------------------------------------------------ 3. which side
    ctrl = read("cnn_predictions_controls.csv")
    if ctrl is not None and len(ctrl) and {"p_site", "p_control"} <= set(ctrl.columns):
        ctrl = ctrl.dropna(subset=["p_site", "p_control"])
        rows = []
        for sub, g in ctrl.groupby("subject"):
            wins = (g["p_site"] > g["p_control"]).astype(float) + 0.5 * (g["p_site"] == g["p_control"])
            rows.append({"patient": sub.replace("sub-", ""), "SOZ sites with a mirror": len(g),
                         "SOZ site scored higher than its mirror": float(wins.mean())})
        lr = pd.DataFrame(rows)
        say("\n3. LEFT-RIGHT TEST (every patient): is each SOZ site scored above the SAME place on the other side?")
        show = lr.copy()
        show["SOZ site scored higher than its mirror"] = show["SOZ site scored higher than its mirror"].map(
            lambda v: f"{v:.0%}")
        say(show.to_string(index=False))
        m, lo, hi = boot_ci(lr["SOZ site scored higher than its mirror"])
        say(f"   mean over patients {fmt_ci(m, lo, hi)} (0.50 = cannot tell the side; 1.00 = always the right side)")
        summary.append("Knows the epileptic side (site vs mirror): "
                       + verdict(lo, hi, 0.5, "YES", "NO - prefers the OTHER side", "NOT CLEAR"))
    if sites is not None:
        rows, side_auc, homolog = [], [], []
        for sub, g in sites.groupby("subject"):
            sides = g.loc[(g["target"] == 1) & g["mri_side"].isin(["L", "R"]), "mri_side"]
            if sides.empty:
                continue
            soz_side = sides.mode().iloc[0]
            known = g[g["mri_side"].isin(["L", "R"])]
            if known["mri_side"].nunique() < 2:
                continue
            top5 = known.sort_values("p_cnn", ascending=False).head(5)
            non = known[known["target"] == 0]
            a = auroc((non["mri_side"] == soz_side).astype(int), non["p_cnn"]) if non["mri_side"].nunique() == 2 \
                else np.nan
            side_auc.append(a)
            rows.append({"patient": sub.replace("sub-", ""), "SOZ side": soz_side,
                         "sites on SOZ side": f"{(known['mri_side'] == soz_side).mean():.0%}",
                         "CNN top 5 on SOZ side": f"{(top5['mri_side'] == soz_side).mean():.0%}",
                         "non-SOZ: SOZ side scored higher (AUROC)": "-" if not np.isfinite(a) else f"{a:.2f}"})
            for rn, gg in g[g["target"] == 1].groupby("region_name"):
                tw = twin(rn)
                other = g[g["region_name"] == tw] if tw else g.iloc[0:0]
                if len(other):
                    homolog.append({"patient": sub.replace("sub-", ""), "SOZ region": rn,
                                    "CNN: SOZ sites": round(float(gg["p_cnn"].mean()), 2),
                                    "CNN: same region, other side": round(float(other["p_cnn"].mean()), 2)})
        say("\n3b. THE SAME QUESTION WITH REAL ELECTRODES (patients with electrodes on both sides)")
        if not rows:
            say("   cannot be tested: no patient has electrodes on both sides")
            summary.append("Knows the epileptic side (both-side electrodes): cannot be tested")
        else:
            say(pd.DataFrame(rows).to_string(index=False))
            m, lo, hi = boot_ci(side_auc)
            say(f"   non-SOZ sites: is the SOZ side scored higher than the other side? AUROC {fmt_ci(m, lo, hi)} "
                f"(0.5 = cannot tell; below 0.5 = prefers the OTHER side)")
            if homolog:
                h = pd.DataFrame(homolog)
                say("\n   The epileptic region against the SAME region on the other side:")
                say(h.to_string(index=False))
                higher = int((h["CNN: SOZ sites"] > h["CNN: same region, other side"]).sum())
                say(f"   the epileptic region scored higher in {higher} of {len(h)} comparisons")
            summary.append("Knows the epileptic side (both-side electrodes): "
                           + verdict(lo, hi, 0.5, "YES", "NO - prefers the OTHER side", "NOT CLEAR"))

    # ------------------------------------------------------------------ 4. lateralization
    lat = read("lateralization_metrics.csv")
    if lat is not None:
        say("\n4. WHICH SIDE FROM MRI FEATURES (07_lateralization.py, every patient with a clinical side; side_model = "
            "version 9)")
        say(lat.to_string(index=False))

    # ------------------------------------------------------------------ 5. deep structures (version 9)
    deep = read("cnn_predictions_deep.csv")
    if deep is not None and len(deep) and sites is not None:
        soz_p = sites.loc[sites["target"] == 1, "p_cnn"].mean()
        non_p = sites.loc[sites["target"] == 0, "p_cnn"].mean()
        med = sites.groupby("subject")["p_cnn"].median()
        below = float((deep["p_cnn"] < deep["subject"].map(med)).mean())
        say("\n5. DEEP STRUCTURES (version 9 fix): 'not SOZ' cubes from thalamus, caudate, putamen, pallidum, accumbens, "
            "each predicted by the model that never saw the patient")
        say(f"   mean probability: deep cubes {deep['p_cnn'].mean():.2f}, SOZ sites {soz_p:.2f}, other sites {non_p:.2f}"
            "   (v8.1: deep 0.74, SOZ 0.57, other 0.43)")
        say(f"   deep cubes scored below the patient's median site: {below:.0%} of {len(deep)}")
        summary.append("Deep structures scored low (v9 fix works):   " + ("YES" if deep["p_cnn"].mean() < non_p else "NO"))

    # ------------------------------------------------------------------ 6. shortcut check (version 9)
    npz_path = os.path.join(config.OUT_DIR, "patches.npz")
    if sites is not None and os.path.exists(npz_path):
        z = np.load(npz_path)
        is_site = z["kind"].astype(str) == "site" if "kind" in z.files else np.ones(len(z["y"]), bool)
        c = z["X"].shape[-1] // 4                                            # the central half of each cube
        centre = z["X"][is_site, 0, c:-c, c:-c, c:-c].astype(np.float32)
        key = pd.DataFrame({"subject": z["subject"][is_site].astype(str), "contact": z["contact"][is_site].astype(str),
                            "fluid": (centre < -1.5).mean(axis=(1, 2, 3))})
        m = sites.merge(key, on=["subject", "contact"], how="inner")
        rho = [g["fluid"].corr(g["p_cnn"], method="spearman") for _, g in m.groupby("subject")
               if len(g) >= 10 and g["fluid"].nunique() > 1]
        rho = float(np.nanmean(rho)) if rho else np.nan
        rho_y = float(m["fluid"].corr(m["target"], method="spearman")) if len(m) else np.nan
        say("\n6. SHORTCUT CHECK (version 9): does the score simply follow the amount of fluid (dark voxels) in the cube?")
        say(f"   correlation of the score with fluid, within patients: {rho:+.2f}   (fluid with being SOZ: {rho_y:+.2f})")
        say("   above +0.5 = the model mostly measures fluid (the v8.1 shortcut), not the disease")
        summary.append("Score just follows fluid in the cube:        "
                       + ("cannot be tested" if not np.isfinite(rho) else "YES - a shortcut, check the maps" if rho > 0.5
                          else "NO"))

    if summary:
        say("\n" + "-" * 104)
        say("SUMMARY")
        for s in summary:
            say("   " + s)
    say("\nHow to read this: an interval that includes the reference (0.5, or 0 for a difference) means 'we cannot tell "
        "yet'.\nWith about 10 patients the intervals are wide. Do not change settings after looking at these numbers (or "
        "say so\nwhen you report them): choosing settings on the test patients makes the result look better than it is.")
    with open(os.path.join(config.OUT_DIR, "model_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    zpath, n = pack(lines)
    print(f"\nSaved {os.path.join(config.OUT_DIR, 'model_report.txt')}")
    print(f"Packed {n} files into {zpath}  <- please send this zip to Claude")


if __name__ == "__main__":
    with runlog.Log("model_report"):
        main()
