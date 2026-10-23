"""
02_region_features.py  -  PHASE 4: turn each patient's MRI into a table of numbers per brain region.

For every patient and every atlas region (about 82 regions) we measure:
    size, T1 brightness, T1 texture, T2 brightness and texture (only if EVERY patient has a T2 image),
    cortical thickness (only if every patient has FreeSurfer's stats tables - this dataset has none)
and how different each value is from the same region on the other side of the brain
(asymmetry - epileptic tissue is often one-sided, so asymmetry is a strong clue).

Each region that has at least one electrode contact gets a label (target = 1 if any of its
contacts is a positive contact). Regions with no electrodes are kept but have no label:
"not recorded" is NOT the same as "healthy". A region whose contacts include an "uncertain" site (it strongly
starts a seizure but is not SOZ) and no SOZ site gets no label either.

Output: outputs/region_table.csv
"""
import os

import numpy as np
import pandas as pd

import common
import config
import runlog

EPS = 1e-6


def region_stats(labels, image, n):
    """Mean and standard deviation of `image` inside every label value (fast, uses bincount)."""
    flat = labels.ravel()
    vals = image.ravel().astype(np.float64)
    cnt = np.bincount(flat, minlength=n).astype(np.float64)
    s1 = np.bincount(flat, weights=vals, minlength=n)
    s2 = np.bincount(flat, weights=vals ** 2, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0))
    return cnt, mean, std


def main():
    contacts = pd.read_csv(os.path.join(config.OUT_DIR, "contacts_labeled.csv"))
    subjects = sorted(contacts["subject"].unique())

    no_t2 = [s for s in subjects if not common.has_mri(s, "T2.mgz")] if config.USE_T2 else subjects
    use_t2 = config.USE_T2 and not no_t2
    thickness = {s: common.read_aparc_thickness(s) for s in subjects}
    use_thick = all(thickness.values())
    print(f"Subjects: {len(subjects)}   T2 features: {'yes' if use_t2 else 'no'}"
          + (f" (no T2 image for {', '.join(no_t2)})" if config.USE_T2 and no_t2 else "")
          + f"   thickness features: {'yes' if use_thick else 'no (FreeSurfer stats tables not in the dataset)'}")

    all_rows = []
    for sub in subjects:
        lab_img = common.load_mgz(sub, "aparc+aseg.mgz")
        labels = np.asarray(lab_img.dataobj).astype(np.int32)
        t1 = np.asarray(common.load_mgz(sub, "norm.mgz").dataobj).astype(np.float32)
        n = int(labels.max()) + 1
        brain = labels > 0
        vox_vol = float(np.prod(lab_img.header.get_zooms()[:3]))
        brain_vol = brain.sum() * vox_vol

        cnt, t1_mean, t1_std = region_stats(labels, t1, n)
        t1_ref = t1[brain].mean()
        if use_t2:
            t2 = np.asarray(common.load_mgz(sub, "T2.mgz").dataobj).astype(np.float32)
            _, t2_mean, t2_std = region_stats(labels, t2, n)
            t2_ref = t2[brain].mean()
        thick = thickness[sub] if use_thick else {}

        feats = {}
        for rid in common.REGION_IDS:
            if rid >= n or cnt[rid] == 0:
                continue
            f = {
                "f_volume_rel": cnt[rid] * vox_vol / brain_vol * 100,      # % of brain volume
                "f_t1_rel": t1_mean[rid] / t1_ref,                         # brightness vs whole brain
                "f_t1_cv": t1_std[rid] / (t1_mean[rid] + EPS),             # texture / blurring
            }
            if use_t2:
                f["f_t2_rel"] = t2_mean[rid] / t2_ref
                f["f_t2_cv"] = t2_std[rid] / (t2_mean[rid] + EPS)
            if thick:
                f["f_thickness"] = thick.get(int(rid), np.nan)
            feats[int(rid)] = f

        for rid, f in feats.items():
            row = {"subject": sub, "region_id": rid, "region_name": common.REGIONS[rid],
                   "hemisphere": common.hemisphere(rid), **f}
            other = feats.get(common.homolog(rid))
            for key, val in f.items():
                ov = other.get(key, np.nan) if other else np.nan
                row[key + "_asym"] = (val - ov) / ((abs(val) + abs(ov)) / 2 + EPS)
            all_rows.append(row)
        print(f"  {sub}: {len(feats)} regions")

    table = pd.DataFrame(all_rows)

    # labels from the electrodes
    c = contacts[contacts["region_id"] > 0].copy()
    c["uncertain"] = c["uncertain"].fillna(0).astype(int) if "uncertain" in c.columns else 0
    lab = (c.groupby(["subject", "region_id"])
             .agg(n_contacts=("contact", "count"), target=("target", "max"), n_uncertain=("uncertain", "sum"))
             .reset_index())
    unknown = (lab["target"] == 0) & (lab["n_uncertain"] > 0)   # no SOZ site, but a site that started a seizure
    lab.loc[unknown, "target"] = np.nan
    table = table.merge(lab, on=["subject", "region_id"], how="left")
    table["n_contacts"] = table["n_contacts"].fillna(0).astype(int)
    table["n_uncertain"] = table["n_uncertain"].fillna(0).astype(int)
    table["sampled"] = (table["n_contacts"] > 0) & table["target"].notna()

    path = os.path.join(config.OUT_DIR, "region_table.csv")
    table.to_csv(path, index=False)
    s = table[table["sampled"]]
    print(f"\nRegions total: {len(table)}   with electrodes (usable for training): {len(s)}   "
          f"positive: {int(s['target'].sum())} ({s['target'].mean():.1%})"
          + (f"   no label because of an uncertain site: {int(unknown.sum())}" if unknown.any() else ""))
    print(f"Feature columns: {[col for col in table.columns if col.startswith('f_')]}")
    print(f"Saved {path}")


if __name__ == "__main__":
    with runlog.Log("02_region_features"):
        main()
