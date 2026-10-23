"""
01_label_contacts.py  -  PHASE 3: put every electrode contact into the MRI and give it a label.

Input : CONTACTS_CSV (see config.py), one row per contact:
            subject, contact, x, y, z, soz[, resected]
        e.g. sub-PID01, LA3, -24.1, -8.7, -21.3, 1, 1
Output: outputs/contacts_labeled.csv  (adds voxel position, brain region and the training target)

It also runs two safety checks that catch the most common bug in this whole project:
coordinates in the wrong space, or left and right flipped.
"""
import os
import re

import numpy as np
import pandas as pd

import common
import config
import runlog


def side_from_name(name):
    """Clinical contact names usually start with L or R (e.g. 'LA3', 'RH10').
    Returns 'L', 'R' or None when the name does not follow that pattern."""
    m = re.match(r"^([LR])[A-Z]", str(name).strip().upper())
    return m.group(1) if m else None


def main():
    df = pd.read_csv(config.CONTACTS_CSV)
    need = {"subject", "contact", "x", "y", "z", "soz"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{config.CONTACTS_CSV} is missing columns: {sorted(missing)}")
    if config.LABEL_MODE == "soz_and_resected" and "resected" not in df.columns:
        raise SystemExit("LABEL_MODE='soz_and_resected' needs a 'resected' column in the contacts table")

    rows = []
    print(f"{'subject':<12}{'contacts':>9}{'near brain':>12}{'near GM':>9}{'SOZ':>6}{'L/R matches name':>18}")
    for sub, g in df.groupby("subject"):
        lab_img = common.load_mgz(sub, "aparc+aseg.mgz")
        labels = np.asarray(lab_img.dataobj).astype(np.int32)
        voxel_mm = common.voxel_size(lab_img)                       # 1 mm or 0.8 mm in this dataset
        near_brain = common.near_brain_mask(labels, voxel_mm, 5.0)   # brain + 5 mm (grids sit on the surface)
        ijk = common.ras_to_vox(g[["x", "y", "z"]].values, lab_img)

        assigned, inside, side_ok = 0, 0, []
        for (_, r), v in zip(g.iterrows(), ijk):
            rid, dist = common.assign_region(labels, v, config.REGION_RADIUS_MM / voxel_mm)
            in_bounds = bool(np.all(v >= 0) and np.all(v < np.array(labels.shape) - 0.5))
            is_near = bool(in_bounds and near_brain[tuple(np.round(v).astype(int))])
            inside += is_near
            if rid:
                assigned += 1
                side = side_from_name(r["contact"])
                if side:
                    side_ok.append(common.hemisphere(rid) == side)
            soz = int(r["soz"])
            target = soz if config.LABEL_MODE == "soz" else int(soz and int(r["resected"]))
            rows.append({**r.to_dict(), "i": v[0], "j": v[1], "k": v[2], "in_bounds": in_bounds,
                         "near_brain": is_near, "region_id": rid, "region_name": common.REGIONS.get(rid, "none"),
                         "dist_to_region_vox": dist, "target": target})

        in_brain = inside / len(g)
        near_gm = assigned / len(g)
        side_txt = f"{np.mean(side_ok):.0%}" if side_ok else "n/a"
        bad = in_brain < 0.5 or (side_ok and np.mean(side_ok) < 0.9)
        print(f"{sub:<12}{len(g):>9}{in_brain:>11.0%}{near_gm:>9.0%}{int(g['soz'].sum()):>6}{side_txt:>17}"
              + ("   <-- CHECK" if bad else ""))

    out = pd.DataFrame(rows)
    out["uncertain"] = out["uncertain"].fillna(0).astype(int) if "uncertain" in out.columns else 0
    out.loc[out["target"] == 1, "uncertain"] = 0
    path = os.path.join(config.OUT_DIR, "contacts_labeled.csv")
    out.to_csv(path, index=False)

    print("\nHow to read this:")
    print(" * 'near brain' = in the brain or within 5 mm of it. The outer contacts of depth (stereo-EEG) electrodes")
    print("   sit in the skull, so 70-100% is normal; those contacts are not used for training. If it is very low")
    print("   (< 50%), the coordinates are in the wrong space: try COORD_SPACE = 'tkr' in config.py.")
    print(" * 'near GM' = contacts on or within a few mm of a grey-matter region. Depth electrodes have")
    print("   many contacts in white matter, so 40-80% is normal. Only these contacts get a region.")
    print(" * 'L/R matches name': contacts named L... should be in the left hemisphere, R... in the right.")
    print("   If this is near 0%, left and right are flipped somewhere. 'n/a' = names have no L/R prefix.")
    print(" * 'uncertain' sites strongly start a seizure but are not SOZ (e.g. a one-off start): they are used neither")
    print("   as SOZ nor as 'not SOZ' (left out of the patches, and their region gets no label unless it has a SOZ site).")
    print(f"\nPositive targets: {int(out['target'].sum())} of {len(out)} contacts "
          f"({out['target'].mean():.1%}); {int((~out['near_brain']).sum())} contacts outside the brain and "
          f"{int(out['uncertain'].sum())} uncertain contacts will not be used for training.  Saved {path}")


if __name__ == "__main__":
    with runlog.Log("01_label_contacts"):
        main()
