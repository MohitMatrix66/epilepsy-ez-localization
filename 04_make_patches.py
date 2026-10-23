"""
04_make_patches.py  -  PHASE 6a: cut a small 3D cube of MRI around every electrode contact - and, since version 8,
the MIRROR cube from the same place on the other side of the brain.

Why cubes and not whole brains? With ~10-30 patients a network that looks at the whole brain has only ~10-30
examples to learn from and will memorise them. A cube around each contact gives hundreds of labelled examples (every
contact is one example), each with a clear answer from iEEG: "was this contact in the seizure onset zone?"

Why the mirror cube (version 8)? Version 6.1 learned ANATOMY: "the hippocampus is often SOZ" - and scored the healthy
hippocampus on the other side as high as the epileptic one. A doctor compares the two sides: a sclerotic hippocampus
looks different from its healthy twin. So every example now has two images of the same anatomy:
    channel 1 = the cube around the contact,
    channel 2 = the cube from the same place on the other side, mirrored through the plane between the hemispheres
                (found from the centres of left / right structure pairs; works when the head is tilted).
Mirror controls: for every SOZ site, the same place on the healthy side is added as a "not SOZ" example (its cube is
the healthy side, its mirror is the epileptic side). Healthy and epileptic hippocampus are the same anatomy, so the
only way to tell them apart is the disease. A control is used for TRAINING only when the clinical record says the
seizures start on one side only ('Left only' / 'Right only'), the SOZ is on that side, the site is >= 5 mm from the
midline, and no SOZ or uncertain contact lies within 10 mm of the mirror place. Every SOZ site still gets a control
for the TEST: "does the model score the SOZ site above its mirror?" - the left-right test (tools/model_report.py).

Contacts more than 5 mm outside the brain (the outer end of depth electrodes, in the skull) are left out: they would
be trivially "not SOZ". "Uncertain" sites (they strongly start a seizure but are not SOZ) are left out too.
Images: T1 (norm.mgz) and, when EVERY patient has it, T2 (T2.mgz) - each with its mirror. Intensities are
standardised per patient (mean 0, std 1 inside the brain). Cubes are sampled with trilinear interpolation.

Version 8.1 - cubes in MILLIMETRES: the dataset was scanned with two 3T T1 protocols (0.8 mm and 1 mm, see the dataset
paper), so FreeSurfer's volumes are 0.8 mm for some patients and 1 mm for others. A cube of 32 VOXELS was therefore
32 mm wide in one patient and 25.6 mm in another - the same anatomy at two sizes. Now every cube is PATCH_SIZE samples
PATCH_MM apart (32 x 1 mm = 32 mm) in every patient, and the 5 mm margin around the brain is 5 mm in every patient.

Version 9 - fixing the shortcut v8.1 learned (it pointed to the caudate in 19 of 20 new patients):
  * intensities are no longer set to 0 outside FreeSurfer's labels (common.standardise), so fluid is dark everywhere
    and not only in the ventricles;
  * 'deep' cubes: DEEP_PER_STRUCTURE cubes from each deep grey structure of every patient (thalamus, caudate, putamen,
    pallidum, accumbens, both sides; at least DEEP_CLEAR_MM from any SOZ or uncertain contact) are added as 'not SOZ'
    training examples - electrodes almost never go there, so the model had no examples to learn from.

Output: outputs/patches.npz   (X, y, subject, contact, region_id, kind = site / control / deep, train_ok, pair,
                               channels, patch_mm, ijk = cube centre in voxels, planes = mid-sagittal plane and voxel
                               size of every patient)
"""
import json
import os

import numpy as np
import pandas as pd

import common
import config
import runlog

VERSION = "9"
DEEP_PER_STRUCTURE = 2   # version 9: 'not SOZ' cubes per deep structure and side
DEEP_CLEAR_MM = 10.0     # ... at least this far from any SOZ / uncertain contact
MIDLINE_MIN_MM = 5.0      # no mirror control for sites closer than this to the plane between the hemispheres
CONTROL_CLEAR_MM = 10.0   # ... or when a SOZ / uncertain contact lies this close to the mirror place


def cubes(vols, centres, plane, n_channels, spacing):
    """Cube + mirror cube of every image (T1, T2) for every centre: (N, n_channels, size, size, size) float16.
    spacing = distance between the cube's samples in voxels (config.PATCH_MM / voxel size)."""
    if not len(centres):
        return np.zeros((0, n_channels) + (config.PATCH_SIZE,) * 3, np.float16)
    return np.concatenate([common.pair_patches(v, centres, plane, config.PATCH_SIZE, spacing) for v in vols], axis=1)


def main():
    allc = pd.read_csv(os.path.join(config.OUT_DIR, "contacts_labeled.csv"))
    allc["uncertain"] = allc["uncertain"].fillna(0).astype(int) if "uncertain" in allc.columns else 0
    keep = allc["in_bounds"].astype(bool) & (allc["near_brain"].astype(bool) if "near_brain" in allc.columns else True)
    keep &= allc["uncertain"] == 0          # neither SOZ nor 'not SOZ'
    sites = allc[keep].reset_index(drop=True)
    subjects = sorted(sites["subject"].unique())
    no_t2 = [s for s in subjects if not common.has_mri(s, "T2.mgz")] if config.USE_T2 else subjects
    use_t2 = config.USE_T2 and not no_t2
    images = ["T1"] + (["T2"] if use_t2 else [])
    channels = [c for img in images for c in (img, img + "-mirror")]
    clinical = common.clinical_record()
    print(f"{len(sites)} contact sites from {len(subjects)} patients, channels: {channels}, cubes "
          f"{config.PATCH_SIZE}^3 samples {config.PATCH_MM:g} mm apart ({config.PATCH_SIZE * config.PATCH_MM:g} mm wide)"
          + (f"  (no T2 image for {', '.join(no_t2)})" if config.USE_T2 and no_t2 else ""))
    if not clinical:
        print("NOTE: the clinical table (metadata_3T_7T.xlsx) was not found - mirror controls are used for the test only.")

    blocks, info, planes = [], [], {}
    print(f"\n{'patient':<11}{'voxel':>7}{'sites':>6}{'SOZ':>5}{'controls':>9}{'(training)':>11}{'deep':>6}"
          f"{'midline tilt':>14}{'pairs':>7}  clinical side")
    for sub_no, sub in enumerate(subjects):
        lab_img = common.load_mgz(sub, "aparc+aseg.mgz")
        labels = np.asarray(lab_img.dataobj).astype(np.int32)
        brain = labels > 0
        voxel_mm = common.voxel_size(lab_img)                 # 0.8 mm or 1 mm in this dataset
        spacing = config.PATCH_MM / voxel_mm                  # cube samples, in voxels: 1.25 on a 0.8 mm MRI
        near = common.near_brain_mask(labels, voxel_mm, 5.0)
        plane = common.midsagittal_plane(labels, voxel_mm=voxel_mm)
        planes[sub] = {"normal": [round(float(v), 6) for v in plane["normal"]], "offset": round(plane["offset"], 3),
                       "pairs": plane["pairs"], "tilt_deg": plane["tilt_deg"], "spread_mm": plane["spread_mm"],
                       "voxel_mm": round(voxel_mm, 4)}
        vols = [common.standardise(np.asarray(common.load_mgz(sub, "norm.mgz").dataobj), brain)]
        if use_t2:
            vols.append(common.standardise(np.asarray(common.load_mgz(sub, "T2.mgz").dataobj), brain))

        g = sites[sites["subject"] == sub]
        centres = g[["i", "j", "k"]].to_numpy(dtype=float)
        # mirror controls of the SOZ sites
        side_txt = clinical.get(sub, ("", "", ""))[1]
        one_sided = side_txt.lower().startswith(("left only", "right only"))
        clin_side = clinical.get(sub, ("", "", ""))[0]
        hot = allc[(allc["subject"] == sub) & ((allc["target"] == 1) | (allc["uncertain"] == 1))
                   & allc["in_bounds"].astype(bool)][["i", "j", "k"]].to_numpy(dtype=float)
        ctrl_rows = []
        for row_pos, (idx, r) in enumerate(g.iterrows()):
            if int(r["target"]) != 1:
                continue
            c = centres[row_pos]
            dist_mid = float(common.side_of_plane(c, plane)[0])
            if abs(dist_mid * voxel_mm) < MIDLINE_MIN_MM:
                continue
            m = common.reflect(c, plane)
            mi = np.round(m).astype(int)
            if np.any(mi < 0) or np.any(mi >= labels.shape) or not near[tuple(mi)]:
                continue
            clear = (np.linalg.norm(hot - m, axis=1).min() * voxel_mm >= CONTROL_CLEAR_MM) if len(hot) else True
            soz_side = "L" if dist_mid > 0 else "R"
            ok = bool(one_sided and clear and soz_side == clin_side)
            rid, _ = common.assign_region(labels, m, config.REGION_RADIUS_MM / voxel_mm)
            ctrl_rows.append({"centre": m, "contact": r["contact"], "region_id": int(rid), "train_ok": ok,
                              "site_index": idx})
        ctrl_centres = np.array([c["centre"] for c in ctrl_rows]).reshape(-1, 3)
        # version 9: 'not SOZ' cubes from the deep grey structures, away from every SOZ / uncertain contact
        rng = np.random.default_rng(config.SEED + sub_no)
        deep_centres, deep_rows = [], []
        for rid in common.DEEP_IDS:
            vox = np.argwhere(labels == rid).astype(float)
            if len(vox) and len(hot):
                d2 = np.min([np.sum((vox - h) ** 2, axis=1) for h in hot], axis=0)
                vox = vox[np.sqrt(d2) * voxel_mm >= DEEP_CLEAR_MM]
            for k, j in enumerate(rng.choice(len(vox), size=min(DEEP_PER_STRUCTURE, len(vox)), replace=False)
                                  if len(vox) else []):
                deep_centres.append(vox[j])
                deep_rows.append({"contact": f"deep-{common.REGIONS[int(rid)]}-{k + 1}", "region_id": int(rid)})
        deep_centres = np.array(deep_centres).reshape(-1, 3)
        blocks.append(cubes(vols, centres, plane, len(channels), spacing))
        info.append(pd.DataFrame({"subject": sub, "contact": g["contact"].astype(str).to_numpy(),
                                  "region_id": g["region_id"].to_numpy(dtype=int),
                                  "target": g["target"].to_numpy(dtype=int), "kind": "site", "train_ok": True,
                                  "site_index": g.index.to_numpy(), "i": centres[:, 0], "j": centres[:, 1],
                                  "k": centres[:, 2]}))
        if ctrl_rows:
            blocks.append(cubes(vols, ctrl_centres, plane, len(channels), spacing))
            info.append(pd.DataFrame({"subject": sub, "contact": [c["contact"] for c in ctrl_rows],
                                      "region_id": [c["region_id"] for c in ctrl_rows], "target": 0, "kind": "control",
                                      "train_ok": [c["train_ok"] for c in ctrl_rows],
                                      "site_index": [c["site_index"] for c in ctrl_rows], "i": ctrl_centres[:, 0],
                                      "j": ctrl_centres[:, 1], "k": ctrl_centres[:, 2]}))
        if deep_rows:
            blocks.append(cubes(vols, deep_centres, plane, len(channels), spacing))
            info.append(pd.DataFrame({"subject": sub, "contact": [r["contact"] for r in deep_rows],
                                      "region_id": [r["region_id"] for r in deep_rows], "target": 0, "kind": "deep",
                                      "train_ok": True, "site_index": -1, "i": deep_centres[:, 0],
                                      "j": deep_centres[:, 1], "k": deep_centres[:, 2]}))
        n_ok = sum(c["train_ok"] for c in ctrl_rows)
        print(f"{sub:<11}{voxel_mm:>5.2f}mm{len(g):>6}{int(g['target'].sum()):>5}{len(ctrl_rows):>9}{n_ok:>11}"
              f"{len(deep_rows):>6}{plane['tilt_deg']:>12.1f} deg{plane['pairs']:>6}  {side_txt or '?'}")
        del vols, labels, brain, near

    X = np.concatenate(blocks, axis=0)
    meta = pd.concat(info, ignore_index=True)
    # 'pair': for a control, the row of its SOZ site in this file; -1 for sites
    site_row = {(s, i): k for k, (s, i, kind) in enumerate(zip(meta["subject"], meta["site_index"], meta["kind"]))
                if kind == "site"}
    pair = np.array([site_row.get((s, i), -1) if kind == "control" else -1
                     for s, i, kind in zip(meta["subject"], meta["site_index"], meta["kind"])], dtype=np.int64)
    path = os.path.join(config.OUT_DIR, "patches.npz")
    np.savez_compressed(path, X=X, y=meta["target"].to_numpy(dtype=np.int8),
                        subject=np.array(meta["subject"].astype(str).tolist(), dtype=str),
                        contact=np.array(meta["contact"].astype(str).tolist(), dtype=str),
                        region_id=meta["region_id"].to_numpy(dtype=int),
                        kind=np.array(meta["kind"].tolist(), dtype=str),
                        train_ok=meta["train_ok"].to_numpy(dtype=bool), pair=pair,
                        channels=np.array(channels, dtype=str), version=np.array(VERSION),
                        patch_mm=np.array(float(config.PATCH_MM)), ijk=meta[["i", "j", "k"]].to_numpy(dtype=float),
                        planes=np.array(json.dumps(planes)))
    n_site = int((meta["kind"] == "site").sum())
    n_ctrl = int((meta["kind"] == "control").sum())
    n_deep = int((meta["kind"] == "deep").sum())
    n_ok = int((meta["train_ok"] & (meta["kind"] == "control")).sum())
    sizes = pd.Series({s: p["voxel_mm"] for s, p in planes.items()}).round(2).value_counts().sort_index()
    print(f"\nSaved {path}  shape={X.shape}: {n_site} sites ({int(meta['target'].sum())} SOZ) + {n_ctrl} mirror controls "
          f"({n_ok} used for training, the rest only for the left-right test) + {n_deep} deep-structure cubes "
          f"('not SOZ', version 9)")
    print("MRI voxel sizes: " + ", ".join(f"{v:g} mm: {k} patient(s)" for v, k in sizes.items())
          + f" - every cube is {config.PATCH_SIZE * config.PATCH_MM:g} mm wide in all of them.")
    if n_ok == 0:
        print("WARNING: no mirror control can be used for training - the model cannot learn the side from them.")


if __name__ == "__main__":
    with runlog.Log("04_make_patches"):
        main()
