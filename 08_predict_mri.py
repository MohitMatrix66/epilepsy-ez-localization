"""
08_predict_mri.py  -  PHASE 8: the FINAL MODEL on an MRI: where does it think the seizures start?

For every point of the brain's grey matter (every --step mm) the trained 3D CNN looks at the cube around the point and
the mirror cube from the other side (like a doctor comparing the two hippocampi) and gives a probability. The result:
  outputs/predictions/<name>/<name>_ez_probability.nii.gz   probability map in the patient's MRI space
                                                            (open it on top of the T1 in ITK-SNAP / Freeview / Slicer)
  outputs/predictions/<name>/<name>_ez_zone.nii.gz          the highest-scoring --zone-percent of the grey matter
  outputs/predictions/<name>/<name>_ez_probability.png      three views + which side and which regions
  outputs/predictions/<name>/<name>_regions.csv             mean probability of every brain region
  outputs/predictions/<name>/<name>_summary.txt

Which model: a patient who was used for training is shown with the models of HIS cross-validation fold (they never
saw him), so the picture is an honest test; any other patient gets the final models (trained on all patients,
05_train_cnn.py --final). The summary says which.

Run (laptop, after copying the Colab results into the project, see README section 5):
  python 08_predict_mri.py --subject sub-PID01            a patient of the dataset (its FreeSurfer files)
  python 08_predict_mri.py --fs-dir D:\\newpatient --name P31
                                                          a NEW patient: a FreeSurfer subject folder that contains
                                                          mri\\norm.mgz and mri\\aparc+aseg.mgz
  python 08_predict_mri.py --unseen                       every dataset patient the model was NOT trained on: the
                                                          predicted side against the clinical record - the honest
                                                          accuracy to show next to any picture
About 2-5 minutes per patient on a laptop (--step 6 is faster, --step 4 finer); on a GPU a few seconds.
Any voxel size works (version 8.1): the cubes are cut in millimetres exactly as for training (cnn_model.json says how
many mm), and --step is in mm (rounded to whole voxels: 5 mm = 5 voxels at 1 mm, 6 voxels = 4.8 mm at 0.8 mm).
Version 9: the map covers the regions the model was trained on - cortex, hippocampus, amygdala; the deep grey
structures (thalamus, caudate, putamen, pallidum, accumbens) are outlined in grey as NOT ASSESSED (v8.1 wrongly pointed
to the caudate). Every prediction is averaged with its left-right flipped copy. The side is also given by the side
model of 07_lateralization.py (hippocampus / amygdala / thalamus volume, hippocampus T2, temporal thickness): for a
dataset patient its leave-one-out answer (a model that never saw him), for a new MRI the final side model.
--unseen also reports the fair side score (AUROC, not affected by a lean to one side) and the sclerosis check: in
patients whose MRI report names hippocampal sclerosis, is the sick hippocampus scored above the healthy one?
Scope: drug-resistant FOCAL epilepsy. Not for generalized epilepsy; deep lesions (hypothalamus, heterotopia) are not
assessed; weakest when the MRI looks normal.
This is a research prototype trained on 10 patients - not a clinical tool.
"""
import argparse
import importlib.util
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

import common  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 256                       # grid points per batch of cubes (memory: ~35 MB per channel pair)
LEFT_RIGHT = {"L": "left", "R": "right"}


# =========================================================================== the model
_CNN_MODULE = None


def load_cnn_module():
    global _CNN_MODULE
    if _CNN_MODULE is None:
        spec = importlib.util.spec_from_file_location("train_cnn", os.path.join(HERE, "05_train_cnn.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CNN_MODULE = mod
    return _CNN_MODULE


def flip_average(net, xb):
    """The same prediction as in training (05_train_cnn.flip_average): cube + left-right flipped copy, averaged."""
    return load_cnn_module().flip_average(net, xb)


def side_model_answer(subject=None, fs_dir=None, labels=None):
    """(answer 'L' / 'R' / 'not sure' / '', p(left), how) from the version 9 side model of 07_lateralization.py: a
    dataset patient gets its leave-one-out answer (a side model that never saw him); a new MRI gets the final model."""
    pred_path = os.path.join(config.OUT_DIR, "lateralization_predictions.csv")
    if subject and os.path.exists(pred_path):
        t = pd.read_csv(pred_path)
        row = t[t["subject"] == subject]
        if len(row) and "p_left_side_model" in t.columns and np.isfinite(row["p_left_side_model"].iloc[0]):
            p = float(row["p_left_side_model"].iloc[0])
            how = ("leave-one-out: a side model that never saw this patient" if str(row["side"].iloc[0]) in ("L", "R")
                   else "final side model (this patient has no clinical side and was not used to train it)")
            return str(row["side_model_call"].iloc[0]), p, how
    card_path = os.path.join(config.OUT_DIR, "side_model.json")
    if not os.path.exists(card_path):
        return "", np.nan, "no side model (run 07_lateralization.py)"
    with open(card_path) as fh:
        card = json.load(fh)
    if subject:
        labels, t2, surf, label = common.side_inputs(subject)
    else:
        mri = os.path.join(fs_dir, "mri") if os.path.isdir(os.path.join(fs_dir, "mri")) else fs_dir
        t2 = np.asarray(nib.load(os.path.join(mri, "T2.mgz")).dataobj) if os.path.exists(os.path.join(mri, "T2.mgz")) \
            else None
        surf, label = os.path.join(fs_dir, "surf"), os.path.join(fs_dir, "label")
    f = common.side_features(labels, t2, surf, label)
    x = np.array([f[k] if np.isfinite(f[k]) else 0.0 for k in card["features"]])
    z = (x - np.array(card["mean"])) / np.array(card["scale"])
    p = float(1.0 / (1.0 + np.exp(-(z @ np.array(card["coef"]) + card["intercept"]))))
    answer = "not sure" if abs(p - 0.5) < card["not_sure"] else ("L" if p >= 0.5 else "R")
    how = "final side model (trained on the patients with a clinical side)"
    if subject and subject in set(card.get("trained_on", [])):
        how += " - WARNING: this patient was one of them, so this is NOT a test"
    return answer, p, how


def read_card():
    path = os.path.join(config.OUT_DIR, "cnn_model.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - train the CNN first (05_train_cnn.py --lopo --resume --final on Colab) "
                         "and copy its results into the outputs folder (README section 5).")
    with open(path) as fh:
        card = json.load(fh)
    if str(card.get("version")) != "9" or "patch_mm" not in card:
        raise SystemExit("cnn_model.json is from an older version - make the patches again (04_make_patches.py, "
                         "version 9) and train the version 9 model (05_train_cnn.py on Colab).")
    return card


def choose_models(card, subject, mode):
    """(model files, description). Training patients get the models of their own fold (an honest test)."""
    trained = subject in set(card.get("training_subjects", []))
    final = card.get("final") or {}
    if mode == "auto":
        mode = "fold" if trained else ("final" if final.get("models") else "folds")
    if mode == "fold":
        for f in card["folds"]:
            if subject in f["test_subjects"]:
                return f["models"], f"the {len(f['models'])} model(s) of fold {f['fold']}, which never saw this patient"
        raise SystemExit(f"{subject} was not tested in any fold - use --models final or folds")
    if mode == "final":
        if not final.get("models"):
            raise SystemExit("no final models - run 05_train_cnn.py with --final (see README section 5)")
        note = "the final models (trained on all training patients)"
        if trained:
            note += " - WARNING: this patient was one of them, so this picture is NOT a test"
        return final["models"], note
    models = [f["models"][0] for f in card["folds"]]      # one model of every fold
    note = f"{len(models)} fold models (one per fold)"
    if trained:
        note += " - WARNING: all but one of them saw this patient, so this picture is NOT a test"
    return models, note


def build_models(card, files, device):
    mod = load_cnn_module()
    nets = []
    for f in files:
        path = os.path.join(config.OUT_DIR, f)
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing - copy all the files of the Colab results into outputs (README 5).")
        net = mod.SmallCNN3D(len(card["channels"]), base=card.get("base", 16), dropout=card.get("dropout", 0.3))
        net.load_state_dict(torch.load(path, map_location=device))
        nets.append(net.to(device).eval())
    return nets


# =========================================================================== one brain
def load_brain(subject=None, fs_dir=None, need_t2=False):
    """(label image, labels, {'T1': volume, 'T2': volume or missing})."""
    if fs_dir:
        mri = os.path.join(fs_dir, "mri") if os.path.isdir(os.path.join(fs_dir, "mri")) else fs_dir
        lab_path = next((os.path.join(mri, f) for f in ("aparc+aseg.mgz", "aparc_aseg.mgz")
                         if os.path.exists(os.path.join(mri, f))), None)
        if lab_path is None or not os.path.exists(os.path.join(mri, "norm.mgz")):
            raise SystemExit(f"{mri} must contain norm.mgz and aparc+aseg.mgz (FreeSurfer recon-all output)")
        lab_img = nib.load(lab_path)
        vols = {"T1": np.asarray(nib.load(os.path.join(mri, "norm.mgz")).dataobj)}
        if need_t2:
            if not os.path.exists(os.path.join(mri, "T2.mgz")):
                raise SystemExit("the model uses T2 images, but this folder has no mri\\T2.mgz")
            vols["T2"] = np.asarray(nib.load(os.path.join(mri, "T2.mgz")).dataobj)
    else:
        lab_img = common.load_mgz(subject, "aparc+aseg.mgz")
        vols = {"T1": np.asarray(common.load_mgz(subject, "norm.mgz").dataobj)}
        if need_t2:
            vols["T2"] = np.asarray(common.load_mgz(subject, "T2.mgz").dataobj)
    labels = np.asarray(lab_img.dataobj).astype(np.int32)
    return lab_img, labels, vols


@torch.no_grad()
def predict_brain(labels, vols, card, nets, device, step_mm, zone_percent, voxel_mm):
    """Probability at every grid point of the grey matter, the map, region means and the side.
    step_mm: distance between grid points in mm (rounded to whole voxels); voxel_mm: the MRI's voxel size."""
    brain = labels > 0
    images = [c for c in card["channels"] if not c.endswith("-mirror")]
    std = {k: common.standardise(vols[k], brain) for k in images}
    plane = common.midsagittal_plane(labels, voxel_mm=voxel_mm)
    gm = np.isin(labels, common.ASSESSED_IDS)           # version 9: cortex, hippocampus, amygdala only
    deep = np.isin(labels, common.DEEP_IDS)             # shown as 'not assessed'
    step = max(1, int(round(step_mm / voxel_mm)))           # grid step in voxels
    spacing = float(card["patch_mm"]) / voxel_mm            # cube samples in voxels, as in training
    start = [int(step // 2)] * 3
    axes = [np.arange(s, n, step) for s, n in zip(start, labels.shape)]
    gi, gj, gk = np.meshgrid(*axes, indexing="ij")
    on = gm[gi, gj, gk]
    pts = np.stack([gi[on], gj[on], gk[on]], axis=1).astype(np.float64)
    probs = np.zeros(len(pts))
    size = int(card["patch_size"])
    for s in range(0, len(pts), CHUNK):
        c = pts[s:s + CHUNK]
        x = np.concatenate([common.pair_patches(std[k], c, plane, size, spacing) for k in images], axis=1)
        xb = torch.from_numpy(x.astype(np.float32)).to(device)
        probs[s:s + CHUNK] = np.mean([flip_average(net, xb).cpu().numpy() for net in nets], axis=0)
        done = min(s + CHUNK, len(pts))
        if done == len(pts) or (s // CHUNK) % 4 == 3:
            print(f"    {done} of {len(pts)} points", flush=True)

    # full-resolution map: the coarse grid (nearest prediction where the grid point is not grey matter), trilinear
    coarse = np.full(gi.shape, np.nan)
    coarse[on] = probs
    missing = np.isnan(coarse)
    if missing.any():
        idx = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        coarse = coarse[tuple(idx)]
    prob = np.zeros(labels.shape, np.float32)
    where = np.argwhere(gm)
    coords = ((where - np.array(start)) / step).T
    prob[gm] = ndimage.map_coordinates(coarse, coords, order=1, mode="nearest")

    point_labels = labels[pts[:, 0].astype(int), pts[:, 1].astype(int), pts[:, 2].astype(int)]
    df = pd.DataFrame({"region_id": point_labels, "p": probs})
    regions = (df.groupby("region_id")["p"].agg(["mean", "count"]).reset_index()
               .rename(columns={"mean": "mean_probability", "count": "points"}))
    regions["region_name"] = regions["region_id"].map(common.REGIONS)
    regions["side"] = regions["region_name"].map(lambda r: "L" if str(r).startswith(("Left-", "ctx-lh-")) else "R")
    regions["lobe"] = regions["region_id"].map(common.lobe)
    regions = regions.sort_values("mean_probability", ascending=False).reset_index(drop=True)
    point_side = np.array(["L" if str(common.REGIONS.get(int(r), "")).startswith(("Left-", "ctx-lh-")) else "R"
                           for r in point_labels])
    side_val = {s: float(probs[point_side == s].mean()) if (point_side == s).any() else np.nan for s in ("L", "R")}
    thr = float(np.percentile(probs, 100 - zone_percent)) if len(probs) else 1.0
    hip = {sd: float(probs[point_labels == rid].mean()) if (point_labels == rid).any() else np.nan
           for sd, rid in (("L", 17), ("R", 53))}
    return {"plane": plane, "points": pts, "probs": probs, "prob": prob, "gm": gm, "deep": deep, "regions": regions,
            "side_mean": side_val, "zone_threshold": thr, "grid_mm": step * voxel_mm, "hippocampus": hip}


def summarise(res, zone_percent, voxel_mm):
    regions, side = res["regions"], res["side_mean"]
    pred_side = "L" if side["L"] > side["R"] else "R"
    zone = (res["prob"] >= res["zone_threshold"]) & res["gm"]
    big = regions[regions["points"] >= 3]          # a region needs a few grid points for a fair mean
    top = (big if len(big) else regions).head(5)
    top_lobe = top.iloc[0]["lobe"] if len(top) else ""
    return {"pred_side": pred_side, "margin": side["L"] - side["R"], "mean_left": side["L"], "mean_right": side["R"],
            "top_regions": list(top["region_name"]), "top_lobe": top_lobe, "zone": zone,
            "zone_ml": float(zone.sum() * voxel_mm ** 3 / 1000.0), "zone_percent": zone_percent,
            "hip_left": res["hippocampus"]["L"], "hip_right": res["hippocampus"]["R"]}


def picture(path, name, t1, res, summ, note, contacts=None):
    prob, zone = res["prob"], summ["zone"]
    # the three views go through the highest point of the map
    i0, j0, k0 = (int(v) for v in np.unravel_index(np.argmax(np.where(res["gm"], prob, -1)), prob.shape))
    # FreeSurfer volumes are stored as (i: right->left, j: superior->inferior, k: posterior->anterior)
    views = [("Sagittal", t1[i0, :, :], prob[i0, :, :], zone[i0, :, :], "upper", ("k", "j"), "i", i0),
             ("Coronal  (R | L)", t1[:, :, k0].T, prob[:, :, k0].T, zone[:, :, k0].T, "upper", ("i", "j"), "k", k0),
             ("Axial  (R | L)", t1[:, j0, :].T, prob[:, j0, :].T, zone[:, j0, :].T, "lower", ("i", "k"), "j", j0)]
    # colour only the upper half of this brain's grey-matter probabilities (the lower half stays grey), so the most
    # suspicious places stand out whatever the overall level of the model's probabilities
    lo = float(np.median(res["probs"])) if len(res["probs"]) else 0.0
    hi = float(np.percentile(res["probs"], 99.5)) if len(res["probs"]) else 1.0
    hi = hi if hi > lo else lo + 1e-3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="black")
    im = None
    for ax, (title, bg, hm, zn, origin, (cx, cy), slice_axis, slice_val) in zip(axes, views):
        ax.imshow(bg, cmap="gray", origin=origin)
        im = ax.imshow(np.ma.masked_where(hm < lo, hm), cmap="hot", vmin=lo, vmax=hi, alpha=0.65, origin=origin)
        if zn.any():
            ax.contour(zn.astype(float), levels=[0.5], colors="cyan", linewidths=0.8, origin=origin)
        dp = {"i": res["deep"][slice_val, :, :], "k": res["deep"][:, :, slice_val].T,
              "j": res["deep"][:, slice_val, :].T}[slice_axis]
        if dp.any():
            ax.contour(dp.astype(float), levels=[0.5], colors="silver", linewidths=0.6, linestyles="dashed",
                       origin=origin)
        if contacts is not None and len(contacts):
            near = contacts[np.abs(contacts[slice_axis] - slice_val) <= 3]
            for sel, colour in ((near[near["target"] == 0], "white"), (near[near["target"] == 1], "lime")):
                ax.scatter(sel[cx], sel[cy], s=12, c=colour, edgecolors="black", linewidths=0.4)
        ax.set_title(title, color="white")
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("model probability (coloured: above this brain's median)", color="white")
    cb.ax.tick_params(colors="white")
    side = LEFT_RIGHT[summ["pred_side"]]
    text = (f"{name}: model points to the {side.upper()} side (mean probability left {summ['mean_left']:.2f}, right "
            f"{summ['mean_right']:.2f}); top regions: {', '.join(summ['top_regions'][:3])}\n"
            f"cyan line = highest {summ['zone_percent']:g}% of the assessed grey matter (cortex, hippocampus, amygdala); "
            f"grey dashed = deep structures, NOT ASSESSED"
            + ("; dots = electrode contacts (green = SOZ)" if contacts is not None and len(contacts) else "")
            + f".  Models: {note}.  Research prototype - not a clinical tool.")
    fig.suptitle(text, color="white", fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def run_one(name, card, device, step, zone_percent, subject=None, fs_dir=None, mode="auto", maps=True):
    files, note = choose_models(card, subject or "", mode)
    nets = build_models(card, files, device)
    need_t2 = any(c.startswith("T2") for c in card["channels"])
    lab_img, labels, vols = load_brain(subject, fs_dir, need_t2)
    voxel_mm = common.voxel_size(lab_img)
    print(f"  {name}: MRI voxels {voxel_mm:.2f} mm; predicting with {note} ...")
    res = predict_brain(labels, vols, card, nets, device, step, zone_percent, voxel_mm)
    summ = summarise(res, zone_percent, voxel_mm)
    summ["side_model"], summ["side_model_p_left"], side_how = side_model_answer(subject, fs_dir, labels)
    out_dir = os.path.join(config.OUT_DIR, "predictions", name)
    if maps:
        os.makedirs(out_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(res["prob"], lab_img.affine), os.path.join(out_dir, f"{name}_ez_probability.nii.gz"))
        nib.save(nib.Nifti1Image(summ["zone"].astype(np.uint8), lab_img.affine),
                 os.path.join(out_dir, f"{name}_ez_zone.nii.gz"))
        res["regions"].to_csv(os.path.join(out_dir, f"{name}_regions.csv"), index=False)
        contacts = None
        lab_path = os.path.join(config.OUT_DIR, "contacts_labeled.csv")
        if subject and os.path.exists(lab_path):
            c = pd.read_csv(lab_path)
            contacts = c[c["subject"] == subject]
        picture(os.path.join(out_dir, f"{name}_ez_probability.png"), name, np.asarray(vols["T1"], np.float32),
                res, summ, note, contacts)
        plane = res["plane"]
        sm = summ["side_model"]
        lines = [f"{name}", f"models: {note}",
                 f"side (CNN): {LEFT_RIGHT[summ['pred_side']]} (mean probability over the assessed grey matter: left "
                 f"{summ['mean_left']:.3f}, right {summ['mean_right']:.3f})",
                 "side (side model: hippocampus / amygdala / thalamus / T2 / temporal thickness): "
                 + (f"{LEFT_RIGHT.get(sm, sm)} (p(left) = {summ['side_model_p_left']:.2f}; {side_how})" if sm
                    else side_how),
                 f"top regions: {', '.join(summ['top_regions'])}",
                 f"zone = highest {zone_percent:g}% of the assessed grey matter (cortex, hippocampus, amygdala): "
                 f"{summ['zone_ml']:.1f} ml (probability >= {res['zone_threshold']:.2f}); deep structures not assessed",
                 f"hippocampus: left {summ['hip_left']:.3f}, right {summ['hip_right']:.3f}",
                 f"midline: tilt {plane['tilt_deg']} deg, {plane['pairs']} left/right pairs",
                 f"grid: every {res['grid_mm']:.1f} mm, {len(res['probs'])} points; MRI voxel size {voxel_mm:.2f} mm; "
                 f"cubes {card['patch_size']} samples x {float(card['patch_mm']):g} mm",
                 "Scope: drug-resistant focal epilepsy; not for generalized epilepsy; deep lesions not assessed; "
                 "weakest when the MRI looks normal.",
                 "Research prototype trained on a small number of patients - not a clinical tool."]
        with open(os.path.join(out_dir, f"{name}_summary.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("    " + "\n    ".join(lines[1:6]))
        print(f"    saved in {out_dir}")
    return summ, note


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def unseen(card, device, step, zone_percent, maps):
    import aws_data
    trained = set(card.get("training_subjects", []))
    clinical = common.clinical_record()
    mri = common.mri_record()
    subs = [s for s in aws_data.subjects() if s not in trained and common.has_mri(s, "aparc+aseg.mgz")
            and common.has_mri(s, "norm.mgz")]
    if not subs:
        raise SystemExit("no patient with FreeSurfer files outside the training patients")
    print(f"{len(subs)} patients the model was not trained on: {', '.join(subs)}")
    rows = []
    for s in subs:
        summ, _ = run_one(s, card, device, step, zone_percent, subject=s, mode="auto", maps=maps)
        side, side_txt, lobe_txt = clinical.get(s, ("", "", ""))
        lobe = lobe_txt.strip().lower()
        lobe = lobe if lobe in ("temporal", "frontal", "parietal", "occipital", "insula") else ""
        res_txt, lesion, lesion_side = mri.get(s, ("", "", ""))
        mts = common.is_mts(lesion) and lesion_side in ("L", "R")
        hl, hr = summ["hip_left"], summ["hip_right"]
        sm = summ["side_model"]
        rows.append({"subject": s, "clinical_side": side, "clinical_side_text": side_txt, "predicted_side":
                     summ["pred_side"], "side_correct": (summ["pred_side"] == side) if side else np.nan,
                     "margin_left_minus_right": round(summ["margin"], 4), "side_model": sm,
                     "side_model_p_left": round(summ["side_model_p_left"], 3),
                     "side_model_correct": (sm == side) if side and sm in ("L", "R") else np.nan,
                     "clinical_lobe": lobe, "top_region": summ["top_regions"][0] if summ["top_regions"] else "",
                     "top_lobe": summ["top_lobe"], "lobe_correct": (summ["top_lobe"] == lobe) if lobe else np.nan,
                     "mri_result": res_txt, "mri_lesion": lesion, "mri_lesion_side": lesion_side,
                     "hippocampal_sclerosis_side": lesion_side if mts else "",
                     "hippocampus_left": round(hl, 3), "hippocampus_right": round(hr, 3),
                     "sick_hippocampus_higher": ((hl > hr) == (lesion_side == "L")) if mts and np.isfinite(hl + hr)
                     else np.nan})
    df = pd.DataFrame(rows)
    out_dir = os.path.join(config.OUT_DIR, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "unseen_validation.csv"), index=False)
    lines = ["=" * 100, "THE FINAL MODEL ON PATIENTS IT NEVER SAW - against the clinical record (version 9)", "=" * 100]
    show = df[["subject", "clinical_side_text", "predicted_side", "side_correct", "margin_left_minus_right",
               "side_model", "side_model_correct", "clinical_lobe", "top_region", "mri_result"]]
    lines.append(show.to_string(index=False))
    known = df[df["clinical_side"].isin(["L", "R"])].copy()
    known["side_correct"] = known["side_correct"].astype(float)
    k, n = int(known["side_correct"].sum()), len(known)
    lo, hi = wilson(k, n)
    maj = known["clinical_side"].value_counts()
    maj_acc = (maj.iloc[0] / n) if n else np.nan
    bal = np.mean([(known.loc[known["clinical_side"] == s, "side_correct"]).mean() for s in ("L", "R")
                   if (known["clinical_side"] == s).any()]) if n else np.nan
    yl = (known["clinical_side"] == "L").astype(int).to_numpy()
    lines.append("")
    if n:
        lines.append(f"Side, CNN: correct in {k} of {n} patients = {k / n:.0%} (95% interval {lo:.0%}-{hi:.0%}); "
                     f"balanced accuracy {bal:.2f}; FAIR SCORE (AUROC of left-minus-right, not affected by a lean to "
                     f"one side) {common.safe_auroc(yl, known['margin_left_minus_right'].to_numpy(float)):.2f}.")
        lines.append(f"      always guessing the more common side ({maj.index[0]}) would be right in {maj_acc:.0%}; "
                     f"the CNN said 'left' for {int((df['predicted_side'] == 'L').sum())} of {len(df)} patients.")
        sm_ok = known["side_model_correct"].dropna().astype(float)
        p_sm = known["side_model_p_left"].to_numpy(float)
        auc_sm = common.safe_auroc(yl[np.isfinite(p_sm)], p_sm[np.isfinite(p_sm)]) if np.isfinite(p_sm).any() \
            else np.nan
        lines.append(f"Side, side model (leave-one-out): correct in {int(sm_ok.sum())} of {len(sm_ok)} patients where "
                     f"it was sure ({n - len(sm_ok)} 'not sure'); fair score (AUROC) {auc_sm:.2f}.")
    else:
        lines.append("Side: no patient with a clinical side.")
    hs = df[df["sick_hippocampus_higher"].notna()]
    if len(hs):
        lines.append(f"Sclerosis check: in {int(hs['sick_hippocampus_higher'].astype(bool).sum())} of {len(hs)} patients "
                     "whose MRI report names one-sided hippocampal sclerosis, the CNN scored the SICK hippocampus "
                     f"above the healthy one ({', '.join(hs['subject'])}).")
    lob = df[df["clinical_lobe"] != ""].copy()
    if len(lob):
        kl = int(lob["lobe_correct"].astype(float).sum())
        always = (lob["clinical_lobe"] == "temporal").mean()
        lines.append(f"Lobe of the top region = the clinical lobe in {kl} of {len(lob)} patients "
                     f"(always saying 'temporal' would be right in {always:.0%}).")
    deep_top = int(df["top_region"].astype(str).str.contains("Caudate|Putamen|Pallidum|Thalamus|Accumbens").sum())
    lines.append(f"Top region in a deep structure: {deep_top} of {len(df)} patients (v8.1: 19 of 20; version 9 does not "
                 "assess them).")
    lines.append("Show this table next to any picture: it is the honest accuracy of the model on new patients.")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, "unseen_validation.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser(description="The final model on an MRI: a probability map of the seizure onset zone.")
    ap.add_argument("--subject", help="a patient of the dataset, e.g. sub-PID01")
    ap.add_argument("--fs-dir", dest="fs_dir", help="a FreeSurfer subject folder of a NEW patient")
    ap.add_argument("--name", help="name of the outputs (default: the subject / folder name)")
    ap.add_argument("--unseen", action="store_true", help="every dataset patient the model was not trained on")
    ap.add_argument("--models", default="auto", choices=("auto", "final", "fold", "folds"))
    ap.add_argument("--step", type=float, default=None, help="mm between predictions (default 5; 6 with --unseen)")
    ap.add_argument("--zone-percent", type=float, default=2.0, dest="zone_percent")
    ap.add_argument("--no-maps", action="store_true", help="with --unseen: only the table, no maps")
    args = ap.parse_args()
    if not (args.subject or args.fs_dir or args.unseen):
        ap.error("give --subject, --fs-dir or --unseen")
    card = read_card()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step = float(args.step or (6 if args.unseen else 5))
    print(f"Model version {card['version']}, channels {card['channels']}, device {device}, grid every {step:g} mm "
          f"(rounded to whole voxels), cubes {card['patch_size']} x {float(card['patch_mm']):g} mm")
    if args.unseen:
        unseen(card, device, step, args.zone_percent, maps=not args.no_maps)
        return
    if args.fs_dir:
        name = args.name or os.path.basename(os.path.normpath(args.fs_dir))
        run_one(name, card, device, step, args.zone_percent, fs_dir=args.fs_dir)
    else:
        run_one(args.name or args.subject, card, device, step, args.zone_percent, subject=args.subject,
                mode=args.models)


if __name__ == "__main__":
    with runlog.Log("08_predict_mri"):
        main()
