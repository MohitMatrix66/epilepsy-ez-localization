"""
06_make_heatmap.py  -  PHASE 7: show WHERE the model thinks the epileptogenic zone is.

Every atlas region of one patient is coloured by its predicted probability. The prediction
comes from the cross-validation fold in which this patient was NOT used for training, so the
picture is an honest "unseen patient" result.

Run:   python 06_make_heatmap.py --subject sub-PID01
       python 06_make_heatmap.py --subject sub-PID01 --col p_logreg

Outputs (in outputs/heatmaps/):
  <subject>_<col>_ez_probability.nii.gz   open on top of the T1 in Freeview / ITK-SNAP / 3D Slicer
  <subject>_<col>_ez_probability.png      quick three-view picture with the electrode contacts
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--pred", default=os.path.join(config.OUT_DIR, "baseline_all_regions.csv"))
    ap.add_argument("--col", default="p_random_forest")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    pred = pred[(pred["subject"] == args.subject) & pred[args.col].notna()]
    if pred.empty:
        raise SystemExit(f"No predictions for {args.subject} in {args.pred} (column {args.col})")

    lab_img = common.load_mgz(args.subject, "aparc+aseg.mgz")
    labels = np.asarray(lab_img.dataobj).astype(np.int32)
    t1 = np.asarray(common.load_mgz(args.subject, "norm.mgz").dataobj).astype(np.float32)

    heat = np.zeros(labels.shape, dtype=np.float32)
    for rid, p in zip(pred["region_id"], pred[args.col]):
        heat[labels == rid] = p

    out_dir = os.path.join(config.OUT_DIR, "heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    nii_path = os.path.join(out_dir, f"{args.subject}_{args.col}_ez_probability.nii.gz")
    nib.save(nib.Nifti1Image(heat, lab_img.affine), nii_path)

    ranked = pred.sort_values(args.col, ascending=False)
    cols = ["region_name", args.col] + [c for c in ("sampled", "target") if c in ranked.columns]
    print(f"Top 10 regions for {args.subject} ({args.col}):")
    print(ranked[cols].head(10).to_string(index=False, na_rep="-"))
    if "target" in cols:
        print("(target: 1 = SOZ region, 0 = recorded but not SOZ, - = no electrode there, so unknown)")

    # centre the three views on the highest-scoring region
    best = int(ranked["region_id"].iloc[0])
    i0, j0, k0 = np.round(np.argwhere(labels == best).mean(axis=0)).astype(int)

    contacts = pd.read_csv(os.path.join(config.OUT_DIR, "contacts_labeled.csv"))
    contacts = contacts[contacts["subject"] == args.subject]

    # FreeSurfer volumes are stored as (i: right->left, j: superior->inferior, k: posterior->anterior)
    views = [
        ("Sagittal", t1[i0, :, :], heat[i0, :, :], "upper", ("k", "j"), "i", i0),
        ("Coronal  (R | L)", t1[:, :, k0].T, heat[:, :, k0].T, "upper", ("i", "j"), "k", k0),
        ("Axial  (R | L)", t1[:, j0, :].T, heat[:, j0, :].T, "lower", ("i", "k"), "j", j0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="black")
    for ax, (title, bg, hm, origin, (cx, cy), slice_axis, slice_val) in zip(axes, views):
        ax.imshow(bg, cmap="gray", origin=origin)
        im = ax.imshow(np.ma.masked_where(hm <= 0, hm), cmap="hot", vmin=0, vmax=1, alpha=0.65, origin=origin)
        near = contacts[np.abs(contacts[slice_axis] - slice_val) <= 3]
        unc = near["uncertain"].fillna(0).astype(int) == 1 if "uncertain" in near.columns else near["target"] < 0
        for sel, colour in ((near[(near["target"] == 0) & ~unc], "white"), (near[unc], "yellow"),
                            (near[near["target"] == 1], "cyan")):
            ax.scatter(sel[cx], sel[cy], s=14, c=colour, edgecolors="black", linewidths=0.4)
        ax.set_title(title, color="white")
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("predicted probability", color="white")
    cb.ax.tick_params(colors="white")
    fig.suptitle(f"{args.subject} - predicted epileptogenic regions ({args.col}); "
                 f"cyan = SOZ, yellow = uncertain, white = other contacts", color="white")
    png_path = os.path.join(out_dir, f"{args.subject}_{args.col}_ez_probability.png")
    fig.savefig(png_path, dpi=110, bbox_inches="tight", facecolor="black")
    print(f"\nSaved {nii_path}\nSaved {png_path}")


if __name__ == "__main__":
    with runlog.Log("06_make_heatmap"):
        main()
