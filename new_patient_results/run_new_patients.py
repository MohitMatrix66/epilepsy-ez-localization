r"""
new_patient_results/run_new_patients.py  -  the FINAL model (version 9) on the patients it was never trained on, compared
with the clinical record of the dataset (metadata_3T_7T.xlsx: seizure-onset side and lobe, MRI report, 7T report).

Nothing in the project is changed: the trained models and the side model are only READ from outputs\, the MRI files
come from the public dataset on AWS (downloaded the first time into your data folder, like every other step), and
everything this script writes goes into THIS folder (new_patient_results\).

Which patients:
    (default)        the 15 patients WITHOUT intracranial EEG (MRI only) - the model never saw them
    --all-unseen     also the 5 iEEG patients whose labels were not good enough for training (PID03, 07, 16, 20, 27)
                     -> all 20 patients the model was never trained on
    --subjects sub-PID06 sub-PID28 ...   only these patients
Only 10 patients were used for training (those with reliable iEEG labels); a training patient given with --subjects is
shown with the models of his own cross-validation fold (they never saw him), exactly like 08_predict_mri.py.

For every patient, in new_patient_results\<patient>\:
    <patient>_ez_probability.nii.gz   probability map (open it on top of the patient's norm.mgz - see README.txt)
    <patient>_ez_zone.nii.gz          the highest 2% of the assessed grey matter: the model's 'zone' (a segmentation)
    <patient>_ez_probability.png      three views through the map's maximum + side + regions
    <patient>_regions.csv             mean probability of every brain region
    <patient>_summary.txt
For all patients, in new_patient_results\:
    comparison_<which>.csv            one row per patient: model against the clinical record (and the file paths)
    comparison_<which>.txt            the table and the accuracy  <- send this to Claude
    run_log_<which>.txt               everything printed
    (<which> = mri_only, all_unseen or selected, so the three ways of running do not overwrite each other)
Run (from the project folder, after 07_lateralization.py and the Colab training):
    python new_patient_results\run_new_patients.py
About 1-3 minutes per patient on a laptop. --step 5 gives finer maps (slower), --step 8 is faster.
This is a research prototype trained on 10 patients - not a clinical tool.
"""
import argparse
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import aws_data  # noqa: E402
import common  # noqa: E402
import config  # noqa: E402

LOBES = ("temporal", "frontal", "parietal", "occipital", "insula")


class Tee:
    """Print to the screen and to the run log."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.screen = sys.stdout

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()


def load_predict_module():
    """08_predict_mri.py as a module (its functions are used unchanged)."""
    spec = importlib.util.spec_from_file_location("predict_mri", os.path.join(PROJECT, "08_predict_mri.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def side_of(text):
    low = str(text).strip().lower()
    return "L" if low.startswith(("left only", "left >")) else "R" if low.startswith(("right only", "right >")) else ""


def lesion_side(text):
    return {"left": "L", "right": "R", "bilateral": "B"}.get(str(text).strip().lower(), "")


def clinical_table():
    """{subject: {column: text}} from metadata_3T_7T.xlsx ('' for empty cells)."""
    df = aws_data.metadata_table()
    if df is None:
        raise SystemExit("metadata_3T_7T.xlsx not found - run SOZ\\01_scan_dataset.py first.")
    cols = {str(c).strip().lower().rstrip(":").strip(): c for c in df.columns}
    want = {"side": "seizure onset lateralization", "lobe": "seizure onset localization", "mri": "mri result",
            "lesion": "mri lesion type", "lesion_side": "mri lesion laterality", "lesion_where": "mri lesion localization",
            "mri7": "7t mri result", "lesion7": "7t mri lesion type", "lesion7_side": "7t mri lesion laterality",
            "lesion7_where": "7t mri lesion localization"}
    sub_col = next((c for k, c in cols.items() if "subject" in k or "participant" in k), None)
    out = {}
    for _, r in df.iterrows():
        sid = str(r[sub_col]).strip()
        if not sid or sid.lower() == "nan":
            continue
        sid = sid if sid.startswith("sub-") else "sub-" + sid
        out[sid] = {k: (str(r[cols[v]]).strip() if v in cols and str(r[cols[v]]).strip().lower() != "nan" else "")
                    for k, v in want.items()}
    return out


def has_ieeg(subjects):
    """{subject: True if the patient has intracranial EEG recordings} - from step 1's inventory when it exists,
    otherwise from the dataset's folders on AWS (a session 'ses-clinical...' holds the iEEG)."""
    inv = os.path.join(config.SOZ_RESULTS_DIR, "dataset_inventory.csv")
    if os.path.exists(inv):
        t = pd.read_csv(inv).set_index("subject")
        if "n_seizure_recordings" in t.columns:
            return {s: bool(t.loc[s, "n_seizure_recordings"] > 0) if s in t.index else False for s in subjects}
    out = {}
    for s in subjects:
        _, folders = aws_data.listing(f"sourcedata/{s}/")
        out[s] = any("ses-clinical" in f for f in folders)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def save_patient(P, name, card, res, summ, note, side_how, lab_img, vols, out_dir, zone_percent, voxel_mm):
    """The same files as 08_predict_mri.py, written into this folder."""
    os.makedirs(out_dir, exist_ok=True)
    files = {"map": os.path.join(out_dir, f"{name}_ez_probability.nii.gz"),
             "zone": os.path.join(out_dir, f"{name}_ez_zone.nii.gz"),
             "picture": os.path.join(out_dir, f"{name}_ez_probability.png")}
    nib.save(nib.Nifti1Image(res["prob"], lab_img.affine), files["map"])
    nib.save(nib.Nifti1Image(summ["zone"].astype(np.uint8), lab_img.affine), files["zone"])
    res["regions"].to_csv(os.path.join(out_dir, f"{name}_regions.csv"), index=False)
    P.picture(files["picture"], name, np.asarray(vols["T1"], np.float32), res, summ, note)
    sm = summ["side_model"]
    lr = P.LEFT_RIGHT
    lines = [f"{name}", f"models: {note}",
             f"side (CNN): {lr[summ['pred_side']]} (mean probability over the assessed grey matter: left "
             f"{summ['mean_left']:.3f}, right {summ['mean_right']:.3f})",
             "side (side model of 07_lateralization.py): "
             + (f"{lr.get(sm, sm)} (p(left) = {summ['side_model_p_left']:.2f}; {side_how})" if sm else side_how),
             f"top regions: {', '.join(summ['top_regions'])}",
             f"zone = highest {zone_percent:g}% of the assessed grey matter (cortex, hippocampus, amygdala): "
             f"{summ['zone_ml']:.1f} ml (probability >= {res['zone_threshold']:.2f}); deep structures not assessed",
             f"hippocampus: left {summ['hip_left']:.3f}, right {summ['hip_right']:.3f}",
             f"grid: every {res['grid_mm']:.1f} mm, {len(res['probs'])} points; MRI voxel size {voxel_mm:.2f} mm",
             f"Research prototype trained on {len(card.get('training_subjects', []))} patients - not a clinical "
             "tool."]
    with open(os.path.join(out_dir, f"{name}_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return files


def main():
    ap = argparse.ArgumentParser(description="The final model on the patients it was never trained on, against the "
                                             "clinical record. Writes only into new_patient_results\\.")
    ap.add_argument("--all-unseen", action="store_true", dest="all_unseen",
                    help="also the 5 iEEG patients not used for training (20 patients in all)")
    ap.add_argument("--subjects", nargs="+", help="only these patients, e.g. sub-PID06 sub-PID28")
    ap.add_argument("--step", type=float, default=6.0, help="mm between predictions (default 6)")
    ap.add_argument("--zone-percent", type=float, default=2.0, dest="zone_percent")
    args = ap.parse_args()
    tag = "selected" if args.subjects else "all_unseen" if args.all_unseen else "mri_only"
    sys.stdout = Tee(os.path.join(HERE, f"run_log_{tag}.txt"))

    P = load_predict_module()
    card = P.read_card()                                   # refuses a model that is not version 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = set(card.get("training_subjects", []))
    everyone = [s for s in aws_data.subjects() if common.has_mri(s, "aparc+aseg.mgz") and common.has_mri(s, "norm.mgz")]
    if args.subjects:
        subs = [s if s.startswith("sub-") else "sub-" + s for s in args.subjects]
        which = "the patients given with --subjects"
    elif args.all_unseen:
        subs = [s for s in everyone if s not in trained]
        which = "every patient the model was never trained on (MRI-only + iEEG patients not used for training)"
    else:
        ieeg = has_ieeg(everyone)
        subs = [s for s in everyone if not ieeg[s] and s not in trained]
        which = "the patients without intracranial EEG (MRI only)"
    if not subs:
        raise SystemExit("no patient to run")
    clinical = clinical_table()
    print(f"Model version {card['version']} ({card['channels']}), device {device}; {len(subs)} patients - {which}:")
    print("  " + ", ".join(subs))
    print(f"Results go to {HERE}\n")

    cache, rows = {}, []
    for no, s in enumerate(subs, start=1):
        files, note = P.choose_models(card, s, "auto")      # final models; a training patient: his fold's models
        key = tuple(files)
        if key not in cache:
            cache[key] = P.build_models(card, files, device)
        lab_img, labels, vols = P.load_brain(s, None, any(c.startswith("T2") for c in card["channels"]))
        voxel_mm = common.voxel_size(lab_img)
        print(f"[{no}/{len(subs)}] {s}: MRI voxels {voxel_mm:.2f} mm; {note}")
        res = P.predict_brain(labels, vols, card, cache[key], device, args.step, args.zone_percent, voxel_mm)
        summ = P.summarise(res, args.zone_percent, voxel_mm)
        summ["side_model"], summ["side_model_p_left"], side_how = P.side_model_answer(s, None, labels)
        paths = save_patient(P, s, card, res, summ, note, side_how, lab_img, vols, os.path.join(HERE, s),
                             args.zone_percent, voxel_mm)
        c = clinical.get(s, {})
        side = side_of(c.get("side", ""))
        lobe = c.get("lobe", "").strip().lower()
        lobe = lobe if lobe in LOBES else ""
        les_side = lesion_side(c.get("lesion_side", ""))
        les_where = c.get("lesion_where", "").lower()
        mts = common.is_mts(c.get("lesion", "")) and les_side in ("L", "R")
        hl, hr = summ["hip_left"], summ["hip_right"]
        sm = summ["side_model"]
        rows.append({
            "subject": s, "used_for_training": s in trained, "models": note,
            "clinical_side_text": c.get("side", ""), "clinical_side": side,
            "cnn_side": summ["pred_side"], "cnn_left_minus_right": round(summ["margin"], 4),
            "cnn_side_correct": (summ["pred_side"] == side) if side else np.nan,
            "side_model": sm, "side_model_p_left": round(summ["side_model_p_left"], 3),
            "side_model_correct": (sm == side) if side and sm in ("L", "R") else np.nan,
            "clinical_lobe_text": c.get("lobe", ""), "top_region": summ["top_regions"][0] if summ["top_regions"] else "",
            "top_lobe": summ["top_lobe"], "lobe_correct": (summ["top_lobe"] == lobe) if lobe else np.nan,
            "mri_result": c.get("mri", ""), "mri_lesion": c.get("lesion", ""), "mri_lesion_side": les_side,
            "mri_lesion_location": c.get("lesion_where", ""),
            "cnn_side_matches_mri_lesion": (summ["pred_side"] == les_side) if les_side in ("L", "R") else np.nan,
            "top_lobe_in_mri_lesion_location": (summ["top_lobe"] in les_where) if les_where and summ["top_lobe"]
            else np.nan,
            "hippocampal_sclerosis_side": les_side if mts else "", "hippocampus_left": round(hl, 3),
            "hippocampus_right": round(hr, 3),
            "sick_hippocampus_higher": ((hl > hr) == (les_side == "L")) if mts and np.isfinite(hl + hr) else np.nan,
            "mri_7T_result": c.get("mri7", ""), "mri_7T_lesion": c.get("lesion7", ""),
            "mri_7T_lesion_side": c.get("lesion7_side", ""), "mri_7T_lesion_location": c.get("lesion7_where", ""),
            "zone_ml": round(summ["zone_ml"], 1), "map_file": paths["map"], "zone_file": paths["zone"],
            "picture_file": paths["picture"], "t1_to_view": common.mri_path(s, "norm.mgz")})
        print(f"    CNN side {summ['pred_side']} (clinical: {c.get('side', '') or '?'}); side model "
              f"{sm or '-'}; top region {rows[-1]['top_region']} (clinical lobe: {c.get('lobe', '') or '?'})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, f"comparison_{tag}.csv"), index=False)
    lines = ["=" * 110, "THE FINAL MODEL ON PATIENTS IT WAS NEVER TRAINED ON - against the clinical record "
             "(metadata_3T_7T.xlsx)", "=" * 110,
             f"{len(df)} patients: {which}.  Models: version {card['version']}, trained on {len(trained)} iEEG patients.",
             ""]
    show = df[["subject", "clinical_side_text", "cnn_side", "cnn_side_correct", "side_model", "clinical_lobe_text",
               "top_region", "mri_lesion", "mri_lesion_side"]].copy()
    show["mri_lesion"] = show["mri_lesion"].str.slice(0, 32)
    lines.append(show.to_string(index=False))
    lines.append("")
    known = df[df["clinical_side"].isin(["L", "R"])]
    n = len(known)
    if n:
        k = int(known["cnn_side_correct"].astype(float).sum())
        lo, hi = wilson(k, n)
        y = (known["clinical_side"] == "L").astype(int).to_numpy()
        auc = common.safe_auroc(y, known["cnn_left_minus_right"].to_numpy(float))
        maj = known["clinical_side"].value_counts()
        lines.append(f"SIDE (CNN): correct in {k} of {n} patients with a clinical side = {k / n:.0%} (95% interval "
                     f"{lo:.0%}-{hi:.0%}); fair score (AUROC, not affected by a lean to one side) {auc:.2f}.")
        lines.append(f"      always guessing the more common side ({maj.index[0]}) would be right in "
                     f"{maj.iloc[0] / n:.0%}; the CNN said 'left' for {int((df['cnn_side'] == 'L').sum())} of {len(df)}.")
        sure = known["side_model_correct"].dropna().astype(float)
        lines.append(f"SIDE (side model of 07): correct in {int(sure.sum())} of {len(sure)} patients where it was sure "
                     f"({n - len(sure)} 'not sure').")
    else:
        lines.append("SIDE: no patient with a one-sided clinical record (Inconclusive / Bilateral are not counted).")
    les = df[df["cnn_side_matches_mri_lesion"].notna()]
    if len(les):
        lines.append(f"SIDE vs the MRI report: the CNN's side = the side of the one-sided MRI lesion in "
                     f"{int(les['cnn_side_matches_mri_lesion'].astype(bool).sum())} of {len(les)} patients "
                     f"({', '.join(les['subject'])}).")
    hs = df[df["sick_hippocampus_higher"].notna()]
    if len(hs):
        lines.append(f"HIPPOCAMPAL SCLEROSIS: the sick hippocampus scored above the healthy one in "
                     f"{int(hs['sick_hippocampus_higher'].astype(bool).sum())} of {len(hs)} patients "
                     f"({', '.join(hs['subject'])}).")
    lob = df[df["lobe_correct"].notna()]
    if len(lob):
        lines.append(f"LOBE: the top region's lobe = the clinical lobe in {int(lob['lobe_correct'].astype(bool).sum())} "
                     f"of {len(lob)} patients (always saying 'temporal' would be right in "
                     f"{(lob['clinical_lobe_text'].str.lower() == 'temporal').mean():.0%}).")
    lines += ["",
              "How to read this: patients with an Inconclusive or Bilateral clinical side are shown but not counted for",
              "the side. With so few patients the 95% intervals are wide. The model's design was changed once after its",
              "first test on these patients (version 8.1 -> 9), so say so when you report these numbers.",
              "Maps: open each <patient>\\<patient>_ez_probability.nii.gz on top of the file in the column t1_to_view of",
              f"comparison_{tag}.csv (README.txt). Research prototype trained on {len(trained)} patients - not a "
              "clinical tool."]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(HERE, f"comparison_{tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\nSaved comparison_{tag}.csv, comparison_{tag}.txt and one folder per patient in {HERE}")


if __name__ == "__main__":
    main()
