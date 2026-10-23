"""
SOZ/03_build_contacts.py  -  STEP 3: put the SOZ labels and the electrode coordinates together.

Run from the project folder (after step 2):    python SOZ\\03_build_contacts.py

For every patient with SOZ labels it:
  1. downloads the iEEG-recon electrode coordinates and the MRI they were made on (research 3T),
     plus FreeSurfer's norm.mgz and aparc+aseg (called aparc_aseg.mgz in this dataset),
  2. converts the coordinates to scanner RAS millimetres; when the mm file and the voxel file disagree,
     it keeps the version whose contacts lie in/next to the FreeSurfer brain,
  3. checks that this MRI and the FreeSurfer MRI are in the same space (image correlation test),
     and that most contacts lie in or next to the brain (the outer contacts of depth electrodes sit in
     the skull, so 70-90% is normal),
  4. gives every site (contact, or the midpoint of a bipolar pair) its x, y, z, hemisphere and lobe,
  5. compares the side and lobe of the SOZ sites with the clinical seizure-onset lateralization and
     localization in metadata_3T_7T.xlsx (the multidisciplinary surgery conference's decision; a sanity check only -
     the clinical columns are not used as labels).

Patients listed in EXCLUDE_FROM_TRAINING in config.py (for example because their seizures start on contacts
that have no coordinates - see tools/check_unlocalized.py) are kept in contacts_all.csv but not in contacts.csv.

Every site keeps two labels from step 2: soz (1 = seizure onset zone) and uncertain (1 = e.g. it is an onset site in
too few seizures, or was already active before the onset; the model steps use such sites neither as SOZ nor as
'not SOZ'). Version 6.1: SOZ sites on the side that holds less than MINORITY_SIDE_SHARE (25%) of a patient's SOZ sites
become uncertain too (on the expert benchmark every patient's onset channels were on one side). For comparison the
side and lobe of the version 5 (EI) SOZ are checked against the clinical record too.

Results (in SOZ/results/):
  contacts.csv            training table: patients whose labels are good enough (label quality ok / few seizures)
  contacts_all.csv        every patient that passed the coordinate checks (also the ones left out of training)
  coordinate_check.csv    one row per patient: coordinate source, alignment test, clinical comparison
  step3_report.txt        the summary printed at the end  <- send this to Claude
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

import aws_data  # noqa: E402
import common  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402

SIDE = {1: "L", 2: "R"}
# The outer contacts of depth (stereo-EEG) electrodes sit in the skull, so 70-90% of contacts within 5 mm of the
# brain is normal for those patients. Below this share the coordinates are probably in the wrong space.
MIN_NEAR_BRAIN = 0.5


def clinical_table():
    """{subject: (side 'L'/'R'/'', lobe, original texts)} from metadata_3T_7T.xlsx."""
    df = aws_data.metadata_table()
    if df is None:
        return {}
    cols = {str(c).strip().lower().rstrip(":").strip(): c for c in df.columns}
    sub_col = next((c for k, c in cols.items() if "subject" in k or "participant" in k), None)
    lat_col = next((c for k, c in cols.items() if "onset lateral" in k), None)
    loc_col = next((c for k, c in cols.items() if "onset locali" in k), None)
    if sub_col is None:
        return {}
    out = {}
    for _, r in df.iterrows():
        sid = str(r[sub_col]).strip()
        if not sid or sid.lower() == "nan":
            continue
        sid = sid if sid.startswith("sub-") else "sub-" + sid
        lat = str(r[lat_col]).strip() if lat_col is not None and pd.notna(r[lat_col]) else ""
        loc = str(r[loc_col]).strip() if loc_col is not None and pd.notna(r[loc_col]) else ""
        low = lat.lower()
        side = "L" if low.startswith(("left only", "left >")) else "R" if low.startswith(("right only", "right >")) else ""
        lobe = next((lb for lb in ("temporal", "frontal", "parietal", "occipital", "insula")
                     if loc.lower().startswith(lb)), "")
        out[sid] = {"clinical_side": side, "clinical_lobe": lobe, "clinical_lateralization": lat,
                    "clinical_localization": loc}
    return out


def to_vox(img, xyz):
    return nib.affines.apply_affine(np.linalg.inv(img.affine), np.asarray(xyz, float).reshape(-1, 3))


def near_fraction(near, img, df):
    ijk = np.round(to_vox(img, df[["x", "y", "z"]].values)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(near.shape)), axis=1)
    ok = np.zeros(len(ijk), bool)
    ok[inside] = near[tuple(ijk[inside].T)]
    return float(ok.mean()) if len(ok) else 0.0


def site_table(lab, coords):
    """x, y, z of every labelled site: a contact directly, a bipolar pair 'A-B' at the midpoint of A and B."""
    xyz = {n: np.array([x, y, z]) for n, x, y, z in coords[["contact_norm", "x", "y", "z"]].itertuples(index=False)}
    pts = []
    for norm in lab["contact_norm"]:
        parts = [xyz.get(p) for p in str(norm).split("-")]
        pts.append(np.mean(parts, axis=0) if parts and all(p is not None for p in parts) else [np.nan] * 3)
    pts = np.array(pts, dtype=float).reshape(-1, 3)
    out = lab.copy()
    out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    return out


def majority(values):
    v = pd.Series([x for x in values if x])
    if v.empty:
        return ""
    counts = v.value_counts()
    return counts.index[0] if len(counts) == 1 or counts.iloc[0] > counts.iloc[1] else "mixed"


def count_text(values):
    """'temporal 3, insula 1' - every lobe (or side) of the SOZ sites, most common first."""
    v = pd.Series([x if x else "?" for x in values])
    return ", ".join(f"{k} {n}" for k, n in v.value_counts().items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-unaligned", action="store_true",
                    help="keep patients whose MRI alignment test failed (not recommended)")
    args = ap.parse_args()

    out = config.SOZ_RESULTS_DIR
    lab_path = os.path.join(out, "soz_labels.csv")
    if not os.path.exists(lab_path):
        raise SystemExit("Run step 2 first:  python SOZ\\02_compute_soz.py")
    labels = pd.read_csv(lab_path, keep_default_na=False, na_values=[""])
    summary = pd.read_csv(os.path.join(out, "soz_summary.csv")).set_index("subject")
    v5_path = os.path.join(out, "soz_labels_ei_v5.csv")
    v5 = pd.read_csv(v5_path) if os.path.exists(v5_path) else pd.DataFrame(columns=["subject", "contact_norm", "soz_ei"])
    inv = pd.read_csv(os.path.join(out, "dataset_inventory.csv"), keep_default_na=False).set_index("subject")
    clinical = clinical_table()

    rows, checks = [], []
    for sub, lab in labels.groupby("subject"):
        r = inv.loc[sub]
        print(f"\n{sub}")
        chk = {"subject": sub}
        try:
            rec = {"names": r["recon_names"], "coords_mm": r["recon_coords_mm"], "coords_vox": r["recon_coords_vox"],
                   "t1": r["recon_t1"]}
            if not rec["t1"]:
                rec = aws_data.recon_files(sub)   # inventory from an older version of step 1
            names = aws_data.fetch(rec["names"], missing_ok=True) if rec["names"] else None
            mm = aws_data.fetch(rec["coords_mm"], missing_ok=True) if rec["coords_mm"] else None
            vox = aws_data.fetch(rec["coords_vox"], missing_ok=True) if rec["coords_vox"] else None
            t1_path = aws_data.fetch(rec["t1"], missing_ok=True) if rec["t1"] else None
            t1 = nib.load(t1_path) if t1_path else None
            print(f"    reference MRI: {rec['t1'].split('/')[-1] if t1_path else 'not found'}")

            fs_norm = common.load_mgz(sub, "norm.mgz")
            fs_lab = common.load_mgz(sub, "aparc+aseg.mgz")
            seg = np.asarray(fs_lab.dataobj).astype(np.int32)
            # brain + 5 voxels = 5 mm on the 1 mm MRIs, 4 mm on the 0.8 mm MRIs (only a check of the coordinates; the
            # model steps use 5 mm for everyone - common.near_brain_mask). Grids lie on the surface.
            near = ndimage.binary_dilation(seg > 0, iterations=5)

            cands, info = soz_lib.coordinate_candidates(names, mm, vox, t1)
            if not cands:
                raise ValueError("no electrode coordinates found")
            chk.update(info)
            frac = {k: near_fraction(near, fs_lab, df) for k, df in cands.items()}
            agree = min(info.get("mm_vs_vox_0based_mm", np.inf), info.get("mm_vs_vox_1based_mm", np.inf)) < 1.5
            if "mm" in cands and (agree or len(cands) == 1):
                pick = "mm"
                src = "mm file (agrees with voxel file + MRI header)" if agree else "mm file (no voxel file / MRI to check)"
            else:
                order = ["mm", "vox0", "vox1"]
                pick = max(cands, key=lambda k: (round(frac[k], 2), -order.index(k)))
                src = f"{pick} (mm and voxel files disagree; this one fits the brain best: " + \
                      ", ".join(f"{k} {v:.0%}" for k, v in frac.items()) + ")"
            coords = cands[pick]
            chk["coord_source"] = src
            print(f"    coordinates: {len(coords)} contacts from {src}")

            if t1 is not None:
                sc = soz_lib.space_check(fs_norm, fs_lab, t1)
                chk.update(sc)
                print(f"    MRI alignment test: correlation {sc['corr_same_position']} at the same position vs "
                      f"{sc['corr_best_shifted']} when shifted 4 mm -> {'ALIGNED' if sc['aligned'] else 'NOT ALIGNED'}")
            else:
                chk["aligned"] = None
            chk["contacts_near_brain"] = round(frac[pick], 3)
            print(f"    contacts in or within 5 voxels (4-5 mm) of the brain: {frac[pick]:.0%}")

            sites = site_table(lab, coords)
            chk["sites_labelled"] = len(sites)
            chk["sites_without_coords"] = int(sites["x"].isna().sum())
            sites = sites.dropna(subset=["x"]).reset_index(drop=True)
            side_vol = common.side_codes(seg)
            ijk = to_vox(fs_lab, sites[["x", "y", "z"]].values)
            sides, lobes = [], []
            for v in ijk:
                rid, _ = common.assign_region(seg, v, 5.0)
                lobes.append(common.lobe(rid) if rid else "")
                sides.append(SIDE.get(common.nearest_code(side_vol, v, 5.0), ""))
            sites["mri_side"], sites["mri_lobe"] = sides, lobes
            if "uncertain" not in sites.columns:
                sites["uncertain"] = 0
            sites["uncertain"] = sites["uncertain"].fillna(0).astype(int)
            if "uncertain_reason" not in sites.columns:
                sites["uncertain_reason"] = ""
            sites["uncertain_reason"] = sites["uncertain_reason"].fillna("").astype(str)
            # SOZ sites on the side that holds only a small share of the patient's SOZ sites -> uncertain (on the
            # expert benchmark every patient's onset channels were on ONE side)
            chk["soz_other_side"] = 0
            known = sites[(sites["soz"] == 1) & sites["mri_side"].isin(["L", "R"])]
            share = float(getattr(config, "MINORITY_SIDE_SHARE", 0) or 0)
            if share > 0 and len(known) >= 2 and not (sites.get("label_source", pd.Series(dtype=str))
                                                       .astype(str).eq("clinician marks").any()):
                cnt = known["mri_side"].value_counts()
                minor = [side for side, k in cnt.items() if k < share * len(known)]
                m = (sites["soz"] == 1) & sites["mri_side"].isin(minor)
                if m.any():
                    sites.loc[m, "soz"] = 0
                    sites.loc[m, "uncertain"] = 1
                    sites.loc[m, "uncertain_reason"] = "SOZ, but on the other side than most of the SOZ"
                    chk["soz_other_side"] = int(m.sum())
                    print(f"    {int(m.sum())} SOZ site(s) on the other side than most of the SOZ -> uncertain: "
                          f"{', '.join(sites.loc[m, 'contact'].astype(str))}")
            chk["sites_used"] = len(sites)
            chk["soz_sites_used"] = int(sites["soz"].sum())
            chk["uncertain_sites_used"] = int(sites["uncertain"].sum())
            print(f"    matched {len(sites)} labelled sites ({chk['sites_without_coords']} without coordinates)"
                  + (f"; {chk['uncertain_sites_used']} uncertain (not used as 'not SOZ')"
                     if chk["uncertain_sites_used"] else ""))

            s = sites[sites["soz"] == 1]
            soz_side, soz_lobe = majority(s["mri_side"]), majority(s["mri_lobe"])
            cl = clinical.get(sub, {})
            # the version 5 (EI) SOZ of the same patient, for comparison
            v5s = set(v5.loc[(v5["subject"] == sub) & (v5["soz_ei"] == 1), "contact_norm"]) if len(v5) else set()
            s5 = site_table(pd.DataFrame({"contact_norm": sorted(v5s)}), coords).dropna(subset=["x"])
            l5, d5 = [], []
            for v in to_vox(fs_lab, s5[["x", "y", "z"]].values) if len(s5) else []:
                rid, _ = common.assign_region(seg, v, 5.0)
                l5.append(common.lobe(rid) if rid else "")
                d5.append(SIDE.get(common.nearest_code(side_vol, v, 5.0), ""))
            v5_side, v5_lobe = majority(d5), majority(l5)
            chk["v5_soz_side"], chk["v5_soz_lobe"] = v5_side, v5_lobe
            chk["v5_side_agrees"] = (v5_side == cl["clinical_side"]) if v5_side in ("L", "R") and cl.get("clinical_side") \
                else None
            chk["v5_lobe_agrees"] = (v5_lobe == cl["clinical_lobe"]) if v5_lobe not in ("", "mixed") and \
                cl.get("clinical_lobe") else None
            chk.update({"soz_side_mri": soz_side, "soz_lobe_mri": soz_lobe,
                        "soz_sides_all": count_text(s["mri_side"]) if len(s) else "",
                        "soz_lobes_all": count_text(s["mri_lobe"]) if len(s) else "",
                        "clinical_lateralization": cl.get("clinical_lateralization", ""),
                        "clinical_localization": cl.get("clinical_localization", "")})
            chk["side_agrees"] = (soz_side == cl["clinical_side"]) if soz_side in ("L", "R") and cl.get("clinical_side") \
                else None
            chk["lobe_agrees"] = (soz_lobe == cl["clinical_lobe"]) if soz_lobe not in ("", "mixed") and \
                cl.get("clinical_lobe") else None
            if len(s):
                print(f"    SOZ sites: side {chk['soz_sides_all']}; lobes {chk['soz_lobes_all']}   |   clinical "
                      f"onset: {cl.get('clinical_lateralization') or '?'}, {cl.get('clinical_localization') or '?'}")

            quality = summary.loc[sub, "label_quality"] if sub in summary.index else ""
            train = bool(summary.loc[sub, "use_for_training"]) if sub in summary.index else False
            excluded = str((getattr(config, "EXCLUDE_FROM_TRAINING", None) or {}).get(sub, "") or "")
            sites["label_quality"] = quality
            good = (chk.get("aligned") is not False) and chk["contacts_near_brain"] >= MIN_NEAR_BRAIN
            chk["label_quality"] = quality
            chk["excluded_reason"] = excluded
            chk["used"] = bool(good or args.keep_unaligned)
            chk["used_for_training"] = bool(chk["used"] and (train or not config.TRAIN_ONLY_CONSISTENT)
                                            and not excluded)
            if not good:
                print("    WARNING: alignment or brain-contact check failed - "
                      + ("kept (--keep-unaligned)" if args.keep_unaligned else "patient left out"))
            elif excluded:
                print(f"    left out of training: {excluded}  (EXCLUDE_FROM_TRAINING in config.py)")
            elif not chk["used_for_training"]:
                print(f"    left out of training: label quality '{quality}'")
            if chk["used"]:
                sites["use_for_training"] = chk["used_for_training"]
                rows.append(sites)
        except Exception as e:  # noqa: BLE001
            chk["used"] = False
            chk["error"] = f"{type(e).__name__}: {e}"
            print(f"    SKIPPED - {chk['error']}")
        checks.append(chk)

    check = pd.DataFrame(checks)
    check.to_csv(os.path.join(out, "coordinate_check.csv"), index=False)
    if not rows:
        raise SystemExit("No patient passed the checks - see coordinate_check.csv")
    allc = pd.concat(rows, ignore_index=True)
    keep = ["subject", "contact", "x", "y", "z", "soz", "uncertain", "uncertain_reason", "label_source",
            "label_quality", "use_for_training", "n_seizures", "n_onset", "oi_mean", "oi_max", "start_s_median",
            "ei_v5_soz", "montage", "hemisphere", "mri_side", "mri_lobe"]
    allc = allc[[c for c in keep if c in allc.columns]]
    allc[["x", "y", "z"]] = allc[["x", "y", "z"]].round(3)
    allc.to_csv(os.path.join(out, "contacts_all.csv"), index=False)
    train = allc[allc["use_for_training"].astype(bool)]
    if train.empty:
        raise SystemExit("No patient has labels good enough for training - see soz_summary.csv")
    train.to_csv(config.CONTACTS_CSV, index=False)

    lines = []

    def say(txt=""):
        print(txt)
        lines.append(txt)

    say("\n" + "=" * 100)
    t = check.copy()
    for c in ("aligned", "contacts_near_brain", "soz_sites_used", "soz_side_mri", "clinical_lateralization",
              "side_agrees", "soz_lobe_mri", "clinical_localization", "lobe_agrees", "label_quality",
              "used_for_training", "excluded_reason"):
        if c not in t.columns:
            t[c] = None
    excl = t["excluded_reason"].fillna("").astype(str) != ""
    t.loc[excl, "label_quality"] = "excluded"
    t = t[["subject", "aligned", "contacts_near_brain", "soz_sites_used", "soz_side_mri", "clinical_lateralization",
           "side_agrees", "soz_lobe_mri", "clinical_localization", "lobe_agrees", "label_quality", "used_for_training"]]
    t.columns = ["subject", "aligned", "near_brain", "SOZ", "SOZ side", "clinical side", "agree", "SOZ lobe",
                 "clinical lobe", "agree ", "label quality", "train"]
    say(t.to_string(index=False, max_colwidth=16))
    say("=" * 100)

    def agree_text(col_side, col_lobe, who):
        sa = check[col_side].dropna() if col_side in check else pd.Series(dtype=object)
        la = check[col_lobe].dropna() if col_lobe in check else pd.Series(dtype=object)
        if len(sa) or len(la):
            say(f"{who}: side agrees with the clinical lateralization in {int(sa.astype(bool).sum())} of {len(sa)} "
                f"patients; lobe agrees in {int(la.astype(bool).sum())} of {len(la)}.")

    agree_text("side_agrees", "lobe_agrees", "SOZ version 6.1")
    agree_text("v5_side_agrees", "v5_lobe_agrees", "SOZ version 5 (EI), same patients")
    say("(The clinical lateralization / localization is the decision of the multidisciplinary epilepsy surgery "
        "conference (dataset paper, Imtiaz et al. 2026) - a strong clinical reference, but not an independent "
        "ground truth: for the patients with iEEG it may already include the iEEG.)")
    say("SOZ lobe = the most common lobe of the SOZ sites (all lobes are listed above for every patient); "
        "'mixed' = no single most common lobe.")
    warn = check[check.get("used_for_training", pd.Series(False, index=check.index)).fillna(False).astype(bool)
                 & ((check.get("side_agrees") == False) | (check.get("lobe_agrees") == False))]  # noqa: E712
    if len(warn):
        say("\nPLEASE CHECK (used for training, but the SOZ is not where the clinical record says): "
            + ", ".join(warn["subject"]) + "  -> look at their plots (tools\\plot_seizure.py --audit) before training.")
    say(f"\ncontacts.csv: {len(train)} sites, {int(train['soz'].sum())} SOZ, {int(train['uncertain'].sum())} "
        f"uncertain (not used as 'not SOZ'), {train['subject'].nunique()} patients for training -> "
        f"{config.CONTACTS_CSV}")
    say(f"  patients: {', '.join(sorted(train['subject'].unique()))}")
    say(f"contacts_all.csv: {len(allc)} sites from {allc['subject'].nunique()} patients (all that passed the checks)")
    say("Next: python tools\\plot_seizure.py --audit   then   python 01_label_contacts.py  (and the model steps)")
    with open(os.path.join(out, "step3_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    with runlog.Log("03_build_contacts"):
        main()
