"""
smoke_test.py  -  runs the WHOLE project on fake patients, so you know your installation works (version 9).

Run from the project folder:   python tools\\smoke_test.py
Takes about 20-30 minutes on a laptop CPU (the CNN step is the slow one; it is only 3 epochs here).
Everything goes to synthetic_data/ and outputs_synthetic/; your real data and results are not touched
(also not the downloaded files in your data folder). Results on fake patients mean nothing clinically.

What it checks, in this order:
  1. the onset detector on fake signals with a known answer (tools/onset_selftest.py)
  2. the expert benchmark: a fake benchmark is "downloaded" from a local server exactly like the real one; the
     benchmark must test every method and choose one that beats the EI (the fake data contains rhythmic onsets
     the EI cannot see); the following steps then use "EI + line length", the method the real benchmark chose
  3. the SOZ steps 1-3 on fake patients that copy the real dataset's quirks (see tools/make_synthetic_data.py)
  4. tools/check_unlocalized.py, the model steps 01-07, and tools/plot_seizure.py (including --audit)
  5. version 8: the plane between the hemispheres (straight and tilted head) and the mirror cubes, the mirror
     controls, the CNN's left-right test and final models, 08_predict_mri.py (a training patient, a new FreeSurfer
     folder, --unseen), tools/model_report.py and tools/v7_car_linelength.py
  6. version 8.1: cubes in millimetres - the same brain at 1 mm and at 0.8 mm must give the same cube (5 of the fake
     patients have 0.8 mm MRIs, like 7 of the 10 real training patients), and tools/check_dataset_extras.py
  7. version 9: deep-structure 'not SOZ' cubes (far from every SOZ contact), the side model (07), the map only on the
     assessed regions with the deep structures 'not assessed', the fair side score, the sclerosis check, and the
     report's deep-structure and shortcut checks
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SOZ"))
sys.path.insert(0, os.path.join(HERE, "tools"))
VERSION = "9"
DATA_VERSION = "9"      # version of the fake patients (tools/make_synthetic_data.py); new fakes only when it changes


def run(cmd, env):
    print("\n" + "=" * 80 + "\n$ python " + " ".join(cmd) + "\n" + "=" * 80, flush=True)
    subprocess.run([sys.executable] + cmd, cwd=HERE, env=env, check=True)


def fail(msg):
    raise SystemExit(f"\nSMOKE TEST FAILED: {msg}\nPlease send me outputs_synthetic\\logs (zip it) and this message.")


def check_onset_detector():
    import onset_selftest
    print("\n" + "=" * 80 + "\nOnset detector on fake signals (known answer)\n" + "=" * 80, flush=True)
    problems = onset_selftest.main(verbose=True)
    if problems:
        fail("onset detector self-test: " + "; ".join(problems))
    print("Onset detector self-test passed.")


def run_benchmark(env):
    import range_server
    server, base = range_server.start(os.path.join(HERE, "synthetic_data", "benchmark_server"))
    try:
        url = base + "nm000402/objects/"
        run(["tools/get_benchmark.py", "--test", "--base-url", url], env)
        run(["tools/get_benchmark.py", "--base-url", url], env)
        run(["tools/get_benchmark.py", "--base-url", url], env)          # again: must keep what is there
    finally:
        server.shutdown()
    run(["tools/benchmark_onset.py"], env)
    path = os.path.join(HERE, env["EZ_SOZ_RESULTS_DIR"], "benchmark_choice.json")
    with open(path) as fh:
        ch = json.load(fh)
    if ch["method"] == "ei" or not ch["winner_passed_test"] or not ch["test_f1"] > ch["ei_test_f1"]:
        fail(f"the expert benchmark did not choose a method better than the EI on the fake data: {ch}")
    fams = set(pd.read_csv(os.path.join(HERE, env["EZ_SOZ_RESULTS_DIR"], "benchmark_scores.csv"))["family"])
    if fams != {"ei", "linelength", "multifeature", "ei+linelength"}:
        fail(f"the benchmark did not test every method ({sorted(fams)})")
    print(f"\nBenchmark checks passed: chosen {ch['method']} ({ch['montage']}, {ch['setting']}), test F1 "
          f"{ch['test_f1']:.2f} vs EI {ch['ei_test_f1']:.2f}.")
    # The steps after the benchmark use 'EI + line length' - the method the REAL benchmark chose - written into
    # benchmark_choice.json the way benchmark_onset.py writes it (the fake benchmark is too simple to tell the methods
    # apart: several score 1.00 on it).
    import soz_lib
    ch.update({"method": "ei+linelength", "montage": "bipolar", "setting": "EI>=0.3 | z=4 oi=0.3",
               "params": soz_lib.method_params("ei+linelength")})
    with open(path, "w") as fh:
        json.dump(ch, fh, indent=1)
    print("For the next steps the smoke test uses 'EI + line length' (the real benchmark's choice).")


def check_soz_labels(env):
    """Compare the SOZ labels with the answer planted in the fake recordings, and check the special cases
    the fake data was built to contain (see tools/make_synthetic_data.py)."""
    from soz_lib import normalize_name
    res = os.path.join(HERE, env["EZ_SOZ_RESULTS_DIR"])
    lab = pd.read_csv(os.path.join(res, "soz_labels.csv"))
    summ = pd.read_csv(os.path.join(res, "soz_summary.csv")).set_index("subject")
    per = pd.read_csv(os.path.join(res, "soz_onset_per_seizure.csv"))
    info = pd.read_csv(os.path.join(res, "seizure_info.csv"))
    inv = pd.read_csv(os.path.join(res, "dataset_inventory.csv"), keep_default_na=False).set_index("subject")
    truth = pd.read_csv(os.path.join(HERE, "synthetic_data", "truth_contacts.csv"))
    soz = {(s, normalize_name(c)) for s, c, y in zip(truth["subject"], truth["contact"], truth["soz"]) if y == 1}
    multifocal = set(truth.loc[truth["multifocal"], "subject"])
    problems = []
    montage = lab["montage"].iloc[0]

    lab["parts"] = lab["contact_norm"].astype(str).str.count("-") + 1
    lab["n_true"] = [sum((s, c) in soz for c in str(n).split("-")) for s, n in zip(lab["subject"], lab["contact_norm"])]
    cons = lab[~lab["subject"].isin(multifocal)]
    core, none = cons[cons["n_true"] == cons["parts"]], cons[cons["n_true"] == 0]
    sens, spec = core["soz_computed"].mean(), 1 - none["soz_computed"].mean()
    print(f"\nComputed labels vs planted SOZ ({montage} montage): sites fully in the SOZ found {sens:.0%} ({len(core)}), "
          f"sites outside the SOZ correctly negative {spec:.0%} ({len(none)})")
    if not (sens >= 0.8 and spec >= 0.95):
        problems.append(f"sensitivity {sens:.0%} / specificity {spec:.0%} too low")
    for sub in sorted(set(cons["subject"])):
        g = cons[cons["subject"] == sub]
        if not ((g["soz_computed"] == 1) & (g["n_true"] > 0)).any():
            problems.append(f"{sub}: no SOZ site found")
    for sub in multifocal:   # every seizure starts somewhere else: no SOZ can be trusted
        if (summ.loc[sub, "label_quality"] not in ("SOZ rarely leads", "no repeated onset")
                or bool(summ.loc[sub, "use_for_training"])):
            problems.append(f"{sub}: its seizures all start in different places but the patient was not left out "
                            f"({summ.loc[sub, 'label_quality']})")
    if "sub-PID09" in summ.index and int(summ.loc["sub-PID09", "seizures_usable"]) != 3:
        problems.append(f"sub-PID09: the seizure that starts with slow waves only was not used "
                        f"({summ.loc['sub-PID09', 'seizures_usable']} usable seizures instead of 3)")
    p8 = per[(per["subject"] == "sub-PID08") & (per["montage"] == "average")]
    info8 = info[info["subject"] == "sub-PID08"]
    if len(p8) and not ({"P03", "C03", "O01", "O02"} <= set(p8["contact"]) and not {"P3", "C3", "O1"} & set(p8["contact"])
                        and (info8["n_scalp_ignored"] == 5).all()):
        problems.append("sub-PID08: scalp EEG channels were mixed up with intracranial contacts")
    p2 = per[(per["subject"] == "sub-PID02") & (per["montage"] == "average")]
    if len(p2) and not ({"RG01", "RG02"} <= set(p2["contact"]) and {"G1", "G2"} <= set(p2["contact_norm"])):
        problems.append("sub-PID02: the electrode-name fix (CHANNEL_ALIASES) was not used")
    p7 = per[(per["subject"] == "sub-PID07") & (per["montage"] == "bipolar")]["contact_norm"].astype(str)
    if len(p7) and (p7.str.contains(r"G(?:4-[LR]?G5|8-[LR]?G9|12-[LR]?G13)$").any()
                    or not p7.str.contains(r"G1-[LR]?G2$").any()):
        problems.append("sub-PID07: grid contacts paired across rows")
    for sub in ("sub-PID01", "sub-PID02", "sub-PID03"):
        if sub in inv.index and not inv.loc[sub, "recon_t1"].endswith("_T1w_ras_defaced.nii.gz"):
            problems.append(f"{sub}: reference MRI of the older iEEG-recon layout not found")
    chk = pd.read_csv(os.path.join(res, "coordinate_check.csv"))
    if not chk["used"].all() or (chk["aligned"].astype(str) != "True").any():
        problems.append("coordinate checks failed: " + ", ".join(chk.loc[~chk["used"], "subject"]))
    c9 = chk.set_index("subject")
    if "sub-PID09" in c9.index and not (0.5 <= float(c9.loc["sub-PID09", "contacts_near_brain"]) < 0.9):
        problems.append("sub-PID09: its contacts in the skull were not handled as expected")
    if "sub-PID02" in c9.index and int(c9.loc["sub-PID02", "sites_without_coords"]) != 0:
        problems.append("sub-PID02: sites without coordinates after the name fix")
    contacts = pd.read_csv(os.path.join(res, "contacts.csv"))
    if set(contacts["subject"]) & multifocal:
        problems.append("a patient without a consistent SOZ is in the training table")
    allc = pd.read_csv(os.path.join(res, "contacts_all.csv"))
    if "sub-PID06" in set(contacts["subject"]) or "sub-PID06" not in set(allc["subject"]):
        problems.append("sub-PID06 (listed in EZ_EXCLUDE_FROM_TRAINING) was not left out of training correctly")

    role = {(s, normalize_name(c)): r for s, c, r in zip(truth["subject"], truth["contact"],
                                                         truth["role"].fillna("").astype(str))}

    def roles(sub, norm):
        return {role.get((sub, c), "") for c in str(norm).split("-")}

    g4 = lab[lab["subject"] == "sub-PID04"]
    if not any("focus_b" in roles("sub-PID04", n) and y == 1 for n, y in zip(g4["contact_norm"], g4["soz_computed"])):
        problems.append("sub-PID04: the second focus (it starts 2 of 7 seizures) was not found as SOZ")
    if "sub-PID04" in summ.index and (summ.loc["sub-PID04", "label_quality"] != "ok"
                                      or int(summ.loc["sub-PID04", "seizures_usable"]) != 7):
        problems.append(f"sub-PID04: label quality '{summ.loc['sub-PID04', 'label_quality']}', "
                        f"{summ.loc['sub-PID04', 'seizures_usable']} of 7 seizures usable (2 of them start with slow "
                        "waves after the slow waves of the normal period fade - like the real sub-PID04)")
    g6 = lab[lab["subject"] == "sub-PID06"]
    spread = [n for n, y in zip(g6["contact_norm"], g6["soz_computed"])
              if "spread" in roles("sub-PID06", n) and "soz" not in roles("sub-PID06", n) and y == 1]
    if spread:
        problems.append(f"sub-PID06: sites the seizure reaches 4 s after it starts were made SOZ ({', '.join(spread)})")
    if "sub-PID06" in summ.index and int(summ.loc["sub-PID06", "soz_sites"]) == 0:
        problems.append("sub-PID06: its seizures start 4 s before the marked onset and no SOZ was found")
    g1 = lab[lab["subject"] == "sub-PID01"]
    spor = g1[[roles("sub-PID01", n) == {"sporadic"} for n in g1["contact_norm"]]]
    if spor.empty or (spor["soz_computed"] == 1).any() or not (spor["uncertain"] == 1).any():
        problems.append("sub-PID01: the sites that started one extra seizure are not 'uncertain'")
    g7 = lab[(lab["subject"] == "sub-PID07") & (lab["soz_computed"] == 1)]
    if (g7["n_true"] == 0).any():
        problems.append(f"sub-PID07: muscle activity made SOZ sites outside the SOZ ({', '.join(g7.loc[g7['n_true'] == 0, 'contact'])})")
    g8 = lab[lab["subject"] == "sub-PID08"]
    popped = g8[[("pop" in roles("sub-PID08", n)) and (y == 1 or u == 1)
                 for n, y, u in zip(g8["contact_norm"], g8["soz_computed"], g8["uncertain"])]]
    if len(popped):
        problems.append(f"sub-PID08: the contact with electrode pops became an onset site ({', '.join(popped['contact'])})")
    if problems:
        fail("SOZ checks:\n  " + "\n  ".join(problems))
    print("SOZ checks passed: planted SOZ found, unreliable patient left out, slow-wave seizure used, scalp channels "
          "ignored (intracranial ones typed EEG kept), name fix used, grid rows kept apart, contacts in the skull "
          "handled, both iEEG-recon layouts read, patient listed in EXCLUDE_FROM_TRAINING left out, second focus found, "
          "early start handled, one-off start 'uncertain';\nversion 6: seizures starting with slow waves after fading "
          "slow waves used (PID04-like), muscle activity and electrode pops not taken for seizure starts.")


def check_unlocalized_tool(env):
    """tools/check_unlocalized.py: the fake patients have coordinates for every contact; with an electrode
    'hidden' it must say OK when that electrode is far from the SOZ and DOUBTFUL when it is in the SOZ."""
    path = os.path.join(HERE, env["EZ_SOZ_RESULTS_DIR"], "unlocalized_check.csv")
    truth = pd.read_csv(os.path.join(HERE, "synthetic_data", "truth_contacts.csv"))
    t1 = truth[truth["subject"] == "sub-PID01"]
    elec = t1["contact"].str.extract(r"^([A-Z]*[A-Z])\d+$")[0]
    soz_e = elec[t1["role"] == "soz"].mode()[0]
    special = set(elec[t1["role"].fillna("") != ""])
    far = [e for e in dict.fromkeys(elec) if e not in special][:2]
    run(["tools/check_unlocalized.py", "--all"], env)
    first = pd.read_csv(path)
    run(["tools/check_unlocalized.py", "--subject", "sub-PID01", "--hide"] + far, env)
    far = pd.read_csv(path).set_index("subject").loc["sub-PID01", "result"]
    run(["tools/check_unlocalized.py", "--subject", "sub-PID01", "--hide", soz_e], env)
    near = pd.read_csv(path).set_index("subject").loc["sub-PID01", "result"]
    if not (first["result"] == "nothing to check").all() or (far, near) != ("OK", "DOUBTFUL"):
        fail(f"check_unlocalized.py gave unexpected results ({far}, {near})")
    print("\ncheck_unlocalized.py works: contacts without coordinates far from the SOZ -> OK, in the SOZ -> DOUBTFUL.")


def rotation(a1, a2):
    a1, a2 = np.radians(a1), np.radians(a2)
    r1 = np.array([[np.cos(a1), 0, np.sin(a1)], [0, 1, 0], [-np.sin(a1), 0, np.cos(a1)]])
    r2 = np.array([[np.cos(a2), -np.sin(a2), 0], [np.sin(a2), np.cos(a2), 0], [0, 0, 1]])
    return r2 @ r1


def check_mirror_geometry():
    """Version 8: the plane between the hemispheres must be found also when the head is tilted, and the mirror cube of
    an exactly mirror-symmetric image must equal the cube itself (and minus it for an anti-symmetric image)."""
    import common
    import make_synthetic_data as msd
    from scipy import ndimage
    print("\n" + "=" * 80 + "\nMirror geometry (version 8) and cubes in mm (version 8.1)\n" + "=" * 80, flush=True)
    labels = msd.build_atlas()
    centre = np.array([128.0, 128.0, 128.0])
    problems = []
    for name, rot in (("straight", np.eye(3)), ("tilted 8 and 5 degrees", rotation(8, 5))):
        lab = labels if name == "straight" else ndimage.affine_transform(labels, rot.T, offset=centre - rot.T @ centre,
                                                                          order=0)
        plane = common.midsagittal_plane(lab)
        angle = np.degrees(np.arccos(min(1.0, abs(float(plane["normal"] @ (rot @ np.array([1.0, 0, 0])))))))
        cen, _ = common.label_centroids(lab)
        errs = [float(np.linalg.norm(common.reflect(cen[a], plane) - cen[b])) for a, b in ((17, 53), (10, 49), (18, 54))]
        print(f"  {name}: plane normal off by {angle:.2f} deg; mirrored centres of hippocampus, thalamus, amygdala "
              f"off by {max(errs):.2f} voxels")
        if angle > 1.0 or max(errs) > 1.5:
            problems.append(f"{name}: plane off by {angle:.2f} deg / {max(errs):.2f} voxels")
    rng = np.random.default_rng(0)
    n = 96
    f = ndimage.gaussian_filter(rng.normal(size=(n, n, n)).astype(np.float32), 3)
    plane = {"normal": rotation(8, 5) @ np.array([1.0, 0, 0]), "offset": float((rotation(8, 5) @ np.array([1.0, 0, 0]))
                                                                                @ np.full(3, n / 2))}
    pts = np.indices((n, n, n), dtype=np.float64).reshape(3, -1).T
    fm = ndimage.map_coordinates(f, common.reflect(pts, plane).T, order=1, mode="nearest").reshape(f.shape)
    cs = np.array([[30.3, 40.1, 50.7], [60.2, 45.5, 41.0], [47.9, 52.2, 48.3]])
    for name, vol, sign in (("symmetric", f + fm, 1.0), ("anti-symmetric", f - fm, -1.0)):
        pp = common.pair_patches(vol, cs, plane, 16).astype(np.float32)
        err = float(np.abs(pp[:, 0] - sign * pp[:, 1]).mean() / (np.abs(pp[:, 0]).mean() + 1e-9))
        print(f"  {name} image: cube and mirror cube differ by {err:.1%}")
        if err > 0.05:
            problems.append(f"{name} image: mirror cube differs by {err:.1%}")
    # version 8.1: the same brain in 1 mm and in 0.8 mm voxels must give the same cube (and mirror cube)
    big = ndimage.gaussian_filter(rng.normal(size=(128, 128, 128)).astype(np.float32), 3)
    hr = ndimage.affine_transform(big, np.full(3, 0.8), output_shape=(160, 160, 160), order=1, output=np.float32)
    p1 = {"normal": np.array([1.0, 0.0, 0.0]), "offset": 64.0}
    p2 = {"normal": np.array([1.0, 0.0, 0.0]), "offset": 64.0 / 0.8}
    cs = np.array([[50.3, 45.2, 70.7], [80.0, 64.0, 64.0], [35.5, 75.1, 48.9]])
    a = common.pair_patches(big, cs, p1, 32, 1.0).astype(np.float32)
    same = common.pair_patches(hr, cs / 0.8, p2, 32, 1.0 / 0.8).astype(np.float32)
    old = common.pair_patches(hr, cs / 0.8, p2, 32, 1.0).astype(np.float32)          # cubes in voxels (< 8.1)
    e_same = float(np.abs(a - same).mean() / np.abs(a).mean())
    e_old = float(np.abs(a - old).mean() / np.abs(a).mean())
    print(f"  same brain at 1 mm and at 0.8 mm: cubes differ by {e_same:.1%} (cubes in voxels, as before 8.1: {e_old:.0%})")
    if e_same > 0.03 or e_old < 0.2:
        problems.append(f"cubes in mm: 1 mm vs 0.8 mm differ by {e_same:.1%} (voxel cubes {e_old:.0%})")
    if problems:
        fail("mirror geometry: " + "; ".join(problems))
    print("Mirror geometry checks passed.")


def check_patches(env):
    z = np.load(os.path.join(HERE, env["EZ_OUT_DIR"], "patches.npz"))
    if list(z["channels"]) != ["T1", "T1-mirror"]:
        fail(f"patches should be T1 + its mirror (a fake patient has no T2 image), not {list(z['channels'])}")
    planes = json.loads(str(z["planes"]))
    sizes = {round(p["voxel_mm"], 2) for p in planes.values()}
    if "patch_mm" not in z.files or float(z["patch_mm"]) != 1.0 or sizes != {0.8, 1.0}:
        fail(f"version 8.1 patches: patch_mm {z['patch_mm'] if 'patch_mm' in z.files else 'missing'}, voxel sizes "
             f"{sorted(sizes)} (expected 1 mm cubes from 0.8 mm and 1 mm MRIs)")
    kind, y, pair, ok = z["kind"].astype(str), z["y"], z["pair"], z["train_ok"]
    lab = pd.read_csv(os.path.join(HERE, env["EZ_OUT_DIR"], "contacts_labeled.csv"))
    outside = lab[~lab["near_brain"]]
    certain = lab["uncertain"] == 0
    if outside.empty or int((kind == "site").sum()) != int((lab["in_bounds"] & lab["near_brain"] & certain).sum()):
        fail("contacts outside the brain or uncertain contacts were not left out of the patches")
    ctrl = np.flatnonzero(kind == "control")
    if not len(ctrl) or (y[ctrl] != 0).any():
        fail("no mirror controls, or a mirror control is not labelled 'not SOZ'")
    if any(pair[i] < 0 or z["subject"][pair[i]] != z["subject"][i] or z["contact"][pair[i]] != z["contact"][i]
           or y[pair[i]] != 1 for i in ctrl):
        fail("a mirror control does not point to its SOZ site")
    if not (ok[ctrl].any() and (~ok[ctrl]).any()):
        fail("mirror controls: expected some for training and some only for the test (fake sub-PID09's clinical table "
             "gives the other side)")
    if ok[ctrl][z["subject"][ctrl] == "sub-PID09"].any():
        fail("sub-PID09: the clinical side disagrees with the SOZ side - its mirror controls must not be used for "
             "training")
    X = z["X"]
    i = int(ctrl[0])
    diff = float(np.abs(X[pair[i], 1].astype(np.float32)[::-1] - X[i, 0].astype(np.float32)).mean())
    if diff > 0.05:   # the fake brain is symmetric about a straight plane: the site's mirror = the control's cube
        fail(f"the control's cube is not the mirror of its SOZ site's cube (difference {diff:.3f})")
    if certain.all():
        fail("no uncertain contact reached the model steps (fake sub-PID01 has some)")
    deep = np.flatnonzero(kind == "deep")
    if not len(deep) or (y[deep] != 0).any() or not ok[deep].all() or (pair[deep] != -1).any():
        fail("version 9: no deep-structure cubes, or they are not 'not SOZ' training examples")
    import common
    if not set(z["region_id"][deep].tolist()) <= set(common.DEEP_IDS.tolist()):
        fail("a deep-structure cube is not in a deep structure")
    vox = {sub: p["voxel_mm"] for sub, p in json.loads(str(z["planes"])).items()}
    hot = lab[(lab["target"] == 1) | (lab["uncertain"] == 1)]
    for i in deep:
        h = hot[hot["subject"] == z["subject"][i]][["i", "j", "k"]].to_numpy(float)
        if len(h) and np.sqrt(((h - z["ijk"][i]) ** 2).sum(axis=1)).min() * vox[str(z["subject"][i])] < 10.0 - 1e-6:
            fail(f"deep-structure cube {z['contact'][i]} of {z['subject'][i]} is closer than 10 mm to a SOZ contact")
    rt = pd.read_csv(os.path.join(HERE, env["EZ_OUT_DIR"], "region_table.csv"))
    if ((rt["n_uncertain"] > 0) & (rt["target"] == 0)).any():
        fail("a region with an uncertain site was labelled 'not SOZ'")


def check_cnn_outputs(env):
    out = os.path.join(HERE, env["EZ_OUT_DIR"])
    card = json.load(open(os.path.join(out, "cnn_model.json")))
    if card["version"] != "9" or card.get("patch_mm") != 1.0 or not card.get("final") or not card["final"]["models"]:
        fail("05_train_cnn.py --final did not write the final models (and patch_mm) into cnn_model.json")
    deep = pd.read_csv(os.path.join(out, "cnn_predictions_deep.csv"))
    if deep.empty or deep["p_cnn"].isna().any():
        fail("cnn_predictions_deep.csv: every deep-structure cube needs its out-of-fold probability")
    for m in [f for fold in card["folds"] for f in fold["models"]] + card["final"]["models"]:
        if not os.path.exists(os.path.join(out, m)):
            fail(f"model file {m} is missing")
    ctrl = pd.read_csv(os.path.join(out, "cnn_predictions_controls.csv"))
    if ctrl.empty or ctrl[["p_site", "p_control"]].isna().any().any():
        fail("cnn_predictions_controls.csv: every mirror control needs its probability and its SOZ site's")
    sites = pd.read_csv(os.path.join(out, "cnn_predictions_contacts.csv"))
    regions = pd.read_csv(os.path.join(out, "cnn_predictions_regions.csv"))
    r = regions.iloc[0]
    mean = sites[(sites["subject"] == r["subject"]) & (sites["region_id"] == r["region_id"])]["p_cnn"].mean()
    if not np.isclose(mean, r["p_cnn"]):
        fail("region probabilities must be the MEAN of their sites")
    print(f"CNN checks passed: {len(card['folds'])} folds + final models, {len(ctrl)} left-right pairs predicted.")


def check_side_model(env):
    """Version 9: 07_lateralization.py - the side model on 5 measures, tested leave-one-out, shown separately for the
    patients with hippocampal sclerosis on the MRI report, saved for 08 (side_model.json)."""
    out = os.path.join(HERE, env["EZ_OUT_DIR"])
    met = pd.read_csv(os.path.join(out, "lateralization_metrics.csv"))
    pred = pd.read_csv(os.path.join(out, "lateralization_predictions.csv")).set_index("subject")
    card = json.load(open(os.path.join(out, "side_model.json")))
    if "side_model" not in set(met["model"]) or not met["model"].str.contains("hippocampal sclerosis").any():
        fail("07_lateralization.py: no side_model results (all patients and the hippocampal-sclerosis group)")
    if len(card["features"]) != 5 or pred["p_left_side_model"].isna().any() or len(pred) != 12:
        fail("07_lateralization.py: the side model must give p(left) for every patient with an MRI (12 fakes)")
    if not np.isfinite(pred["side_asym_thickness_temporal"]).all() or np.isfinite(pred.loc["sub-PID04",
                                                                                             "side_asym_t2_Hippocampus"]):
        fail("side features: the surface thickness must be read for every fake patient, and fake sub-PID04 has no T2")
    print(f"Side model checks passed: {len(pred)} patients, measures {', '.join(card['features'])}.")


def check_final_model(env):
    """08_predict_mri.py on a training patient (its own fold's models), on a new FreeSurfer folder and --unseen."""
    import nibabel as nib
    out = os.path.join(HERE, env["EZ_OUT_DIR"], "predictions")
    run(["08_predict_mri.py", "--subject", "sub-PID01", "--step", "8"], env)
    d = os.path.join(out, "sub-PID01")
    for f in ("ez_probability.nii.gz", "ez_zone.nii.gz", "ez_probability.png", "regions.csv", "summary.txt"):
        if not os.path.exists(os.path.join(d, f"sub-PID01_{f}")):
            fail(f"08_predict_mri.py did not write sub-PID01_{f}")
    img = nib.load(os.path.join(d, "sub-PID01_ez_probability.nii.gz"))
    p = np.asarray(img.dataobj)
    mri = nib.load(os.path.join(HERE, "synthetic_data", "files", "derivatives", "sub-PID01", "freesurfer3T", "sub-PID01",
                                "mri", "norm.mgz"))
    if p.shape != mri.shape or not np.allclose(img.affine, mri.affine, atol=1e-3) or p.min() < 0 or p.max() > 1 \
            or not (p > 0).any():
        fail("the probability map is not in the MRI's space or not a probability")
    import common
    lab01 = np.asarray(nib.load(os.path.join(os.path.dirname(mri.get_filename()), "aparc_aseg.mgz")).dataobj)
    if (p[np.isin(lab01, common.DEEP_IDS)] != 0).any() or not (p[np.isin(lab01, common.ASSESSED_IDS)] > 0).all():
        fail("version 9: the map must cover every assessed voxel and leave the deep structures 'not assessed'")
    summary = open(os.path.join(d, "sub-PID01_summary.txt"), encoding="utf-8").read()
    if "never saw this patient" not in summary:
        fail("a training patient must be shown with the models of his own fold (never saw him)")
    if "side (side model" not in summary or "leave-one-out" not in summary:
        fail("version 9: the summary must give the side model's leave-one-out answer")
    fs = os.path.join("synthetic_data", "files", "derivatives", "sub-PID11", "freesurfer3T", "sub-PID11")
    run(["08_predict_mri.py", "--fs-dir", fs, "--name", "NEW11", "--step", "10"], env)
    new_map = os.path.join(out, "NEW11", "NEW11_ez_probability.nii.gz")
    if not os.path.exists(new_map):
        fail("08_predict_mri.py --fs-dir did not write the map")
    img11, mri11 = nib.load(new_map), nib.load(os.path.join(HERE, fs, "mri", "norm.mgz"))
    text11 = open(os.path.join(out, "NEW11", "NEW11_summary.txt"), encoding="utf-8").read()
    if (img11.shape != mri11.shape or not np.allclose(img11.affine, mri11.affine, atol=1e-3)
            or "voxel size 0.80 mm" not in text11 or "every 9.6 mm" not in text11):
        fail("08_predict_mri.py on a 0.8 mm MRI: the map must be in that MRI's space, the grid in mm (10 mm -> 12 voxels "
             "= 9.6 mm)")
    run(["08_predict_mri.py", "--unseen", "--step", "10", "--no-maps"], env)
    val = pd.read_csv(os.path.join(out, "unseen_validation.csv"))
    card = json.load(open(os.path.join(HERE, env["EZ_OUT_DIR"], "cnn_model.json")))
    if val.empty or set(val["subject"]) & set(card["training_subjects"]):
        fail("08_predict_mri.py --unseen must test only patients the model was not trained on")
    text = open(os.path.join(out, "unseen_validation.txt"), encoding="utf-8").read()
    if ("FAIR SCORE" not in text or "Sclerosis check: in" not in text or "side model (leave-one-out)" not in text
            or set(val.loc[val["hippocampal_sclerosis_side"].notna(), "subject"]) != {"sub-PID11", "sub-PID12"}):
        fail("version 9: --unseen must give the fair side score, the side model and the sclerosis check "
             "(fake sub-PID11 and sub-PID12 have hippocampal sclerosis)")
    print(f"Final-model checks passed: map in the MRI space, honest fold models for a training patient, "
          f"{len(val)} unseen patients checked against the clinical record.")


def check_dataset_extras(env):
    """tools/check_dataset_extras.py must see what the fake dataset holds: 0.8 mm and 1 mm MRIs, PID04's raw T2w (it has
    no FreeSurfer T2.mgz), PID11's FLAIR, the 7T hippocampal T2w of PID01 and PID11, the brain-shift files."""
    run(["tools/check_dataset_extras.py"], env)
    out = os.path.join(HERE, env["EZ_OUT_DIR"])
    df = pd.read_csv(os.path.join(out, "dataset_extras.csv")).set_index("subject")
    text = open(os.path.join(out, "dataset_extras.txt"), encoding="utf-8").read()
    sizes = df["fs3T_voxel_mm"].astype(float)
    problems = []
    if sorted(sizes[sizes == 0.8].index) != ["sub-PID07", "sub-PID08", "sub-PID09", "sub-PID11", "sub-PID12"] \
            or (sizes[sizes != 0.8] != 1.0).any():
        problems.append(f"voxel sizes read wrongly: {sizes.to_dict()}")
    if list(df.index[df["t2_could_be_added"]]) != ["sub-PID04"] or df["no_t2_at_all"].any():
        problems.append("T2: only sub-PID04 has a raw T2w without a FreeSurfer T2.mgz")
    if list(df.index[df["has_raw_3T_FLAIR"]]) != ["sub-PID11"] or \
            list(df.index[df["has_raw_7T_T2w"]]) != ["sub-PID01", "sub-PID11"]:
        problems.append("FLAIR / 7T T2w files not found where they are")
    if df.loc[["sub-PID01", "sub-PID02", "sub-PID03"], "recon_brainshift"].isna().any():
        problems.append("brain-shift files of the older iEEG-recon layout not listed")
    if "metadata_3T_7T.xlsx" not in text or "Seizure Onset Lateralization" not in text:
        problems.append("the top-level clinical table and its columns are not listed")
    if problems:
        fail("check_dataset_extras.py: " + "; ".join(problems))
    print("check_dataset_extras.py works: voxel sizes, T2 / FLAIR / 7T scans, brain-shift files and top-level tables found.")


def main():
    env = dict(os.environ,
               EZ_DATA_SOURCE="local",
               EZ_DATA_ROOT=os.path.join("synthetic_data", "files"),
               EZ_CACHE_DIR=os.path.join("synthetic_data", "cache"),       # never the real data folder
               EZ_BENCHMARK_DIR=os.path.join("synthetic_data", "benchmark"),
               EZ_OUT_DIR="outputs_synthetic",
               EZ_SOZ_RESULTS_DIR=os.path.join("outputs_synthetic", "SOZ"),
               EZ_ONSET_METHOD="benchmark",
               EZ_EXCLUDE_FROM_TRAINING="sub-PID06")   # instead of the real-data list in config.py (tests the setting)
    check_onset_detector()
    check_mirror_geometry()
    version_file = os.path.join(HERE, "synthetic_data", ".version")
    current = open(version_file).read().strip() if os.path.exists(version_file) else ""
    if current != DATA_VERSION:
        for d in ("synthetic_data", "outputs_synthetic"):
            shutil.rmtree(os.path.join(HERE, d), ignore_errors=True)
        run(["tools/make_synthetic_data.py", "--out", "synthetic_data", "--n", "12"], env)
        run(["tools/make_synthetic_benchmark.py", "--out", "synthetic_data"], env)
    else:
        shutil.rmtree(os.path.join(HERE, "outputs_synthetic"), ignore_errors=True)
        shutil.rmtree(os.path.join(HERE, "synthetic_data", "cache"), ignore_errors=True)

    run_benchmark(env)
    run(["SOZ/01_scan_dataset.py"], env)
    run(["SOZ/02_compute_soz.py"], env)
    run(["SOZ/03_build_contacts.py"], env)
    check_soz_labels(env)
    check_unlocalized_tool(env)
    run(["01_label_contacts.py"], env)
    run(["02_region_features.py"], env)
    run(["03_train_baseline.py"], env)
    run(["04_make_patches.py"], env)
    check_patches(env)
    run(["05_train_cnn.py", "--epochs", "3", "--patience", "3", "--folds", "3", "--seeds", "1", "--final"], env)
    check_cnn_outputs(env)
    run(["06_make_heatmap.py", "--subject", "sub-PID01"], env)
    run(["06_make_heatmap.py", "--subject", "sub-PID01", "--pred",
         os.path.join("outputs_synthetic", "cnn_predictions_regions.csv"), "--col", "p_cnn"], env)
    run(["07_lateralization.py"], env)
    check_side_model(env)
    check_final_model(env)
    run(["tools/model_report.py"], env)
    report = open(os.path.join(HERE, env["EZ_OUT_DIR"], "model_report.txt"), encoding="utf-8").read()
    if "LEFT-RIGHT TEST" not in report or not os.path.exists(os.path.join(HERE, env["EZ_OUT_DIR"], "for_claude",
                                                                            "model_results.zip")):
        fail("tools/model_report.py did not write the left-right test or the zip")
    if "5. DEEP STRUCTURES" not in report or "6. SHORTCUT CHECK" not in report or "side_model" not in report:
        fail("version 9: the report must show the side model, the deep-structure check and the shortcut check")
    run(["tools/v7_car_linelength.py", "--no-pictures"], env)
    check_dataset_extras(env)
    run(["tools/plot_seizure.py", "--subject", "sub-PID06"], env)
    plots = [f for f in os.listdir(os.path.join(HERE, env["EZ_OUT_DIR"], "seizure_plots")) if f.startswith("sub-PID06")]
    if len(plots) != 2:
        fail("tools/plot_seizure.py did not draw the 2 seizures of the fake sub-PID06")
    run(["tools/plot_seizure.py", "--audit"], env)
    audit = os.listdir(os.path.join(HERE, env["EZ_OUT_DIR"], "seizure_plots", "audit"))
    n_lab = pd.read_csv(os.path.join(HERE, env["EZ_SOZ_RESULTS_DIR"], "soz_summary.csv"))["subject"].nunique()
    if len(audit) != n_lab:
        fail(f"tools/plot_seizure.py --audit drew {len(audit)} pictures for {n_lab} patients")
    logs = os.listdir(os.path.join(HERE, env["EZ_OUT_DIR"], "logs"))
    if not {"02_compute_soz.txt", "benchmark_onset.txt", "03_build_contacts.txt"} <= set(logs):
        fail(f"log files missing in outputs_synthetic/logs ({', '.join(sorted(logs))})")
    print(f"\nSMOKE TEST PASSED (version {VERSION}) - every step ran and every check passed. Next: the real data "
          "(README.md).")


if __name__ == "__main__":
    main()
