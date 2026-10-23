"""
SOZ/02_compute_soz.py  -  STEP 2: find the seizure onset zone (SOZ) from the seizure recordings (version 6.1).

Run from the project folder (after step 1 and the expert benchmark):
    python SOZ\\02_compute_soz.py                          all usable patients (~2.2 GB, one file at a time)
    python SOZ\\02_compute_soz.py --subjects sub-PID02     only some patients
    python SOZ\\02_compute_soz.py --aggregate-only         re-make the labels from the saved results, NO download
    python SOZ\\02_compute_soz.py --subjects sub-PID02 --redo
                                                           analyse these patients again (e.g. after a name fix)

The onset METHOD is the one that agreed best with the epileptologists in tools\\benchmark_onset.py
(SOZ/results/benchmark_choice.json; on the 40 expert seizures: "EI + line length"); this step stops if the
benchmark has not been run.

For every seizure recording of every usable patient it:
  1. downloads the recording from AWS (kept in CACHE_DIR when KEEP_EDF = True),
  2. picks the intracranial contacts that have electrode coordinates (scalp EEG and ECG channels are ignored),
  3. saves, for every contact (common average) and every pair of neighbouring contacts (bipolar), the 8 onset
     features and the EI energy ratio over time (results/seizure_arrays/) - an interrupted run continues where it
     stopped, and labels can be re-made later without downloading again,
  4. finds where the seizure starts. With "EI + line length": the onset sites of the EI and the onset sites of
     line length together. Each detector leaves out sites that were already active before the onset, and is not
     used for a seizure in which many sites were already active (a state change, e.g. slow waves fading) or -
     line length - too many sites start together. A seizure is used when at least one detector gives onset sites.
Then, per patient (rule "repeated"):
    SOZ site       = an onset site in at least SOZ_MIN_ONSETS (2) usable seizures AND in at least SOZ_MIN_SHARE (25%)
                     of them - several foci all count
    uncertain site = an onset site less often than that; or a site already active before the onset in some seizure.
                     Uncertain sites are used neither as SOZ nor as "not SOZ".
A patient is trusted only if it has >= 2 usable seizures, a SOZ, and the SOZ (or a site right next to it) is among the
onset sites in at least SOZ_MIN_LED (30%) of the usable seizures.
At the end the labels are compared with version 5 (EI, multifocal rule).

Results (in SOZ/results/):
  seizure_arrays/            one .npz per seizure; *.error.txt = skipped recordings (with the reason)
  soz_onset_per_seizure.csv  start time, score, onset site yes/no of every site in every seizure (and the EI and
                             line-length parts, and the version 5 EI)
  seizure_info.csv           one row per seizure: usable or why not, earliest start, onset sites
  soz_labels.csv             one row per site: onsets, SOZ yes/no, uncertain (and why)
  soz_summary.csv            one row per patient: usable seizures, SOZ sites, label quality, used for training
  soz_labels_ei_v5.csv, soz_summary_ei_v5.csv   the version 5 EI labels, for comparison
  step2_report.txt           the summary printed at the end  <- send this to Claude
"""
import argparse
import json
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import aws_data  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402

MONTAGES = ("bipolar", "average")
ARR_DIR_NAME = "seizure_arrays"


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def aliases_for(sub):
    return {soz_lib.normalize_name(k): soz_lib.normalize_name(v)
            for k, v in (config.CHANNEL_ALIASES or {}).get(sub, {}).items()}


def load_coordinates(row):
    """Names of the contacts that have electrode coordinates, and {name: xyz in mm} (used to pair
    neighbouring contacts)."""
    names, coords = set(), {}
    try:
        names_p = aws_data.fetch(row.recon_names, missing_ok=True) if row.recon_names else None
        mm_p = aws_data.fetch(row.recon_coords_mm, missing_ok=True) if row.recon_coords_mm else None
        vox_p = aws_data.fetch(row.recon_coords_vox, missing_ok=True) if row.recon_coords_vox else None
        if names_p:
            names |= {soz_lib.normalize_name(n) for n in soz_lib.read_names(names_p)}
        if mm_p:
            df, _ = soz_lib.electrode_coordinates(names_p, mm_p, None, None)
            coords = {n: xyz for n, xyz in zip(df["contact_norm"], df[["x", "y", "z"]].to_numpy())}
            names |= set(coords)
        elif vox_p:
            names |= {soz_lib.normalize_name(r[0]) for r in soz_lib.read_coordinate_file(vox_p) if r[0]}
    except Exception as e:  # noqa: BLE001
        print(f"    (could not read the electrode coordinates: {e})")
    return names, coords


# --------------------------------------------------------------------------- which method
METHODS_ALL = ("ei+linelength", "ei", "linelength", "multifeature")


def load_choice(override=None):
    """The onset method and its settings: the expert benchmark's choice (normal), or a method given for tests."""
    method = override or config.ONSET_METHOD
    if method and method != "benchmark":
        if method not in METHODS_ALL:
            raise SystemExit(f"Unknown onset method {method!r} ({', '.join(METHODS_ALL)} or benchmark)")
        params = soz_lib.method_params(method, diffuse_share=config.DIFFUSE_SHARE)
        return {"method": method, "montage": "bipolar", "params": params,
                "source": f"'{method}' with default settings (NOT chosen by the expert benchmark - for tests only)"}
    path = os.path.join(config.SOZ_RESULTS_DIR, "benchmark_choice.json")
    if not os.path.exists(path):
        raise SystemExit("The onset method has not been chosen yet. Run the expert benchmark first:\n"
                         "    python tools\\get_benchmark.py\n    python tools\\benchmark_onset.py\n"
                         "then this step again.")
    with open(path) as fh:
        ch = json.load(fh)
    if ch["method"] not in METHODS_ALL:
        raise SystemExit(f"{path} names an unknown method {ch['method']!r} - run tools\\benchmark_onset.py again")
    if str(ch.get("version", "6")) != "6.1":
        print("NOTE: benchmark_choice.json was made by version 6 - run  python tools\\benchmark_onset.py  again "
              "(about 1 minute, no download) so that the choice uses the version 6.1 methods.")
    ch["params"] = soz_lib.method_params(ch["method"], ch["params"], diffuse_share=config.DIFFUSE_SHARE)
    ch["source"] = (f"chosen by the expert benchmark ({ch.get('date', '?')}): test F1 {ch.get('test_f1', float('nan')):.2f}"
                    f" vs EI {ch.get('ei_test_f1', float('nan')):.2f}")
    return ch


def describe(choice):
    return soz_lib.describe_method(choice["params"], choice["montage"])


# --------------------------------------------------------------------------- stage 1: recordings -> arrays
def analyse_patient(inv_row, recs, arr_dir, max_seizures, choice):
    sub = inv_row.subject
    coord_names, coords = load_coordinates(inv_row)
    aliases = aliases_for(sub)
    recs = recs.head(max_seizures) if max_seizures else recs
    print(f"\n{sub}: {len(recs)} seizure recordings, {len(coord_names)} contacts with coordinates"
          + (f", name fixes {aliases}" if aliases else ""))
    for i, r in enumerate(recs.itertuples(), start=1):
        tag = f"{sub}__{safe_name(r.task)}_run-{r.run}"
        done = os.path.join(arr_dir, tag + ".npz")
        failed = os.path.join(arr_dir, tag + ".error.txt")
        if os.path.exists(done):
            print(f"  [{i}/{len(recs)}] {r.task} run-{r.run}: already done")
            continue
        t0 = time.time()
        try:
            edf = aws_data.fetch(r.edf)
            channels = soz_lib.read_tsv(aws_data.fetch(r.channels)) if r.channels else None
            events = soz_lib.read_tsv(aws_data.fetch(r.events)) if r.events else None
            sidecar = soz_lib.read_json(aws_data.fetch(r.json)) if r.json else {}
            arr = soz_lib.analyse_recording(edf, channels, events, sidecar, coord_names, coords, aliases,
                                            config.EI_PRE_S, config.EI_POST_S)
            soz_lib.save_arrays(done, arr)
            if os.path.exists(failed):
                os.remove(failed)
            print(f"  [{i}/{len(recs)}] {r.task} run-{r.run}: onset {arr['onset']:.1f}s, "
                  f"{len(arr['avg_names'])} contacts / {len(arr['bip_names'])} pairs, "
                  f"{seizure_text(soz_lib.load_arrays(done), choice)}  [{time.time() - t0:.0f}s]")
        except Exception as e:  # noqa: BLE001 - one bad recording must not stop the others
            with open(failed, "w", encoding="utf-8") as fh:
                fh.write(f"{type(e).__name__}: {e}\n")
            print(f"  [{i}/{len(recs)}] {r.task} run-{r.run}: SKIPPED - {type(e).__name__}: {e}")
        finally:
            if not config.KEEP_EDF:
                aws_data.remove_downloaded(r.edf)


def seizure_tables(arr, choice):
    """(sites table, seizure table) of one saved seizure for the chosen method (both montages) + the version 5 EI
    columns (for comparison)."""
    ei = soz_lib.ei_tables(arr, config.EI_PRE_S, config.EI_POST_S, config.EI_LEAD_S)
    sites, seiz = soz_lib.onset_tables(arr, choice["params"])
    sites = sites.merge(ei[["montage", "contact_norm", "ei", "detect_s"]].rename(
        columns={"ei": "ei_v5", "detect_s": "ei_v5_start_s"}), on=["montage", "contact_norm"], how="left")
    return sites, seiz


def seizure_text(arr, choice):
    sites, seiz = seizure_tables(arr, choice)
    m = choice["montage"]
    s = seiz[seiz["montage"] == m].iloc[0]
    on = sites[(sites["montage"] == m) & sites["onset_site"]].sort_values("start_s")
    shown = ", ".join(f"{c}({t:+.1f}s)" for c, t in zip(on["contact"].head(5), on["start_s"].head(5)))
    if s["flags"]:
        return f"NOT USABLE: {s['flags']}" + (f" (onset sites would be {shown})" if shown else "")
    return f"onset sites: {shown or 'none'}" + (f" (+{len(on) - 5} more)" if len(on) > 5 else "")


# --------------------------------------------------------------------------- stage 2: arrays -> tables
def all_tables(arr_dir, subjects, choice):
    frames, info, meta = [], [], []
    for f in sorted(os.listdir(arr_dir)):
        if not f.endswith(".npz") or f.endswith(".tmp.npz") or "__" not in f:
            continue
        sub, seizure = f[:-4].split("__", 1)
        if sub not in subjects:
            continue
        arr = soz_lib.load_arrays(os.path.join(arr_dir, f))
        if int(arr.get("version", 0)) != soz_lib.VERSION:
            raise SystemExit(f"{f} was made by an older version of this step - run it again with --redo "
                             f"(or delete {arr_dir}).")
        sites, seiz = seizure_tables(arr, choice)
        sites.insert(0, "subject", sub)
        sites.insert(1, "seizure", seizure)
        frames.append(sites)
        row = {"subject": sub, "seizure": seizure}
        row.update({k: arr.get(k) for k in ("onset", "onset_source", "fs", "duration_s", "n_edf_channels",
                                             "channel_selection", "n_scalp_ignored")})
        for _, s in seiz.iterrows():
            for k in ("flags", "usable", "t0_s", "n_onset", "n_started", "n_analysed"):
                row[f"{s['montage']}_{k}"] = s[k]
        info.append(row)
        meta.append({"subject": sub, "seizure": seizure, "analysed": set(arr["avg_norm"]),
                     "unmatched": set(filter(None, str(arr.get("unmatched_edf", "")).split(";")))})
    per = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return per, pd.DataFrame(info), meta


# --------------------------------------------------------------------------- stage 3: labels
def site_contacts(norm):
    return str(norm).split("-")


def patient_labels(ps, usable_seizures, choice, ei_rows=None):
    """SOZ / uncertain of every site of ONE patient (one montage).
    ps: per-seizure site rows (columns seizure, contact, contact_norm, bad_channel, oi, onset_site, edge, start_s);
    usable_seizures: the seizures that can be used. ei_rows: the same patient's EI rows (second opinion).
    Returns (labels table, share of usable seizures started at the SOZ)."""
    p = ps[~ps["bad_channel"].astype(bool)].copy()
    strong = 0.5
    q = p[p["seizure"].isin(usable_seizures)].copy()
    # soz_sites works on a score column called 'ei'; onset sites = score >= strong (any method)
    q["ei"] = np.where(q["onset_site"].astype(bool), np.maximum(q["oi"].fillna(0), strong),
                       np.minimum(q["oi"].fillna(0), strong * 0.999))
    q["detect_s"] = q["start_s"]
    n_used = int(q["seizure"].nunique())
    need = max(config.SOZ_MIN_ONSETS, int(np.ceil(config.SOZ_MIN_SHARE * n_used - 1e-9)))
    if len(q):
        g, started = soz_lib.soz_sites(q, rule="repeated", min_onsets=need, strong_ei=strong)
        g = g.reset_index()
    else:
        g, started = pd.DataFrame(columns=["contact_norm", "contact", "n_seizures", "n_strong", "uncertain",
                                           "soz_ei", "ei_mean", "ei_max", "detect_s_median"]), np.nan
    # every site of the patient appears, also when no usable seizure saw it
    allsites = p.drop_duplicates("contact_norm")[["contact_norm", "contact"]]
    g = allsites.merge(g.drop(columns=["contact"], errors="ignore"), on="contact_norm", how="left")
    g["n_seizures"] = g["n_seizures"].fillna(0).astype(int)
    g["n_onset"] = g.pop("n_strong").fillna(0).astype(int) if "n_strong" in g.columns else 0
    g["soz"] = g.pop("soz_ei").fillna(0).astype(int)
    g["uncertain"] = g["uncertain"].fillna(0).astype(int)
    need_here = min(need, max(n_used, 1))
    g["uncertain_reason"] = [f"onset site in {k} of {n_used} usable seizures (SOZ needs {need_here})" if u == 1 else ""
                             for k, u in zip(g["n_onset"], g["uncertain"])]
    # sites already active before the onset (in any seizure): not SOZ evidence, but not 'not SOZ' either
    act = p[p["edge"].astype(bool)].groupby("contact_norm")["seizure"].nunique()
    m = g["contact_norm"].isin(act.index) & (g["soz"] == 0) & (g["uncertain"] == 0)
    g.loc[m, "uncertain"] = 1
    g.loc[m, "uncertain_reason"] = [f"already active before the onset ({act[c]} seizure{'s' if act[c] > 1 else ''})"
                                    for c in g.loc[m, "contact_norm"]]
    g["ei_v5_soz"] = 0
    if ei_rows is not None and len(ei_rows):
        e = ei_rows[~ei_rows["bad_channel"].astype(bool)].copy()
        det = e.groupby("seizure")["ei_v5_start_s"].apply(lambda v: np.isfinite(v).any())
        e = e[e["seizure"].isin(det[det].index)]
        if len(e):
            ge, _ = soz_lib.soz_sites(e.drop(columns=["detect_s"], errors="ignore").rename(
                                          columns={"ei_v5": "ei", "ei_v5_start_s": "detect_s"}),
                                      rule="multifocal", onset_ei=config.EI_THRESHOLD, mean_ei=config.SOZ_MEAN_EI,
                                      min_onsets=config.SOZ_MIN_ONSETS, strong_ei=config.SOZ_STRONG_EI)
            v5 = set(ge.index[ge["soz_ei"] == 1])
            g["ei_v5_soz"] = g["contact_norm"].isin(v5).astype(int)
            if config.EI_AS_SECOND_OPINION and choice["method"] not in ("ei", "ei+linelength"):
                m = (g["ei_v5_soz"] == 1) & (g["soz"] == 0) & (g["uncertain"] == 0)
                g.loc[m, "uncertain"] = 1
                g.loc[m, "uncertain_reason"] = "SOZ for the version 5 EI only"
    g = g.rename(columns={"ei_mean": "oi_mean", "ei_median": "oi_median", "ei_max": "oi_max",
                          "detect_s_median": "start_s_median"})
    return g, started


def recorded_sites(sub, recs, montage):
    """Intracranial, not-bad sites listed in a patient's channel tables (patients with clinician SOZ marks)."""
    out = {}
    for rel in recs.loc[(recs["subject"] == sub) & (recs["kind"] == "ictal"), "channels"]:
        p = aws_data.local_path(rel) if rel else ""
        if p and os.path.exists(p):
            ch = soz_lib.read_tsv(p)
            idx, _, diag = soz_lib.select_channels(list(ch["name"]), ch, None, aliases_for(sub))
            names = [soz_lib.display_name(ch["name"].iloc[i]) for i in idx]
            norms = diag["norms"]
            if montage == "average":
                out.update({n: nm for n, nm in zip(norms, names) if n not in out})
            else:
                for i, j in soz_lib.bipolar_pairs(norms):
                    out.setdefault(f"{norms[i]}-{norms[j]}", f"{names[i]}-{names[j]}")
    return out


def aggregate(per, info, choice, recs, out_dir, subjects):
    """Labels of every site and one summary row per patient, for the chosen montage."""
    montage = choice["montage"]
    pm = per[per["montage"] == montage]
    usable_col = f"{montage}_usable"
    frames, rows_started = [], {}
    for sub in subjects:
        ps = pm[pm["subject"] == sub]
        if ps.empty:
            continue
        inf = info[info["subject"] == sub]
        usable = list(inf.loc[inf[usable_col].astype(bool), "seizure"])
        g, st = patient_labels(ps, usable, choice, ei_rows=ps)
        g.insert(0, "subject", sub)
        frames.append(g)
        rows_started[sub] = st
    lab = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    marks_path = os.path.join(out_dir, "clinical_soz_marks.csv")
    marks = pd.read_csv(marks_path) if os.path.exists(marks_path) else pd.DataFrame(columns=["subject", "contact_norm"])
    marked = set(zip(marks["subject"], marks["contact_norm"].astype(str)))
    has_marks = set(marks["subject"])
    src = config.SOZ_LABEL_SOURCE
    if src not in ("auto", "computed", "clinical"):
        raise SystemExit("config.py: SOZ_LABEL_SOURCE must be 'auto', 'computed' or 'clinical'")
    if src in ("auto", "clinical") and has_marks:
        extra = []
        for sub in sorted(has_marks & set(subjects)):
            have = set(lab.loc[lab["subject"] == sub, "contact_norm"]) if len(lab) else set()
            for norm, name in recorded_sites(sub, recs, montage).items():
                if norm not in have:
                    extra.append({"subject": sub, "contact_norm": norm, "contact": name, "n_seizures": 0,
                                  "n_onset": 0, "soz": 0, "uncertain": 0, "uncertain_reason": "", "ei_v5_soz": 0})
        if extra:
            lab = pd.concat([lab, pd.DataFrame(extra)], ignore_index=True)
    if src == "clinical" and not has_marks:
        raise SystemExit("SOZ_LABEL_SOURCE = 'clinical' but the dataset has no clinician SOZ marks (see step 1).")
    lab["soz_clinical"] = [int(any((s, c) in marked for c in site_contacts(n)))
                           for s, n in zip(lab["subject"], lab["contact_norm"])]
    use_clin = lab["subject"].isin(has_marks) if src in ("auto", "clinical") else pd.Series(False, index=lab.index)
    if src == "clinical":
        lab = lab[use_clin].copy()
        use_clin = pd.Series(True, index=lab.index)
    text = describe(choice)
    lab["soz_computed"] = lab["soz"].astype(int)
    lab["soz"] = np.where(use_clin, lab["soz_clinical"], lab["soz_computed"]).astype(int)
    lab["uncertain"] = np.where(use_clin, 0, lab["uncertain"].fillna(0)).astype(int)
    lab["uncertain_reason"] = np.where(use_clin, "", lab["uncertain_reason"].fillna(""))
    lab["label_source"] = np.where(use_clin, "clinician marks", text)
    lab["montage"] = montage
    lab["hemisphere"] = lab["contact"].map(soz_lib.hemisphere_from_name)
    lab = lab.sort_values(["subject", "soz", "n_onset"], ascending=[True, False, False]).reset_index(drop=True)

    rows = []
    for sub in subjects:
        g = lab[lab["subject"] == sub]
        inf = info[info["subject"] == sub]
        s = g[g["soz"] == 1]
        n_rec = int(((recs["subject"] == sub) & (recs["kind"] == "ictal")).sum())
        n_use = int(inf[usable_col].astype(bool).sum()) if len(inf) else 0
        flags = inf.loc[~inf[usable_col].astype(bool), f"{montage}_flags"].fillna("").astype(str) if len(inf) else []
        reasons = pd.Series([f.split(";")[0].split(" (")[0] for f in flags if f]).value_counts()
        st = rows_started.get(sub, np.nan) if len(s) else np.nan
        clin = sub in has_marks and src in ("auto", "clinical")
        quality = "clinician marks" if clin else soz_lib.label_quality(n_use, int((g["soz_computed"] == 1).sum()), st,
                                                                       config.SOZ_MIN_LED, "repeated")
        sides = s["hemisphere"].replace("", "?").value_counts()
        v5 = g[g["ei_v5_soz"] == 1]
        both = len(set(s["contact_norm"]) & set(v5["contact_norm"]))
        union = len(set(s["contact_norm"]) | set(v5["contact_norm"]))
        rows.append({
            "subject": sub, "montage": montage, "method": choice["method"], "seizures_recorded": n_rec,
            "seizures_analysed": len(inf), "seizures_usable": n_use,
            "not_usable": "; ".join(f"{k} {v}" for k, v in reasons.items()),
            "sites": len(g), "soz_sites": len(s), "uncertain_sites": int(g["uncertain"].sum()),
            "soz_side": "/".join(f"{k}:{v}" for k, v in sides.items()) or "-",
            "soz_list": ", ".join(s["contact"].head(15)),
            "seizures_started_at_soz": st, "label_quality": quality,
            "use_for_training": bool(clin or quality in ("ok", "few seizures")),
            "label_source": "clinician marks" if clin else text,
            "ei_v5_soz_sites": len(v5), "same_as_ei_v5": (round(both / union, 2) if union else np.nan),
        })
    return lab, pd.DataFrame(rows)


def ei_v5_labels(per, info, recs, out_dir, subjects):
    """The version 5 labels (EI, multifocal rule, search from -10 s), for comparison."""
    choice5 = {"method": "ei", "montage": config.EI_MONTAGE,
               "params": {"lead_s": config.EI_LEAD_S, "threshold": config.SOZ_STRONG_EI}}
    rows, labs = [], []
    pm = per[per["montage"] == config.EI_MONTAGE]
    for sub in subjects:
        ps = pm[pm["subject"] == sub]
        if ps.empty:
            continue
        e = ps[~ps["bad_channel"].astype(bool)]
        det = e.groupby("seizure")["ei_v5_start_s"].apply(lambda v: np.isfinite(v).any())
        inf_s = list(det[det].index)
        q = e[e["seizure"].isin(inf_s)].rename(columns={"ei_v5": "ei", "ei_v5_start_s": "detect_s"})
        if q.empty:
            rows.append({"subject": sub, "seizures_informative": 0, "soz_sites": 0, "label_quality": "no seizure analysed",
                         "use_for_training": False})
            continue
        g, st = soz_lib.soz_sites(q, rule="multifocal", onset_ei=config.EI_THRESHOLD, mean_ei=config.SOZ_MEAN_EI,
                                  min_onsets=config.SOZ_MIN_ONSETS, strong_ei=config.SOZ_STRONG_EI)
        g = g.reset_index()
        g.insert(0, "subject", sub)
        labs.append(g)
        n_soz = int(g["soz_ei"].sum())
        qual = soz_lib.label_quality(len(inf_s), n_soz, st, config.SOZ_MIN_LED, "multifocal")
        rows.append({"subject": sub, "seizures_informative": len(inf_s), "soz_sites": n_soz,
                     "soz_list": ", ".join(g.loc[g["soz_ei"] == 1, "contact"].head(15)),
                     "seizures_started_at_soz": st, "label_quality": qual,
                     "use_for_training": qual in ("ok", "few seizures")})
    _ = choice5
    lab = pd.concat(labs, ignore_index=True) if labs else pd.DataFrame()
    return lab, pd.DataFrame(rows)


def name_report(inv, meta, out_dir):
    rows = []
    for row in inv.itertuples():
        m = [x for x in meta if x["subject"] == row.subject]
        if not m:
            continue
        coord_names, _ = load_coordinates(row)
        analysed = set().union(*[x["analysed"] for x in m])
        unmatched = set().union(*[x["unmatched"] for x in m])
        no_rec = sorted(coord_names - analysed)
        rows.append({"subject": row.subject, "recon_contacts": len(coord_names),
                     "recorded_with_coordinates": len(analysed & coord_names),
                     "recorded_without_coordinates": len(unmatched),
                     "examples_recorded_without_coordinates": " ".join(sorted(unmatched)[:40]),
                     "coordinates_without_recording": len(no_rec),
                     "examples_coordinates_without_recording": " ".join(no_rec[:40])})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(out_dir, "name_matching.csv"), index=False)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--max-seizures", type=int, default=0)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--redo", action="store_true",
                    help="analyse the recordings of the patients given with --subjects again (e.g. after a name fix)")
    ap.add_argument("--method", default=None, choices=list(METHODS_ALL),
                    help="(tests only) use this method with default settings instead of the benchmark's choice")
    args = ap.parse_args()
    if args.redo and not args.subjects:
        raise SystemExit("--redo needs --subjects, e.g.  python SOZ\\02_compute_soz.py --subjects sub-PID02 --redo")
    choice = load_choice(args.method)
    if choice["montage"] not in MONTAGES:
        raise SystemExit(f"benchmark_choice.json: unknown montage {choice['montage']!r}")
    print(f"Onset method: {describe(choice)}\n  ({choice['source']})")

    out = config.SOZ_RESULTS_DIR
    arr_dir = os.path.join(out, ARR_DIR_NAME)
    os.makedirs(arr_dir, exist_ok=True)
    inv_path = os.path.join(out, "dataset_inventory.csv")
    rec_path = os.path.join(out, "seizure_recordings.csv")
    if not (os.path.exists(inv_path) and os.path.exists(rec_path)):
        raise SystemExit("Run step 1 first:  python SOZ\\01_scan_dataset.py")
    inv = pd.read_csv(inv_path, keep_default_na=False)
    inv = inv[inv["usable"].astype(str).str.lower() == "true"]
    recs = pd.read_csv(rec_path, keep_default_na=False, dtype={"run": str})
    todo = inv[inv["subject"].isin(args.subjects)] if args.subjects else inv
    if args.subjects and len(todo) < len(set(args.subjects)):
        raise SystemExit(f"Not usable patients (see dataset_inventory.csv): "
                         f"{', '.join(sorted(set(args.subjects) - set(todo['subject'])))}")
    if args.redo and not args.aggregate_only:
        removed = [f for f in os.listdir(arr_dir) if f.split("__")[0] in set(todo["subject"])]
        for f in removed:
            os.remove(os.path.join(arr_dir, f))
        print(f"--redo: removed {len(removed)} saved results of {', '.join(todo['subject'])}; analysing them again.")

    if not args.aggregate_only:
        total = recs[recs["subject"].isin(todo["subject"])]["edf_bytes"].astype(float).sum() / 1e9
        print(f"{len(todo)} patients, up to {total:.2f} GB of seizure recordings (downloaded once, one at a time).")
        for row in todo.itertuples():
            analyse_patient(row, recs[recs["subject"] == row.subject], arr_dir, args.max_seizures, choice)

    subjects = sorted(inv["subject"])
    per, info, meta = all_tables(arr_dir, set(subjects), choice)
    if per.empty:
        raise SystemExit("No seizure could be analysed - see the *.error.txt files in " + arr_dir)
    subjects = sorted(per["subject"].unique())
    per.to_csv(os.path.join(out, "soz_onset_per_seizure.csv"), index=False)
    info.to_csv(os.path.join(out, "seizure_info.csv"), index=False)
    lab, summary = aggregate(per, info, choice, recs, out, subjects)
    lab.to_csv(os.path.join(out, "soz_labels.csv"), index=False)
    summary.to_csv(os.path.join(out, "soz_summary.csv"), index=False)
    lab5, summ5 = ei_v5_labels(per, info, recs, out, subjects)
    lab5.to_csv(os.path.join(out, "soz_labels_ei_v5.csv"), index=False)
    summ5.to_csv(os.path.join(out, "soz_summary_ei_v5.csv"), index=False)
    names = name_report(inv[inv["subject"].isin(subjects)], meta, out)

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("\n" + "=" * 112)
    say("SOZ LABELS (version 6.1)")
    for k, part in enumerate(textwrap.wrap(f"method: {describe(choice)}", 108)):
        say(("  " if k == 0 else "    ") + part)
    say(f"  {choice['source']}")
    say(f"  SOZ = onset site in at least {config.SOZ_MIN_ONSETS} usable seizures and at least "
        f"{config.SOZ_MIN_SHARE:.0%} of them")
    say("=" * 112)
    show = summary[["subject", "seizures_analysed", "seizures_usable", "sites", "soz_sites", "uncertain_sites",
                    "soz_side", "seizures_started_at_soz", "label_quality", "use_for_training", "ei_v5_soz_sites",
                    "same_as_ei_v5"]].copy()
    show.columns = ["subject", "seizures", "usable", "sites", "SOZ", "uncertain", "side", "started_at_SOZ",
                    "label quality", "train", "v5 SOZ", "same as v5"]
    say(show.to_string(index=False, na_rep="-"))
    say("=" * 112)
    say("usable = seizures that can be used for labels; started_at_SOZ = share of the usable seizures in which the SOZ, "
        "or a site\nright next to it, is an onset site; v5 SOZ = SOZ sites of the version 5 EI; same as v5 = overlap "
        "(1 = the same sites).")
    nu = summary[summary["not_usable"] != ""]
    if len(nu):
        say("\nSeizures that could not be used, and why:")
        for r in nu.itertuples():
            say(f"  {r.subject}: {r.not_usable}")
    say("\nSOZ sites per patient:")
    for r in summary.itertuples():
        say(f"  {r.subject}: {r.soz_list or '-'}")
    errors = [f for f in os.listdir(arr_dir) if f.endswith(".error.txt")]
    if errors:
        say(f"\n{len(errors)} seizure recordings were skipped - reasons in {arr_dir}\\*.error.txt:")
        for f in errors[:20]:
            with open(os.path.join(arr_dir, f), encoding="utf-8") as fh:
                say(f"  {f[:-10]}: {fh.read().strip()[:160]}")
    worst = names[names["recorded_without_coordinates"] > 5] if len(names) else names
    if len(worst):
        say("\nRecorded intracranial channels whose names have no electrode coordinates (see name_matching.csv):")
        for r in worst.itertuples():
            say(f"  {r.subject}: {r.recorded_without_coordinates} channels, e.g. "
                f"{' '.join(r.examples_recorded_without_coordinates.split()[:8])}")
    n_train = int(summary["use_for_training"].sum())
    n5 = int(summ5["use_for_training"].sum()) if len(summ5) else 0
    say(f"\n{int(lab['soz'].sum())} SOZ sites, {int(lab['uncertain'].sum())} uncertain sites; {n_train} of "
        f"{len(summary)} patients have labels good enough for training (version 5 EI: {n5}).")
    say("(Patients in EXCLUDE_FROM_TRAINING in config.py are left out in step 3.)")
    say(f"Saved soz_labels.csv, soz_summary.csv, soz_onset_per_seizure.csv, seizure_info.csv, name_matching.csv, "
        f"soz_*_ei_v5.csv in {out}")
    say("Next: python tools\\check_unlocalized.py --all   then   python SOZ\\03_build_contacts.py")
    with open(os.path.join(out, "step2_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    with runlog.Log("02_compute_soz"):
        main()
