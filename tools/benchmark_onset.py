"""
tools/benchmark_onset.py  -  WHICH onset method agrees best with epileptologists?  (version 6.1)

Run from the project folder, after tools\\get_benchmark.py:
    python tools\\benchmark_onset.py
About 5-15 minutes the first time (the analysed seizures are saved, later runs take ~1 minute).

What it does - like testing a thermometer against a reference thermometer before using it on patients:
  1. Every expert-marked seizure (benchmark/seizures.csv) is analysed exactly like our own seizures in step 2:
     the 63 s around the marked onset, the 30 s before the mark as the normal period, bipolar and common average
     montage.
  2. Each candidate method marks the channels where it thinks the seizure starts, without seeing the answer:
        EI + line length  (version 6.1) an onset site of EITHER the EI or line length; thresholds of both tested
        EI                the Epileptogenicity Index alone (thresholds 0.3 / 0.5 / 0.7, search from -10 or -5 s)
        line length       line length alone (onset_lib, method 'linelength')
        8-feature         line length + energy + 6 frequency bands, Imprint-style (onset_lib, method 'multifeature')
     All of them use the version 6.1 safety rules (a site already active before the onset is not an onset site;
     a seizure with many such sites is not used by that detector; the search ends 10 s after the marked onset).
  3. Score per seizure: F1 between the method's onset channels and the epileptologists' consensus channels
     (1 = the same channels, 0 = no overlap or no answer). A bipolar pair counts as correct when one of its two
     contacts is a consensus channel. Human reference: each epileptologist against the majority of the others.
  4. FAIR TEST, fixed before looking at any result: the patients are split in two halves (alternately by patient
     number). The best setting of every method is chosen on the TUNING half only; all methods are then compared on
     the TEST half, which played no part in choosing. The overall winner is the best setting on the tuning half;
     it is kept only if it is at least as good as the EI on the test half - otherwise the EI is kept.
  5. Also shown: how good the PATIENT labels of step 2 would be with each method (patients with >= 2 seizures: the
     SOZ rule of step 2 against the channels the epileptologists marked).
  6. Writes the choice to SOZ/results/benchmark_choice.json - step 2 (SOZ/02_compute_soz.py) uses it.

Results (SOZ/results/): benchmark_report.txt (send this one to Claude), benchmark_choice.json,
benchmark_scores.csv (every setting), benchmark_per_seizure.csv, benchmark_patients.csv.
"""
import argparse
import datetime
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SOZ"))
sys.path.insert(0, os.path.join(HERE, "tools"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import get_benchmark  # noqa: E402
import onset_lib  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402

VERSION = "6.1"
MONTAGES = (("bip", "bipolar"), ("avg", "average"))
GRID_NEW = {"t_from": (-10.0, -5.0), "z": (3.0, 4.0, 5.0), "persist_s": (5.0, 9.0), "onset_oi": (0.3, 0.5, 0.7),
            "global_q": (0.5, 0.75)}
GRID_EI = {"lead_s": (10.0, 5.0), "threshold": (0.3, 0.5, 0.7)}
# EI + line length: the EI part searches from -10 s; the line-length part from -5 s, lasting 9 s, whole-brain q 0.5
GRID_COMBO = {"threshold": (0.3, 0.5), "z": (3.0, 4.0, 5.0), "onset_oi": (0.3, 0.5)}
COMBO_EI_LEAD = 10.0
COMBO_LL = {"t_from": -5.0, "persist_s": 9.0, "global_q": 0.5}
FAMILY_NAME = {"ei+linelength": "EI + line length (6.1)", "ei": "EI alone (version 5 idea)",
               "linelength": "line length alone", "multifeature": "8-feature alone"}
FAMILY_ORDER = {"ei+linelength": 0, "ei": 1, "linelength": 2, "multifeature": 3}
MONTAGE_NAME = {"bipolar": "bipolar", "average": "common average (CAR)"}


# --------------------------------------------------------------------------- one benchmark seizure
def arrays_dir():
    return os.path.join(config.CACHE_DIR, "benchmark", "arrays")


def electrode_table(sub):
    d = os.path.join(get_benchmark.bench_dir(), "sidecars", sub)
    for f in sorted(os.listdir(d)):
        if f.endswith("_electrodes.tsv"):
            return soz_lib.read_tsv(os.path.join(d, f))
    return None


def seizure_arrays(row, clip_path):
    """The step-2 arrays of one benchmark seizure (made once, then read from the cache)."""
    os.makedirs(arrays_dir(), exist_ok=True)
    cache = os.path.join(arrays_dir(), f"{row['stem']}.npz")
    if os.path.exists(cache):
        try:
            arr = soz_lib.load_arrays(cache)
            if int(arr.get("version", 0)) == soz_lib.VERSION:
                return arr
        except Exception:  # noqa: BLE001 - remake a broken cache file
            pass
    x, names, fs, t_start, onset = get_benchmark.load_clip(clip_path)
    ch = soz_lib.read_tsv(os.path.join(get_benchmark.bench_dir(), "sidecars", row["subject"],
                                       f"{row['stem']}_channels.tsv"))
    el = electrode_table(row["subject"])
    coord_names, coords = set(), {}
    if el is not None and "name" in el.columns:
        coord_names = {soz_lib.normalize_name(n) for n in el["name"]}
        try:
            xyz = el[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
            units = row.get("site", "HUP")
            if units == "HUP" and xyz.notna().all(axis=1).any():     # CHOP tables have voxel indices only
                coords = {soz_lib.normalize_name(n): v for n, v in zip(el["name"], xyz.to_numpy()) if np.isfinite(v).all()}
        except KeyError:
            pass
    idx, how, diag = soz_lib.select_channels(names, ch, coord_names)
    if len(idx) < 3:
        raise ValueError(f"only {len(idx)} usable channels ({how})")
    arr = soz_lib.analyse_signals(x[idx], fs, t_start, onset, [names[i] for i in idx], diag["norms"], coords, 60.0)
    arr["channel_selection"] = how
    soz_lib.save_arrays(cache, arr)
    return soz_lib.load_arrays(cache)


def experts(row):
    norm = soz_lib.normalize_name
    cons = {norm(c) for c in json.loads(row["ueo_consensus"])}
    each = [{norm(c) for c in s} for s in json.loads(row["ueo_each"])]
    return cons, each


def contacts_of(site):
    return set(str(site).split("-"))


def score(pred, truth, usable):
    """F1 between predicted sites (bipolar pairs 'A-B' or contacts 'A') and the expert channels (contacts).
    Only expert channels that the montage can see (usable) count. Returns (f1, precision, recall)."""
    truth = truth & usable
    if not truth:
        return np.nan, np.nan, np.nan
    pred = list(pred)
    if not pred:
        return 0.0, 0.0, 0.0
    covered = set().union(*[contacts_of(p) for p in pred])
    recall = len(truth & covered) / len(truth)
    precision = sum(bool(contacts_of(p) & truth) for p in pred) / len(pred)
    f1 = 0.0 if recall + precision == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def human_scores(each, usable):
    """Each epileptologist against the majority of the others (contact level, channels the montage can see)."""
    out = []
    for i, mine in enumerate(each):
        others = [s & usable for j, s in enumerate(each) if j != i]
        if not others:
            continue
        cnt = {}
        for s in others:
            for c in s:
                cnt[c] = cnt.get(c, 0) + 1
        maj = {c for c, k in cnt.items() if k > len(others) / 2}
        mine = mine & usable
        if not maj:
            continue
        if not mine:
            out.append(0.0)
            continue
        inter = len(mine & maj)
        p, r = inter / len(mine), inter / len(maj)
        out.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return out


def evaluate_seizure(row, arr):
    """Every candidate setting on one seizure -> (result rows, human scores, per-setting onset/already-active sets)."""
    cons, each = experts(row)
    ueo = float(row["ueo_rel_s"])
    t_rel = arr["t"] - arr["onset"]
    rows, human, sets = [], {}, {}
    for key, montage in MONTAGES:
        names, norms = arr[f"{key}_names"], arr[f"{key}_norm"]
        bad = np.asarray(arr[f"{key}_bad"], bool)
        if not len(names):
            continue
        good_sites = [n for n, b in zip(norms, bad) if not b]
        usable = set().union(*[contacts_of(n) for n in good_sites]) if good_sites else set()
        human[montage] = human_scores(each, usable)
        base = {"subject": row["subject"], "patient": row["patient"], "stem": row["stem"], "montage": montage,
                "lvfa": row["lvfa"], "n_sites": len(good_sites), "n_expert": len(cons & usable)}

        def add(family, setting, extra, res):
            flags = res["flags"]
            on = np.asarray(res["onset"], bool) & ~bad
            pred = [] if flags else [n for n, o in zip(norms, on) if o]
            f1, pr, rc = score(pred, cons, usable)
            t0 = res["t0"] if not flags else np.nan
            rows.append({**base, "family": family, "setting": setting, **extra, "f1": f1, "precision": pr,
                         "recall": rc, "flagged": bool(flags), "flags": "; ".join(flags), "n_pred": len(pred),
                         "t0_s": t0, "timing_error_s": t0 - ueo if np.isfinite(t0) else np.nan})
            sets[(montage, family, setting)] = (set(pred), {n for n, e in zip(norms, res["edge"]) if e},
                                                not flags)

        # EI alone (version 6.1 rules)
        ei_res = {}
        for lead in GRID_EI["lead_s"]:
            energy, det = soz_lib.ei_parts(arr, key, lead)
            for thr in GRID_EI["threshold"]:
                r = soz_lib.ei_onset_from_parts(energy, det, bad, {**soz_lib.EI_DEFAULTS, "lead_s": lead,
                                                                   "threshold": thr})
                ei_res[(lead, thr)] = r
                add("ei", f"lead={lead:g} thr={thr:g}", {"lead_s": lead, "threshold": thr}, r)
        # line length / 8-feature alone
        ll_res = {}
        F = arr[f"{key}_feat"]
        for t_from in GRID_NEW["t_from"]:
            p0 = {**onset_lib.DEFAULT_PARAMS, "t_from": t_from}
            try:
                Z = onset_lib.robust_z(F, onset_lib.baseline_mask(t_rel, p0))
            except onset_lib.BaselineError:
                continue
            for method, gq in itertools.product(onset_lib.METHODS, GRID_NEW["global_q"]):
                A = onset_lib.activation(Z, method, global_q=gq)
                A[bad] = np.nan
                for z, persist in itertools.product(GRID_NEW["z"], GRID_NEW["persist_s"]):
                    p = {**p0, "method": method, "z": z, "persist_s": persist, "global_q": gq}
                    det = onset_lib.from_activation(A, t_rel, p)
                    for ooi in GRID_NEW["onset_oi"]:
                        pp = {**p, "onset_oi": ooi}
                        r = {"start": det["start"], "edge": det["edge"] & ~bad, "score": det["oi"],
                             "onset": onset_lib.onset_sites(det, pp) & ~bad, "t0": det["t0"],
                             "flags": onset_lib.seizure_flags(det, len(good_sites), pp)}
                        ll_res[(method, t_from, gq, z, persist, ooi)] = r
                        add(method, f"from={t_from:g} z={z:g} last={persist:g}s oi={ooi:g} q={gq:g}",
                            {"t_from": t_from, "z": z, "persist_s": persist, "onset_oi": ooi, "global_q": gq}, r)
        # EI + line length (version 6.1)
        for thr, z, ooi in itertools.product(GRID_COMBO["threshold"], GRID_COMBO["z"], GRID_COMBO["onset_oi"]):
            e = ei_res.get((COMBO_EI_LEAD, thr))
            ll = ll_res.get(("linelength", COMBO_LL["t_from"], COMBO_LL["global_q"], z, COMBO_LL["persist_s"], ooi))
            if e is None or ll is None:
                continue
            add("ei+linelength", f"EI>={thr:g} | z={z:g} oi={ooi:g}", {"threshold": thr, "z": z, "onset_oi": ooi},
                soz_lib.combine_onsets(e, ll))
    return rows, human, sets


# --------------------------------------------------------------------------- choosing
def split_patients(patients):
    """Fixed split (decided before any result): patients sorted by name, alternately tuning / test."""
    ps = sorted(patients)
    return {p: ("tune" if k % 2 == 0 else "test") for k, p in enumerate(ps)}


def summarise(per):
    g = per.dropna(subset=["f1"]).groupby(["family", "montage", "setting", "split"])
    s = g.agg(seizures=("f1", "size"), f1_mean=("f1", "mean"), f1_median=("f1", "median"),
              precision=("precision", "mean"), recall=("recall", "mean"),
              answered=("n_pred", lambda v: float((v > 0).mean())),
              flagged=("flagged", "mean"),
              timing_abs_median_s=("timing_error_s", lambda v: float(np.nanmedian(np.abs(v))) if np.isfinite(v).any()
                                   else np.nan)).reset_index()
    return s


def bootstrap_ci(per_cfg, n=2000, seed=0):
    """95% interval of the mean F1, resampling PATIENTS (seizures of one patient are not independent)."""
    rng = np.random.default_rng(seed)
    by_p = [g["f1"].to_numpy() for _, g in per_cfg.groupby("patient")]
    if len(by_p) < 2:
        return np.nan, np.nan
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(by_p), len(by_p))
        v = np.concatenate([by_p[i] for i in pick])
        means.append(v.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def choose(summ):
    """Best setting per (family, montage) on the tuning half, then the overall winner (tuning half only)."""
    tune = summ[summ["split"] == "tune"].copy()
    tune["order"] = tune["family"].map(FAMILY_ORDER)
    tune["montage_order"] = (tune["montage"] != "bipolar").astype(int)    # on a tie: bipolar (better localisation)
    tune = tune.sort_values(["f1_mean", "f1_median", "answered", "montage_order", "order", "setting"],
                            ascending=[False, False, False, True, True, True])
    best = tune.groupby(["family", "montage"], sort=False).head(1).reset_index(drop=True)
    return best, best.iloc[0]


def params_of(per, family, montage, setting):
    """Complete settings of one (family, setting) row - what step 2 needs (soz_lib.method_params)."""
    r = per[(per["family"] == family) & (per["montage"] == montage) & (per["setting"] == setting)].iloc[0]
    if family == "ei":
        return soz_lib.method_params("ei", {"lead_s": float(r["lead_s"]), "threshold": float(r["threshold"])})
    if family == "ei+linelength":
        return soz_lib.method_params("ei+linelength", {
            "ei": {"lead_s": COMBO_EI_LEAD, "threshold": float(r["threshold"])},
            "ll": {**COMBO_LL, "z": float(r["z"]), "onset_oi": float(r["onset_oi"])}})
    return soz_lib.method_params(family, {"t_from": float(r["t_from"]), "z": float(r["z"]),
                                          "persist_s": float(r["persist_s"]), "onset_oi": float(r["onset_oi"]),
                                          "global_q": float(r["global_q"])})


def patient_level(sets, seizures, experts_of, usable_of, montage, family, setting):
    """Step-2 SOZ rule on the benchmark patients with >= 2 seizures, against the epileptologists' channels.
    Returns rows: patient, SOZ sites, precision, recall, F1, share of the experts' habitual channels that the
    labels would call 'not SOZ' (neither SOZ nor uncertain)."""
    out = []
    for patient, stems in seizures.items():
        if len(stems) < 2:
            continue
        counts, active, used, cons_n, union, usable = {}, set(), 0, {}, set(), set()
        for s in stems:
            usable |= usable_of[(s, montage)]
            for c in experts_of[s]:
                cons_n[c] = cons_n.get(c, 0) + 1
            union |= experts_of[s]
            got = sets.get((s, montage, family, setting))
            if got is None:
                continue
            pred, edge, ok = got
            active |= edge
            if not ok:
                continue
            used += 1
            for n in pred:
                counts[n] = counts.get(n, 0) + 1
        need = max(config.SOZ_MIN_ONSETS, int(math.ceil(config.SOZ_MIN_SHARE * used))) if used >= 2 else None
        soz = {n for n, c in counts.items() if need and c >= need}
        unc = ({n for n, c in counts.items() if c > 0} | active) - soz
        union &= usable
        habitual = ({c for c, k in cons_n.items() if k >= 2} & usable) or union
        if not habitual:
            continue
        cov = set().union(*[contacts_of(n) for n in soz]) if soz else set()
        cov_any = cov | (set().union(*[contacts_of(n) for n in unc]) if unc else set())
        prec = float(np.mean([bool(contacts_of(n) & union) for n in soz])) if soz else np.nan
        rec = len(habitual & cov) / len(habitual)
        f1 = 0.0 if not soz or not (prec + rec) else 2 * prec * rec / (prec + rec)
        out.append({"family": family, "montage": montage, "setting": setting, "patient": patient,
                    "seizures": len(stems), "usable": used, "soz_sites": len(soz), "precision": prec, "recall": rec,
                    "f1": f1, "expert_channels_labelled_not_soz": len(habitual - cov_any) / len(habitual)})
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also the children's (CHOP) seizures (not used to choose)")
    ap.add_argument("--partial", action="store_true", help=argparse.SUPPRESS)   # (tests) allow missing seizures
    args = ap.parse_args()
    m = get_benchmark.seizure_table(include_children=args.all)
    out = config.SOZ_RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    cdir = get_benchmark.clips_dir()
    adir = arrays_dir()
    have = [r for _, r in m.iterrows() if os.path.exists(os.path.join(cdir, f"{r['stem']}.npz"))
            or os.path.exists(os.path.join(adir, f"{r['stem']}.npz"))]
    if not have:
        raise SystemExit("No benchmark seizures downloaded yet - run first:  python tools\\get_benchmark.py")
    missing = len(m) - len(have)
    print(f"Expert benchmark: {len(have)} of {len(m)} seizures downloaded"
          + (f" ({missing} missing - run tools\\get_benchmark.py again to complete it)" if missing else ""))
    if len(have) < 0.9 * len(m) and not args.partial:
        raise SystemExit("Too many benchmark seizures are missing for a fair choice - run  python tools\\get_benchmark.py "
                         "again (it continues where it stopped), then this script.")

    rows, human_rows, failed, all_sets = [], [], [], {}
    experts_of, usable_of, seizures_of = {}, {}, {}
    for k, r in enumerate(have, start=1):
        try:
            arr = seizure_arrays(r, os.path.join(cdir, f"{r['stem']}.npz"))
            res, human, sets = evaluate_seizure(r, arr)
            rows += res
            for mont, vals in human.items():
                human_rows += [{"patient": r["patient"], "stem": r["stem"], "montage": mont, "f1": v} for v in vals]
            for (mont, fam, setting), v in sets.items():
                all_sets[(r["stem"], mont, fam, setting)] = v
            experts_of[r["stem"]] = experts(r)[0]
            for key, mont in MONTAGES:
                good = [n for n, b in zip(arr[f"{key}_norm"], arr[f"{key}_bad"]) if not b]
                usable_of[(r["stem"], mont)] = set().union(*[contacts_of(n) for n in good]) if good else set()
            seizures_of.setdefault(r["patient"], []).append(r["stem"])
            nb = sum(1 for x in arr["bip_bad"] if not x)
            print(f"  [{k}/{len(have)}] {r['subject']} {r['task']}: {len(arr['avg_names'])} contacts / {nb} pairs, "
                  f"{len(experts(r)[0])} consensus onset channels")
        except Exception as e:  # noqa: BLE001
            failed.append(f"{r['stem']}: {type(e).__name__}: {e}")
            print(f"  [{k}/{len(have)}] {r['subject']} {r['task']}: SKIPPED - {type(e).__name__}: {e}")
    if not rows:
        raise SystemExit("No benchmark seizure could be analysed.")
    per = pd.DataFrame(rows)
    split = split_patients(m.loc[m["site"] == "HUP", "patient"].unique())
    per["split"] = per["patient"].map(split).fillna("extra (children)")
    per.to_csv(os.path.join(out, "benchmark_per_seizure.csv"), index=False)
    human = pd.DataFrame(human_rows)
    human["split"] = human["patient"].map(split).fillna("extra (children)")
    summ = summarise(per)
    summ.to_csv(os.path.join(out, "benchmark_scores.csv"), index=False)

    best, win = choose(summ)
    test = per[per["split"] == "test"]

    def test_row(family, montage, setting):
        t = test[(test["family"] == family) & (test["montage"] == montage) & (test["setting"] == setting)]
        t = t.dropna(subset=["f1"])
        lo, hi = bootstrap_ci(t)
        timing = t["timing_error_s"].abs()
        return {"f1_mean": float(t["f1"].mean()) if len(t) else np.nan,
                "f1_median": float(t["f1"].median()) if len(t) else np.nan, "ci_low": lo, "ci_high": hi,
                "precision": float(t["precision"].mean()) if len(t) else np.nan,
                "recall": float(t["recall"].mean()) if len(t) else np.nan,
                "answered": float((t["n_pred"] > 0).mean()) if len(t) else np.nan, "seizures": len(t),
                "timing_abs_median_s": float(np.nanmedian(timing)) if timing.notna().any() else np.nan}

    best_rows, pat_rows = [], []
    for _, b in best.iterrows():
        tr = test_row(b["family"], b["montage"], b["setting"])
        best_rows.append({"family": b["family"], "montage": b["montage"], "setting": b["setting"],
                          "tune_f1": b["f1_mean"], **{f"test_{k}": v for k, v in tr.items()}})
        hup = {p: s for p, s in seizures_of.items() if p in split}
        pat_rows += patient_level({(s, mo, f, st): v for (s, mo, f, st), v in all_sets.items()}, hup, experts_of,
                                  usable_of, b["montage"], b["family"], b["setting"])
    best_tab = pd.DataFrame(best_rows)
    pats = pd.DataFrame(pat_rows)
    if len(pats):
        pats.to_csv(os.path.join(out, "benchmark_patients.csv"), index=False)
    ei_best = best_tab[best_tab["family"] == "ei"].iloc[0]
    w = best_tab.iloc[0]
    passed = bool(w["family"] == "ei" or (np.isfinite(w["test_f1_mean"]) and
                                          w["test_f1_mean"] >= ei_best["test_f1_mean"]))
    final = w if passed else ei_best
    choice = {"version": VERSION, "date": datetime.datetime.now().isoformat(timespec="seconds"),
              "method": final["family"], "montage": final["montage"], "setting": final["setting"],
              "params": params_of(per, final["family"], final["montage"], final["setting"]),
              "tune_f1": round(float(final["tune_f1"]), 4), "test_f1": round(float(final["test_f1_mean"]), 4),
              "test_f1_ci": [round(float(final["test_ci_low"]), 4), round(float(final["test_ci_high"]), 4)],
              "ei_test_f1": round(float(ei_best["test_f1_mean"]), 4), "winner_on_tuning": w["family"],
              "winner_passed_test": passed, "seizures_used": int(per["stem"].nunique()),
              "patients": int(per["patient"].nunique()),
              "human_f1_median": round(float(human["f1"].median()), 4) if len(human) else None}
    with open(os.path.join(out, "benchmark_choice.json"), "w") as fh:
        json.dump(choice, fh, indent=1)

    # ---------------------------------------------------------------- report
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    n_tune = per.loc[per["split"] == "tune", "stem"].nunique()
    n_test = per.loc[per["split"] == "test", "stem"].nunique()
    say("=" * 112)
    say(f"EXPERT BENCHMARK (version {VERSION}) - which onset method agrees best with epileptologists?")
    say("=" * 112)
    say(f"{per['stem'].nunique()} seizures from {per['patient'].nunique()} patients, marked by 3-5 epileptologists "
        f"(consensus onset channels). Tuning half: {n_tune} seizures, test half: {n_test} seizures (other patients).")
    say("Score = F1 between the method's onset channels and the epileptologists' (1 = same channels, 0 = none right;")
    say("a seizure the method cannot use counts 0). Each method uses its best setting on the TUNING half; the TEST")
    say("half was not used to choose anything.\n")
    hdr = (f"{'method':<28}{'montage':<22}{'tuning F1':>10}{'TEST F1':>9}{'95% interval':>16}{'answered':>10}"
           f"{'precision':>10}{'recall':>8}{'timing':>9}")
    say(hdr)
    say("-" * len(hdr))
    for _, b in best_tab.iterrows():
        mark = "  <- chosen" if (b["family"] == final["family"] and b["montage"] == final["montage"]) else ""
        say(f"{FAMILY_NAME[b['family']]:<28}{MONTAGE_NAME[b['montage']]:<22}{b['tune_f1']:>10.2f}"
            f"{b['test_f1_mean']:>9.2f}{f'{b.test_ci_low:.2f}-{b.test_ci_high:.2f}':>16}{b['test_answered']:>10.0%}"
            f"{b['test_precision']:>10.2f}{b['test_recall']:>8.2f}{b['test_timing_abs_median_s']:>8.1f}s{mark}")
    hb = human[human["montage"] == "bipolar"]
    if len(hb):
        say(f"{'one epileptologist vs the others (human level)':<50}{hb['f1'].mean():>10.2f}"
            f"{hb.loc[hb['split'] == 'test', 'f1'].mean():>9.2f}   (median {hb['f1'].median():.2f})")
    say("")
    say("answered = seizures where the method gave onset channels; timing = median |method start - experts' start|.")
    say("Settings of each row:")
    for _, b in best_tab.iterrows():
        say(f"  {FAMILY_NAME[b['family']]:<28}{b['montage']:<9} {b['setting']}")
    say("")
    if passed:
        say(f"CHOSEN: {FAMILY_NAME[final['family']]}, {MONTAGE_NAME[final['montage']]} ({final['setting']}): best on the "
            f"tuning half, and on the test half F1 {final['test_f1_mean']:.2f} vs EI {ei_best['test_f1_mean']:.2f}.")
    else:
        say(f"The best method on the tuning half ({FAMILY_NAME[w['family']]}, {w['montage']}) was NOT better than the EI "
            f"on the test half ({w['test_f1_mean']:.2f} vs {ei_best['test_f1_mean']:.2f}), so the EI is kept "
            f"({ei_best['montage']}, {ei_best['setting']}).")
    if len(pats):
        say(f"\nPATIENT LABELS with each method (the SOZ rule of step 2 on the {pats['patient'].nunique()} patients with "
            f">= 2 seizures, both halves):")
        say(f"  {'method':<28}{'montage':<10}{'SOZ F1':>8}{'precision':>11}{'recall':>8}"
            f"{'expert channels called not-SOZ':>33}")
        for (fam, mont), g in sorted(pats.groupby(["family", "montage"]),
                                     key=lambda kv: -kv[1]["f1"].mean()):
            say(f"  {FAMILY_NAME[fam]:<28}{mont:<10}{g['f1'].mean():>8.2f}{g['precision'].mean():>11.2f}"
                f"{g['recall'].mean():>8.2f}{g['expert_channels_labelled_not_soz'].mean():>33.0%}")
        say("  (precision = share of the SOZ sites on a channel the experts marked; recall = share of the experts'")
        say("   usual onset channels inside the SOZ; the last column should be small: those would be wrong 'not SOZ'")
        say("   examples for the model)")
    # subgroups: onset with / without low-voltage fast activity
    say("\nBy onset pattern (all seizures, the settings above): mean F1 with LVFA at onset / without LVFA")
    for _, b in best_tab.iterrows():
        q = per[(per["family"] == b["family"]) & (per["montage"] == b["montage"]) & (per["setting"] == b["setting"])]
        q = q.dropna(subset=["f1"])
        yes, no = q.loc[q["lvfa"] == "yes", "f1"], q.loc[q["lvfa"] == "no", "f1"]
        say(f"  {FAMILY_NAME[b['family']]:<28}{b['montage']:<9} with LVFA {yes.mean():.2f} ({len(yes)})   "
            f"without LVFA {no.mean():.2f} ({len(no)})")
    if failed:
        say(f"\n{len(failed)} seizures could not be analysed: " + "; ".join(failed))
    say(f"\nSaved benchmark_choice.json (used by step 2), benchmark_scores.csv, benchmark_per_seizure.csv, "
        f"benchmark_patients.csv in {out}")
    say("Next: python SOZ\\01_scan_dataset.py  (if not done yet)  then  python SOZ\\02_compute_soz.py")
    with open(os.path.join(out, "benchmark_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    with runlog.Log("benchmark_onset"):
        main()
