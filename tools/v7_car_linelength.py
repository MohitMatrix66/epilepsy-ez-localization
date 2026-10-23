"""
tools/v7_car_linelength.py  -  VERSION 7 TRIAL: where do the seizures start when we use ONLY line length on the COMMON
AVERAGE reference (CAR)? Is that better than version 6.1, and does it give MORE patients we can trust for training?

Run from the project folder, after step 2 and step 3 of version 6.1 (nothing is downloaded, nothing is re-analysed):
    python tools\\v7_car_linelength.py
About 1-3 minutes. It changes NOTHING of version 6.1 (contacts.csv, the labels and the model stay as they are).
Writes SOZ\\results\\v7\\:
    v7_report.txt        the summary that is printed at the end  <- send this to Claude
    v7_patients.csv      one row per patient: v7 SOZ, its side, agreement with version 6.1 and the clinical record
    v7_labels.csv        one row per contact: SOZ / uncertain / not SOZ by v7, and why
    v7_benchmark.csv     the expert test, one row per seizure
    pictures\\<patient>.png   what v7 sees in (up to) 4 seizures of each patient

The v7 method, for every contact (signal = contact minus the average of all good contacts, as saved by step 2):
  1. line length = how much the signal moves (sum of |x(t) - x(t-1)|) in 1-s windows every 0.25 s;
  2. divided by the contact's OWN normal level (median of -30 to -12 s before the marked onset): 1 = normal;
  3. a change-point detector (Page-Hinkley, the one the EI uses) finds WHEN it rises and STAYS high (>= 2 x normal),
     searched from -10 to +10 s; a rise before -5 s = "already active" (the contact was busy before the seizure);
  4. strength = line length over the 5 s after the rise; score = strength / (delay after the earliest contact + 1 s),
     the best contact = 1; onset contacts = score >= 0.3.
  The settings (bias 0.1, alarm 10, stay >= 2 x normal, score >= 0.3) were chosen on the 40 seizures marked by
  epileptologists, leave-one-patient-out (they won in 17 of 19 rounds).
Noise and other traps:
  * channels marked bad by step 2 are not used; a channel whose normal period is far more variable than the others
    (spread > 3 x the typical spread: artifacts, a loose contact) is not used either - it becomes "uncertain";
  * a seizure in which >= 20% of the rising contacts were already active (a state change, e.g. one seizure running into
    the next) or in which > 40% of the contacts rise together (diffuse, or a whole-brain artifact) is not used.
Patient labels (the rule of step 2):
  SOZ contact = onset contact in >= 2 usable seizures and in >= 25% of them;
  uncertain   = onset contact less often, already active, noisy, or the next contact on the same electrode as a SOZ or
                uncertain contact (3.5-10 mm away: nearly the same piece of brain for a 32 mm CNN patch);
  not SOZ     = every other good contact.
  A patient's labels are trusted (label quality 'ok' / 'few seizures', like step 2) when there are >= 2 usable
  seizures, a SOZ, and the SOZ (or a contact next to it) is an onset contact in >= 30% of them (SOZ_MIN_LED).
The verdict for each of your patients (no expert marks there, so v7 must also look plausible): label quality 'ok' /
'few seizures', the most regular SOZ contact starts at least HALF of the usable seizures (otherwise the 'SOZ' can be a
coincidence of onsets that change from seizure to seizure), the SOZ lies on one side (>= 75% of its contacts) and that
is the clinical side. Then: TRAIN (version 6.1 trains the patient; does v7 agree on the same electrodes?), CANDIDATE
(only v7 finds a SOZ - look at its picture first) or LEAVE OUT.
"""
import argparse
import json
import math
import os
import sys
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SOZ"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402

VERSION = "7-trial"
STEP_S = 0.25
P = {"lead_s": 10.0, "bias": 0.1, "lam": 10.0, "sustain": 2.0, "threshold": 0.3, "early_s": -5.0, "end_s": 10.0,
     "active_share": 0.2, "diffuse_share": 0.4, "noisy_x": 3.0, "spread_floor": 0.05}
INK, SOZ_C, UNC_C, V61_C = "#1f1f1e", "#256abf", "#c45f1c", "#2e8540"


# =========================================================================== the v7 detector (one seizure)
def car_line_length_onset(LL, t, bad, p=P):
    """LL: log10 line length (contacts x windows, common average); t: window times in s from the marked onset;
    bad: contacts marked bad by step 2.
    Returns a dict of per-contact arrays onset, active (already active), noisy (bool), start (s), score (0-1),
    ratio (line length / normal level, contacts x windows) and flags ([] = the seizure is usable)."""
    LL = np.asarray(LL, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = LL.shape[0]
    good = ~np.asarray(bad, bool)
    out = {"onset": np.zeros(n, bool), "active": np.zeros(n, bool), "noisy": np.zeros(n, bool),
           "start": np.full(n, np.nan), "score": np.zeros(n), "ratio": np.ones_like(LL), "flags": []}
    base = (t >= -30.0) & (t < -p["lead_s"] - 2.0)
    if not good.any():
        out["flags"] = ["no good contacts"]
        return out
    if base.sum() * STEP_S < 8.0:
        out["flags"] = ["short normal period"]
        return out
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(LL[:, base], axis=1)
        spread = 1.4826 * np.nanmedian(np.abs(LL[:, base] - med[:, None]), axis=1)
        typical = max(float(np.nanmedian(spread[good])), p["spread_floor"]) if np.isfinite(spread[good]).any() \
            else p["spread_floor"]
        R = np.nan_to_num(10.0 ** (LL - med[:, None]), nan=0.0, posinf=0.0)
    noisy = good & np.isfinite(spread) & (spread > p["noisy_x"] * typical)
    use = good & ~noisy & np.isfinite(med)
    out["noisy"], out["ratio"] = noisy, R
    mu0 = R[:, base].mean(axis=1)
    win = np.flatnonzero((t >= -p["lead_s"] - 1e-9) & (t <= 20.0 + 1e-9))
    if not len(win) or not use.any():
        out["flags"] = ["no onset"]
        return out
    onset_frame, confirm_frame = int(np.searchsorted(t, 0.0)), int(np.searchsorted(t, 1.0))
    h = int(round(5.0 / STEP_S))
    start, energy = np.full(n, np.nan), np.zeros(n)
    for c in np.flatnonzero(use):
        j = soz_lib.detect_change(R[c], win[0], win[-1] + 1, mu0[c], onset_frame, confirm_frame, bias=p["bias"],
                                  lam=p["lam"], sustain=p["sustain"])
        if j >= 0 and t[j] <= p["end_s"] + 1e-9:
            start[c] = t[j]
            energy[c] = R[c, j:j + h].sum()
    out["start"] = start
    rose = np.isfinite(start)
    active = rose & ((start < p["early_s"] - 1e-9) | (start <= t[win[0]] + STEP_S / 2))
    out["active"] = active
    if not rose.any():
        out["flags"] = ["no onset"]
        return out
    if active.any() and active.sum() >= p["active_share"] * rose.sum():
        out["flags"] = ["state change before the onset"]
        return out
    ok = rose & ~active
    t0 = float(np.nanmin(start[ok]))
    score = np.zeros(n)
    score[ok] = energy[ok] / (start[ok] - t0 + 1.0)
    if score.max() > 0:
        score /= score.max()
    out["score"] = score
    on = ok & (score >= p["threshold"])
    if on.sum() > max(3, p["diffuse_share"] * use.sum()):
        out["flags"] = ["diffuse onset"]
        return out
    out["onset"] = on
    return out


# =========================================================================== names
def neighbours(c):
    """A contact and the contacts next to it on the same electrode ('LA3' -> LA2, LA3, LA4)."""
    s = soz_lib.split_numbered(str(c))
    return {c} if not s else {f"{s[0]}{s[1] - 1}", c, f"{s[0]}{s[1] + 1}"}


def electrode(c):
    s = soz_lib.split_numbered(str(c))
    return s[0] if s else str(c)


def contacts_of(sites):
    """Contacts of bipolar pairs ('LA1-LA2' -> LA1, LA2) or of contacts."""
    out = set()
    for s in sites:
        out |= set(str(s).split("-"))
    return out


def near(sites):
    return set().union(*[neighbours(c) for c in contacts_of(sites)]) if sites else set()


# =========================================================================== patient labels
def repeated(per_seizure):
    """per_seizure: list of onset-site sets (None = seizure not usable). Returns (counts, usable seizures, need):
    need = onsets a SOZ site needs = max(SOZ_MIN_ONSETS, 25% of the usable seizures), like step 2."""
    counts, used = {}, 0
    for on in per_seizure:
        if on is None:
            continue
        used += 1
        for c in on:
            counts[c] = counts.get(c, 0) + 1
    need = max(config.SOZ_MIN_ONSETS, int(math.ceil(config.SOZ_MIN_SHARE * used - 1e-9)))
    return counts, used, min(need, max(used, 1))


def led_share(per_seizure, soz):
    """Share of the usable seizures in which the SOZ (or a contact next to it) is an onset site (step 2's 'leads')."""
    if not soz:
        return np.nan
    zone = near(soz)
    hits = [bool(contacts_of(on) & zone) for on in per_seizure if on is not None]
    return float(np.mean(hits)) if hits else np.nan


def by_chance(per_seizure, n_labelled, need):
    """How many sites would reach `need` onsets if every seizure's onset sites were picked at random (same number of
    onset sites per seizure). A SOZ that is not clearly larger than this can be a coincidence."""
    ps = [len(on) / n_labelled for on in per_seizure if on is not None] if n_labelled else []
    if len(ps) < 2:
        return np.nan
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for q in ps:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return float(n_labelled * dist[need:].sum())


def quality(used, n_soz, led):
    return soz_lib.label_quality(used, n_soz, led, config.SOZ_MIN_LED, "repeated")


def v7_patient(seizures, labelled):
    """seizures: list of dicts onset (set or None), active, noisy (sets of contacts); labelled: the patient's good
    contacts. Returns dict soz (set), unc ({contact: reason}), labelled, used, counts, need, led, quality."""
    counts, used, need = repeated([s["onset"] for s in seizures])
    soz = {c for c, k in counts.items() if k >= need}
    unc = {c: f"onset contact in {k} of {used} usable seizures (SOZ needs {need})" for c, k in counts.items()
           if c not in soz}
    for key, text in (("active", "already active before the onset"), ("noisy", "noisy normal period")):
        for s in seizures:
            for c in sorted(s[key]):
                if c not in soz:
                    unc.setdefault(c, text)
    for c in sorted(near(soz | set(unc))):
        if c in labelled and c not in soz:
            unc.setdefault(c, "next to a SOZ / uncertain contact")
    led = led_share([s["onset"] for s in seizures], soz)
    best = max(counts[c] for c in soz) if soz else 0
    return {"soz": soz, "unc": unc, "labelled": set(labelled), "used": used, "counts": counts, "need": need,
            "led": led, "quality": quality(used, len(soz), led), "best": best,
            "chance": by_chance([s["onset"] for s in seizures], len(labelled), need)}


def v61_patient(seizures, labelled_pairs):
    """The step 2 rule of version 6.1 on bipolar pairs, then turned into contacts (a contact of a SOZ pair = SOZ)."""
    counts, used, need = repeated([s["onset"] for s in seizures])
    soz_p = {c for c, k in counts.items() if k >= need}
    unc_p = set(counts) - soz_p
    for s in seizures:
        unc_p |= s["active"] - soz_p
    soz = contacts_of(soz_p)
    led = led_share([s["onset"] for s in seizures], soz_p)
    return {"soz": soz, "unc": {c: "" for c in contacts_of(unc_p) - soz}, "labelled": contacts_of(labelled_pairs),
            "used": used, "need": need, "led": led, "quality": quality(used, len(soz_p), led)}


# =========================================================================== the expert test
def score_contacts(pred, truth, usable, tol=False):
    """F1, precision, recall of predicted contacts against the expert contacts (only those the CAR can see).
    tol: a contact next to an expert contact (same electrode) also counts."""
    truth = truth & usable
    if not truth:
        return np.nan, np.nan, np.nan
    if not pred:
        return 0.0, 0.0, 0.0
    nb = neighbours if tol else (lambda c: {c})
    cover = set().union(*[nb(c) for c in pred])
    rec = len(truth & cover) / len(truth)
    prec = float(np.mean([bool(nb(c) & truth) for c in pred]))
    return (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)), prec, rec


def patient_metrics(lab, union, habitual, tol):
    """SOZ precision / recall / F1 against the experts' onset contacts of a patient, and the share of the experts'
    usual onset contacts that the labels call NOT SOZ (taught to the model as healthy: the harmful error)."""
    nb = neighbours if tol else (lambda c: {c})
    soz, unc = lab["soz"], set(lab["unc"])
    cover = set().union(*[nb(c) for c in soz]) if soz else set()
    cover_any = cover | (set().union(*[nb(c) for c in unc]) if unc else set())
    harmful = len((habitual & lab["labelled"]) - cover_any) / len(habitual)
    if not soz:
        return {"f1": 0.0, "precision": np.nan, "recall": 0.0, "harmful": harmful, "n_soz": 0}
    prec = float(np.mean([bool(nb(c) & union) for c in soz]))
    rec = len(habitual & cover) / len(habitual)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"f1": f1, "precision": prec, "recall": rec, "harmful": harmful, "n_soz": len(soz)}


def expert_names(row):
    return {soz_lib.normalize_name(c) for c in json.loads(row["ueo_consensus"])}


def benchmark_test(params61):
    """v7 and version 6.1 on the seizures marked by epileptologists. Returns (per seizure, per patient) tables or
    (None, None) when the benchmark (tools/benchmark_onset.py) has not been run."""
    adir = os.path.join(config.CACHE_DIR, "benchmark", "arrays")
    table = os.path.join(config.BENCHMARK_DIR, "seizures.csv")
    if not (os.path.isdir(adir) and os.path.exists(table)):
        return None, None
    m = pd.read_csv(table, dtype={"run": str})
    m = m[m["site"] == "HUP"]
    rows, by_patient = [], {}
    for _, r in m.iterrows():
        path = os.path.join(adir, f"{r['stem']}.npz")
        if not os.path.exists(path):
            continue
        arr = soz_lib.load_arrays(path)
        cons = expert_names(r)
        norms = np.asarray(arr["avg_norm"]).astype(str)
        bad = np.asarray(arr["avg_bad"], bool)
        usable = {c for c, b in zip(norms, bad) if not b}
        d = car_line_length_onset(arr["avg_feat"][0], arr["t"] - arr["onset"], bad)
        v7 = None if d["flags"] else {c for c, o in zip(norms, d["onset"]) if o}
        s61 = soz_lib.site_onsets(arr, "bip", params61)
        bn = np.asarray(arr["bip_norm"]).astype(str)
        bbad = np.asarray(arr["bip_bad"], bool)
        v61 = None if s61["flags"] else {c for c, o in zip(bn, s61["onset"]) if o}
        row = {"stem": r["stem"], "patient": r["patient"], "expert_contacts": len(cons & usable),
               "v7_flags": "; ".join(d["flags"]), "v7_onset": " ".join(sorted(v7 or [])),
               "v61_flags": "; ".join(s61["flags"]), "v61_onset": " ".join(sorted(v61 or [])),
               "experts": " ".join(sorted(cons))}
        for name, pred in (("v7", v7 or set()), ("v61", contacts_of(v61 or []))):
            for tol in (False, True):
                f1, _, _ = score_contacts(sorted(pred), cons, usable, tol)
                row[f"{name}_f1{'_pm1' if tol else ''}"] = f1
        rows.append(row)
        by_patient.setdefault(r["patient"], []).append({
            "cons": cons, "usable": usable, "pairs": {c for c, b in zip(bn, bbad) if not b},
            "v7": {"onset": v7, "active": {c for c, a in zip(norms, d["active"]) if a},
                   "noisy": {c for c, x in zip(norms, d["noisy"]) if x}},
            "v61": {"onset": v61, "active": {c for c, e, b in zip(bn, s61["edge"], bbad) if e and not b}}})
    per = pd.DataFrame(rows)
    if len(per):
        per = per[per["expert_contacts"] > 0]
    prow = []
    for patient, seizures in by_patient.items():
        if len(seizures) < 2:
            continue
        usable = set().union(*[s["usable"] for s in seizures])
        cnt = {}
        for s in seizures:
            for c in s["cons"] & usable:
                cnt[c] = cnt.get(c, 0) + 1
        union = set(cnt)
        habitual = {c for c, k in cnt.items() if k >= 2} or union
        if not habitual:
            continue
        labs = {"v7": v7_patient([s["v7"] for s in seizures], usable),
                "v61": v61_patient([s["v61"] for s in seizures], set().union(*[s["pairs"] for s in seizures]))}
        for name, lab in labs.items():
            for tol in (False, True):
                prow.append({"patient": patient, "method": name, "tol": tol, "quality": lab["quality"],
                             **patient_metrics(lab, union, habitual, tol)})
    return per, pd.DataFrame(prow)


# =========================================================================== your patients
def read_csv(res, name, **kw):
    path = os.path.join(res, name)
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def clinical_side(text):
    s = str(text).strip().lower()
    return "L" if s.startswith(("left only", "left >")) else "R" if s.startswith(("right only", "right >")) else ""


def clinical_lobe(text):
    s = str(text).strip().lower()
    return next((x for x in ("temporal", "frontal", "parietal", "occipital", "insula") if s.startswith(x)), "")


def majority(values, need=0.75):
    """The value that holds >= need of the known values; 'both' when no value does; '' when none is known."""
    v = pd.Series([x for x in values if x and x not in ("?", "nan")], dtype=object)
    if v.empty:
        return ""
    c = v.value_counts()
    return c.index[0] if c.iloc[0] >= need * len(v) else "both"


def text_or_blank(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def picture(sub, panels, labels61, out_path):
    """Line length / normal level of every good contact in (up to) 4 seizures; marks: start of each onset contact
    (white bar), already active (grey x); right edge: v7 SOZ (blue), v7 uncertain (orange), version 6.1 SOZ (green)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = panels[0]["lab"]
    k = len(panels)
    ncol = 1 if k == 1 else 2
    nrow = int(math.ceil(k / ncol))
    n_rows_max = max(int((~p["bad"]).sum()) for p in panels)
    height = min(4.0 + n_rows_max / 30.0, 10.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, height * nrow + 0.8), squeeze=False)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    for ax, p in zip(axes.ravel(), panels):
        keep = np.flatnonzero(~p["bad"])
        names = p["norms"][keep]
        t = p["t"]
        sel = (t >= -20.0) & (t <= 20.0)
        d = p["d"]
        img = np.log2(np.clip(d["ratio"][keep][:, sel], 0.25, 16.0))
        ax.imshow(img, aspect="auto", cmap="magma", vmin=-1.0, vmax=3.0, interpolation="nearest",
                  extent=(t[sel][0], t[sel][-1], len(keep) - 0.5, -0.5))
        for x0, ls in ((0.0, "-"), (P["early_s"], ":"), (P["end_s"], ":")):
            ax.axvline(x0, color="white", lw=0.8, ls=ls, alpha=0.7)
        for row, c in enumerate(keep):
            if np.isfinite(d["start"][c]):
                if d["onset"][c]:
                    ax.plot([d["start"][c]] * 2, [row - 0.45, row + 0.45], color="white", lw=2.2)
                elif d["active"][c]:
                    ax.plot(d["start"][c], row, "x", color="#bbbbbb", ms=4)
        rows_unc = [row for row, c in enumerate(names) if c in lab["unc"] and c not in lab["soz"]]
        rows_soz = [row for row, c in enumerate(names) if c in lab["soz"]]
        rows_61 = [row for row, c in enumerate(names) if c in labels61]
        for rws, x0, col, ms, z in ((rows_unc, 20.5, UNC_C, 3.0, 3), (rows_soz, 20.5, SOZ_C, 4.5, 4),
                                    (rows_61, 21.5, V61_C, 4.5, 4)):
            if rws:
                ax.plot([x0] * len(rws), rws, "s", color=col, ms=ms, clip_on=False, zorder=z, lw=0)
        ticks = [i for i, c in enumerate(names) if i == 0 or electrode(c) != electrode(names[i - 1])]
        ax.set_yticks(ticks)
        ax.set_yticklabels([electrode(names[i]) for i in ticks], fontsize=6)
        ax.set_xlim(-20.0, 20.0)
        ax.set_xlabel("seconds from the marked onset", fontsize=8)
        status = "; ".join(d["flags"]) if d["flags"] else f"{int(d['onset'].sum())} onset contacts"
        ax.set_title(f"{p['name']}  ({status})", fontsize=8, color=INK)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{sub} - v7: line length of every good contact / its own normal level (purple = normal, black = "
                 f"lower, pink = 2 x, orange-yellow = 4-8 x).\nWhite bar = start of an onset contact, grey x = already "
                 f"active; dotted lines at -5 s and +10 s = the search limits.  Right edge: blue = v7 SOZ, orange = "
                 f"v7 uncertain, green = version 6.1 SOZ.", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / (height * nrow + 0.8)))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def your_patients(res, arr_dir, out_dir, pictures):
    side_tab = read_csv(res, "contacts_all.csv")
    lab61 = read_csv(res, "soz_labels.csv")
    summ61 = read_csv(res, "soz_summary.csv")
    cc = read_csv(res, "coordinate_check.csv")
    unloc = read_csv(res, "unlocalized_check.csv")
    contacts61 = read_csv(res, "contacts.csv")
    trained_step3 = set(contacts61["subject"]) if contacts61 is not None else None
    excluded = getattr(config, "EXCLUDE_FROM_TRAINING", {}) or {}
    by_sub = {}
    for f in sorted(os.listdir(arr_dir)):
        if f.endswith(".npz") and not f.endswith(".tmp.npz") and "__" in f:
            by_sub.setdefault(f.split("__")[0], []).append(f)
    label_rows, prow = [], []
    if pictures:
        os.makedirs(os.path.join(out_dir, "pictures"), exist_ok=True)
    for sub, files in sorted(by_sub.items()):
        seizures, labelled, shown, pair_norm, panels = [], set(), {}, {}, []
        for f in files:
            try:
                arr = soz_lib.load_arrays(os.path.join(arr_dir, f))
                norms = np.asarray(arr["avg_norm"]).astype(str)
                bad = np.asarray(arr["avg_bad"], bool)
                t = arr["t"] - arr["onset"]
                d = car_line_length_onset(arr["avg_feat"][0], t, bad)
            except Exception as err:  # noqa: BLE001 - one broken file must not stop the others
                print(f"  {f}: could not be used ({type(err).__name__}: {err}) - skipped")
                continue
            shown.update(dict(zip(norms, np.asarray(arr["avg_names"]).astype(str))))
            pair_norm.update(dict(zip(np.asarray(arr["bip_names"]).astype(str), np.asarray(arr["bip_norm"]).astype(str))))
            labelled |= {c for c, b in zip(norms, bad) if not b}
            seizures.append({"name": f.split("__", 1)[1][:-4],
                             "onset": None if d["flags"] else {c for c, o in zip(norms, d["onset"]) if o},
                             "active": {c for c, a in zip(norms, d["active"]) if a},
                             "noisy": {c for c, x in zip(norms, d["noisy"]) if x}, "flags": "; ".join(d["flags"])})
            if pictures:
                panels.append({"name": seizures[-1]["name"], "t": t, "bad": bad, "norms": norms, "d": d})
        if not seizures:
            continue
        lab = v7_patient(seizures, labelled)
        soz = lab["soz"]
        # side / lobe of each contact: from step 3's bipolar sites (the two contacts of a site share its side / lobe)
        where = {}
        if side_tab is not None:
            for r in side_tab[side_tab["subject"] == sub].itertuples():
                for c in str(pair_norm.get(str(r.contact), "")).split("-"):
                    if c:
                        where.setdefault(c, (text_or_blank(r.mri_side), text_or_blank(r.mri_lobe)))
        side = majority([where.get(c, ("", ""))[0] for c in soz])
        lobe = majority([where.get(c, ("", ""))[1] for c in soz], need=0.5)
        els = sorted({electrode(c) for c in soz})
        clin_s = clin_l = ""
        if cc is not None and (cc["subject"] == sub).any():
            row = cc[cc["subject"] == sub].iloc[0]
            clin_s = clinical_side(row.get("clinical_lateralization", ""))
            clin_l = clinical_lobe(row.get("clinical_localization", ""))
        s61 = set()
        if lab61 is not None:
            s61 = contacts_of(lab61.loc[(lab61["subject"] == sub) & (lab61["soz"] == 1), "contact_norm"])
        els61 = sorted({electrode(c) for c in s61})
        side61 = majority([where.get(c, ("", ""))[0] for c in s61])
        q61, tr61 = "", False
        if summ61 is not None and (summ61["subject"] == sub).any():
            row = summ61[summ61["subject"] == sub].iloc[0]
            q61, tr61 = str(row["label_quality"]), bool(row["use_for_training"])
        tr61 = tr61 and sub not in excluded and (trained_step3 is None or sub in trained_step3)
        doubtful = bool(unloc is not None and ((unloc["subject"] == sub) & (unloc["result"] == "DOUBTFUL")).any())
        # the verdict (fixed rules, the same for every patient)
        why = []
        if lab["quality"] not in ("ok", "few seizures"):
            why.append(f"v7 label quality: {lab['quality']}")
        if soz and lab["best"] < 0.5 * lab["used"]:
            why.append(f"not regular: the most regular SOZ contact starts only {lab['best']} of the {lab['used']} usable "
                       f"seizures (needs half)" + (f" - random onsets would also give about {lab['chance']:.0f} 'SOZ' "
                                                    f"contacts" if lab["chance"] >= 1 else ""))
        if soz and side in ("both", ""):
            why.append("v7 SOZ on both sides" if side == "both" else "v7 SOZ side unknown")
        if soz and clin_s and side in ("L", "R") and side != clin_s:
            why.append(f"v7 SOZ on the {'left' if side == 'L' else 'right'}, the clinical side is "
                       f"{'left' if clin_s == 'L' else 'right'}")
        v7_ok = not why
        shared = sorted(set(els) & set(els61))
        if sub in excluded:
            verdict = "LEAVE OUT (listed in EXCLUDE_FROM_TRAINING)"
        elif tr61:
            if v7_ok and shared and (side61 in ("", "both") or side == side61):
                verdict = f"TRAIN - v7 agrees (same electrode: {' '.join(shared)})"
            elif v7_ok:
                verdict = "TRAIN (version 6.1 labels) - v7 finds a DIFFERENT SOZ: look at the picture"
                why.append(f"v7 SOZ on {' '.join(els)}, version 6.1 SOZ on {' '.join(els61)}")
            else:
                verdict = "TRAIN (version 6.1 labels) - v7 does not confirm"
        elif v7_ok:
            verdict = "CANDIDATE - only v7 finds a SOZ: look at the picture before adding it"
        else:
            verdict = "LEAVE OUT"
        if doubtful:
            verdict += "  [coordinates DOUBTFUL - tools/check_unlocalized.py]"
        jac = len(soz & s61) / len(soz | s61) if (soz | s61) else np.nan
        notes = []                                  # worth a look, but they do not change the verdict
        if soz and lobe and clin_l and lobe != clin_l:
            notes.append(f"v7 SOZ mostly {lobe}, the clinical record says {clin_l}")
        if shared and jac < 0.2:
            notes.append(f"v7 and version 6.1 pick different contacts on these electrodes (overlap {jac:.2f})")
        prow.append({"subject": sub, "seizures": len(seizures), "v7_usable": lab["used"], "v7_quality": lab["quality"],
                     "v7_ok": v7_ok, "v7_soz": len(soz), "v7_uncertain": len(lab["unc"]),
                     "v7_not_soz": len(labelled - soz - set(lab["unc"])),
                     "v7_soz_contacts": ", ".join(shown.get(c, c) for c in sorted(soz)), "v7_electrodes": " ".join(els),
                     "v7_best_onsets": lab["best"], "v7_soz_by_chance": lab["chance"],
                     "v7_side": side, "v7_lobe": lobe, "v7_leads": lab["led"], "clinical_side": clin_s,
                     "clinical_lobe": clin_l, "v61_quality": q61, "v61_trained": tr61, "v61_soz_contacts": len(s61),
                     "v61_electrodes": " ".join(els61), "same_contacts": jac, "shared_electrodes": " ".join(shared),
                     "coordinates_doubtful": doubtful, "verdict": verdict, "why": "; ".join(why),
                     "notes": "; ".join(notes),
                     "seizure_flags": " | ".join(f"{s['name']}: {s['flags'] or 'used'}" for s in seizures)})
        for c in sorted(labelled):
            label_rows.append({"subject": sub, "contact": shown.get(c, c), "contact_norm": c, "soz": int(c in soz),
                               "uncertain": int(c in lab["unc"]), "reason": lab["unc"].get(c, ""),
                               "n_onset": lab["counts"].get(c, 0), "usable_seizures": lab["used"],
                               "side": where.get(c, ("", ""))[0], "lobe": where.get(c, ("", ""))[1],
                               "version61_soz": int(c in s61)})
        if pictures and panels:
            order = [i for i, s in enumerate(seizures) if s["onset"] is not None] + \
                    [i for i, s in enumerate(seizures) if s["onset"] is None]
            chosen = [dict(panels[i], lab=lab) for i in order[:4]]
            try:
                picture(sub, chosen, s61, os.path.join(out_dir, "pictures", f"{sub}.png"))
            except Exception as err:  # noqa: BLE001 - a picture problem must not stop the report
                print(f"  {sub}: picture failed ({type(err).__name__}: {err})")
    return pd.DataFrame(prow), pd.DataFrame(label_rows)


# =========================================================================== report
def main():
    ap = argparse.ArgumentParser(description="Version 7 trial: seizure onset from line length on the common average.")
    ap.add_argument("--no-benchmark", action="store_true", help="skip the expert test")
    ap.add_argument("--no-pictures", action="store_true", help="do not draw the pictures")
    args = ap.parse_args()
    res = config.SOZ_RESULTS_DIR
    arr_dir = os.path.join(res, "seizure_arrays")
    if not os.path.isdir(arr_dir):
        raise SystemExit("Run step 2 of version 6.1 first (SOZ\\02_compute_soz.py): this trial uses the seizure_arrays "
                         "it saved.")
    out_dir = os.path.join(res, "v7")
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    def f2(v):
        return "-" if v is None or not np.isfinite(v) else f"{v:.2f}"

    say("=" * 112)
    say(f"VERSION 7 TRIAL ({VERSION}) - seizure onset from LINE LENGTH on the COMMON AVERAGE reference only")
    say("=" * 112)

    # ---------------------------------------------------------------- 1. the expert test
    choice_path = os.path.join(res, "benchmark_choice.json")
    if args.no_benchmark:
        say("\n1. EXPERT TEST: skipped (--no-benchmark)")
    elif not os.path.exists(choice_path):
        say("\n1. EXPERT TEST: skipped (no benchmark_choice.json - run tools\\benchmark_onset.py first)")
    else:
        with open(choice_path) as fh:
            ch = json.load(fh)
        params61 = soz_lib.method_params(ch["method"], ch.get("params"))
        print("Expert test on the benchmark seizures ...")
        per, pat = benchmark_test(params61)
        if per is None or not len(per):
            say("\n1. EXPERT TEST: skipped (no benchmark seizures found in CACHE_DIR\\benchmark\\arrays)")
        else:
            per.to_csv(os.path.join(out_dir, "v7_benchmark.csv"), index=False)
            n61 = f"version 6.1: {soz_lib.METHOD_NAMES.get(ch['method'], ch['method'])}, {ch.get('montage', 'bipolar')}"
            say(f"\n1. EXPERT TEST - {len(per)} seizures ({per['patient'].nunique()} patients) whose onset contacts were "
                f"marked by epileptologists")
            say("   Scored on contacts (a bipolar pair = its two contacts). F1: 1 = exactly the experts' contacts, "
                "0 = none of them.")
            say("   (+-1): a contact right next to an expert contact (same electrode) also counts.\n")
            say(f"   {'one seizure at a time':<46}{'F1':>8}{'(+-1)':>9}{'seizures answered':>20}")
            for name, text in (("v7", "v7: line length, common average"), ("v61", n61)):
                ans = int((per[f"{name}_flags"] == "").sum())
                say(f"   {text:<46}{per[f'{name}_f1'].mean():>8.2f}{per[f'{name}_f1_pm1'].mean():>9.2f}"
                    f"{f'{ans} of {len(per)}':>20}")
            if len(pat):
                n_pat = pat["patient"].nunique()
                say(f"\n   The PATIENT labels (the rule of step 2) of the {n_pat} patients with >= 2 seizures - what the "
                    f"model would learn from.\n   A patient is used only when its label quality is 'ok' / 'few "
                    f"seizures'; the numbers are over the patients used:")
                say(f"   {'':<44}{'patients':>9}{'SOZ F1':>13}{'precision':>13}{'recall':>13}{'real onset contacts':>21}")
                say(f"   {'':<44}{'used':>9}{'':>13}{'':>13}{'':>13}{'labelled NOT SOZ':>21}")
                for name, text in (("v7", "v7: line length, common average"), ("v61", n61)):
                    g = pat[(pat["method"] == name) & pat["quality"].isin(["ok", "few seizures"])]
                    g0, g1 = g[~g["tol"]], g[g["tol"]]

                    def cell(col, fmt, g0=g0, g1=g1):
                        return "-" if g0.empty else f"{format(g0[col].mean(), fmt)} ({format(g1[col].mean(), fmt)})"
                    used = f"{g0['patient'].nunique()} of {n_pat}"
                    say(f"   {text:<44}{used:>9}{cell('f1', '.2f'):>13}{cell('precision', '.2f'):>13}"
                        f"{cell('recall', '.2f'):>13}{cell('harmful', '.0%'):>21}")
                say("   precision = share of the SOZ contacts that are real onset contacts (the experts' usual ones); "
                    "recall = share of the\n   real onset contacts that get the SOZ label; the last column is the "
                    "dangerous error for training: a real onset\n   contact taught to the model as 'healthy' (lower is "
                    "better). In brackets: +-1 contact. Both methods were tuned on these\n   seizures, so both look a "
                    "little better here than on new patients (v7 leave-one-patient-out: about 0.36 per seizure).")

    # ---------------------------------------------------------------- 2. your patients
    print("Your patients ...")
    pt, labels = your_patients(res, arr_dir, out_dir, not args.no_pictures)
    labels.to_csv(os.path.join(out_dir, "v7_labels.csv"), index=False)
    pt.to_csv(os.path.join(out_dir, "v7_patients.csv"), index=False)
    if pt.empty:
        say("\n2. YOUR PATIENTS: no saved seizures found")
    else:
        say(f"\n2. YOUR PATIENTS ({len(pt)} with seizure recordings)\n")
        show = pd.DataFrame({
            "patient": pt["subject"].str.replace("sub-", "", regex=False), "seizures": pt["seizures"],
            "v7 used": pt["v7_usable"], "v7 quality": pt["v7_quality"], "v7 SOZ": pt["v7_soz"],
            "best": [f"{b}/{u}" if s else "-" for b, u, s in zip(pt["v7_best_onsets"], pt["v7_usable"], pt["v7_soz"])],
            "v7 electrodes": pt["v7_electrodes"].str.slice(0, 20), "side": pt["v7_side"].replace("", "-"),
            "leads": pt["v7_leads"].map(f2),
            "clinical": pt["clinical_side"].replace("", "?") + " " + pt["clinical_lobe"].replace("", "?"),
            "6.1 SOZ": pt["v61_soz_contacts"], "6.1 electrodes": pt["v61_electrodes"].str.slice(0, 20),
            "same": pt["same_contacts"].map(f2)})
        say(show.to_string(index=False))
        say("\n   v7 SOZ / 6.1 SOZ = number of SOZ contacts (version 6.1: the contacts of its SOZ pairs); best = usable "
            "seizures started by\n   the most regular SOZ contact; leads = share of the usable seizures that start "
            "at the SOZ (or next to it);\n   same = overlap of the two SOZ (1 = identical, 0 = no contact in common).")
        say("\n   VERDICT - the same rules for every patient: v7 label quality 'ok' / 'few seizures' (as in step 2), the "
            "most regular SOZ\n   contact starts at least half of the usable seizures, the SOZ is on one side, and "
            "that is the clinical side:")
        for r in pt.itertuples():
            say(f"   {r.subject}: {r.verdict}")
            for label, text in (("why: ", r.why), ("note:", r.notes)):
                for k, reason in enumerate(str(text).split("; ") if text else []):
                    say(f"        {label if k == 0 else '     '} {reason}")
        n_train = int(pt["v61_trained"].sum())
        n_agree = int(pt["verdict"].str.startswith("TRAIN - v7 agrees").sum())
        cand = list(pt.loc[pt["verdict"].str.startswith("CANDIDATE"), "subject"])
        say(f"\n   v7 labels look trustworthy for {int(pt['v7_ok'].sum())} of {len(pt)} patients; version 6.1 trains "
            f"{n_train}, and v7 agrees with it for {n_agree} of them.")
        say(f"   NEW patients that only v7 can label: {len(cand)}{' (' + ', '.join(cand) + ')' if cand else ''}.")
        if not args.no_pictures:
            say(f"   Pictures: {os.path.join(out_dir, 'pictures')}")
    say(f"\nSaved v7_report.txt, v7_patients.csv, v7_labels.csv, v7_benchmark.csv in {out_dir}")
    say("Nothing of version 6.1 was changed. Send v7_report.txt (and the pictures folder, zipped) to Claude.")
    with open(os.path.join(out_dir, "v7_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    with runlog.Log("v7_car_linelength"):
        main()
