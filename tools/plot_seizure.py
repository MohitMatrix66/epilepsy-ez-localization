"""
tools/plot_seizure.py  -  look at the raw EEG of a seizure: does the seizure really start where step 2 says?

Run from the project folder (after SOZ\\02_compute_soz.py):
    python tools\\plot_seizure.py --audit                                  one seizure of EVERY patient (the check
                                                                          before training - zip the folder
                                                                          outputs\\seizure_plots\\audit and send it)
    python tools\\plot_seizure.py --subject sub-PID04                      its first 3 seizures
    python tools\\plot_seizure.py --subject sub-PID04 --seizure ictal399391_run-01 --sites 20
    python tools\\plot_seizure.py --subject sub-PID04 --all                every seizure of the patient
Output: outputs/seizure_plots/<subject>_<seizure>.png  (uses the recordings kept in CACHE_DIR)

Each row is one site of the montage used for the labels (bipolar pair or contact), filtered 1 Hz high-pass +
power-line notch, scaled by its own normal period (clipped at 4.5 x its baseline spread). Rows: the onset sites of
the seizure, then the sites the version 5 EI liked best (for comparison), then the next sites to start.
Filled triangle = start found by the chosen method; a small grey cross = already active before the onset;
open grey triangle = start of the version 5 EI. After the name: which detector made it an onset site in this
seizure (EI, LL = line length, or both). Blue = SOZ site, orange = uncertain site, black = other site.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SOZ"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import aws_data  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402

INK, MUTED, SOZ_C, UNC_C, GRID = "#1f1f1e", "#8a8984", "#256abf", "#c45f1c", "#e1e0d9"
BEFORE, AFTER = 15.0, 20.0     # seconds shown around the marked onset


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def pick_sites(tab, n_sites):
    """Onset sites first (by start), then the EI's favourites (for comparison), then the next sites to start."""
    tab = tab[~tab["bad_channel"].astype(bool)]
    onset = tab[tab["onset_site"].astype(bool)].sort_values("start_s")
    nxt = tab[~tab["onset_site"].astype(bool) & tab["start_s"].notna()].sort_values("start_s")
    ei = tab.sort_values("ei_v5", ascending=False)
    ei = ei[ei["ei_v5"].fillna(0) >= 0.3]
    order = list(dict.fromkeys(list(onset["contact"]) + list(ei["contact"].head(4)) + list(nxt["contact"])))
    return tab.set_index("contact").loc[order[:n_sites]].reset_index()


def plot_one(sub, seizure, rec, info, per, labels, montage, n_sites, out_dir):
    import mne
    tab = per[(per["seizure"] == seizure) & (per["montage"] == montage)]
    if tab.empty:
        print(f"  {seizure}: no results - nothing to plot")
        return None
    tab = pick_sites(tab, n_sites)
    if tab.empty:
        print(f"  {seizure}: no site started - nothing to plot")
        return None
    onset = float(info["onset"])
    arr_path = os.path.join(config.SOZ_RESULTS_DIR, "seizure_arrays", f"{sub}__{seizure}.npz")
    arr = soz_lib.load_arrays(arr_path) if os.path.exists(arr_path) else None
    raw = mne.io.read_raw_edf(aws_data.fetch(rec.edf), preload=False, verbose="error")
    fs = float(raw.info["sfreq"])
    index = {soz_lib.display_name(n): i for i, n in enumerate(raw.ch_names)}
    t0, t1 = max(0.0, onset - BEFORE - 15.0), min(raw.times[-1], onset + AFTER)
    raw.crop(tmin=t0, tmax=t1)
    sidecar = soz_lib.read_json(aws_data.fetch(rec.json)) if rec.json else {}
    if montage == "average":
        if arr is None:
            print(f"  {seizure}: saved arrays missing - run SOZ\\02_compute_soz.py again")
            return None
        names = [n for n, b in zip(arr["avg_names"], arr["avg_bad"]) if n in index]
        good = [n for n, b in zip(arr["avg_names"], arr["avg_bad"]) if n in index and not b]
        picks = [index[n] for n in names]
        x = soz_lib.preprocess(raw.get_data(picks=picks), fs, soz_lib.power_line_frequency(sidecar))
        pos = {n: k for k, n in enumerate(names)}
        avg = x[[pos[n] for n in good]].mean(axis=0)
        sig = {n: x[pos[n]] - avg for n in names}
    else:
        need = sorted({p for site in tab["contact"] for p in str(site).split("-") if p in index})
        x = soz_lib.preprocess(raw.get_data(picks=[index[n] for n in need]), fs, soz_lib.power_line_frequency(sidecar))
        pos = {n: k for k, n in enumerate(need)}
        sig = {}
        for site in tab["contact"]:
            a, _, b = str(site).partition("-")
            if a in pos and b in pos:
                sig[site] = x[pos[a]] - x[pos[b]]
    t = np.arange(x.shape[1]) / fs + t0 - onset
    base = (t >= -30) & (t < -7)
    if base.sum() < fs:
        base = t < -1.0
    show = (t >= -BEFORE)
    soz = set(labels.loc[labels["soz"] == 1, "contact"])
    unc = set(labels.loc[labels["uncertain"].fillna(0) == 1, "contact"])
    rows = [r for r in tab.itertuples() if r.contact in sig]
    height = 0.42 * len(rows) + 2.3
    fig, ax = plt.subplots(figsize=(14, height))
    for k, r in enumerate(rows):
        y = sig[r.contact]
        y = y / (np.std(y[base]) + 1e-12)
        colour = SOZ_C if r.contact in soz else UNC_C if r.contact in unc else INK
        ax.plot(t[show], -k * 12 + np.clip(y[show], -4.5, 4.5), color=colour, lw=0.45)
        active = bool(getattr(r, "edge", False)) and not bool(r.onset_site)
        if np.isfinite(r.start_s) and not active:
            ax.plot(max(r.start_s, -BEFORE), -k * 12 + 6.2, marker="v", ms=8, mfc=colour, mec="white", mew=0.8,
                    zorder=5)
        elif np.isfinite(r.start_s):
            ax.plot(max(r.start_s, -BEFORE), -k * 12 + 6.2, marker="x", ms=6, color=MUTED, mew=1.2, zorder=5)
        if np.isfinite(getattr(r, "ei_v5_start_s", np.nan)) and r.ei_v5 >= 0.3:
            ax.plot(max(r.ei_v5_start_s, -BEFORE), -k * 12 + 6.2, marker="v", ms=7, mfc="none", mec=MUTED, mew=1.0,
                    zorder=4)
        by = "+".join(n for n, f in (("EI", "ei_onset"), ("LL", "ll_onset")) if bool(getattr(r, f, False)))
        lab = (f"{r.contact}  {by + ' ' if by else ''}score {r.oi:.2f}"
               + (f" | v5 EI {r.ei_v5:.2f}" if np.isfinite(r.ei_v5) else ""))
        ax.text(-BEFORE - 0.3, -k * 12, lab, ha="right", va="center", fontsize=8, color=colour)
    flags = str(info.get(f"{montage}_flags", "") or "")
    if flags == "nan":
        flags = ""
    for xv, ls, lab in ((0.0, "-", "0 s: marked onset"),):
        ax.axvline(xv, color=INK, ls=ls, lw=1.2, label=lab)
    for xv in range(int(-BEFORE), int(AFTER) + 1, 5):
        ax.axvline(xv, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(-BEFORE, t[-1])
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("seconds from the marked seizure onset")
    state = f"NOT USED FOR LABELS: {flags}" if flags else "usable seizure"
    ax.set_title(f"{sub}  {seizure}   ({montage}; {state})\nfilled triangle = start (after the name: EI / LL = which "
                 f"detector made it an onset site in this seizure);  grey x = already active before the onset\n"
                 f"open grey triangle = start of the version 5 EI;  blue = SOZ, orange = uncertain", loc="left",
                 fontsize=9.5, color=INK)
    fig.subplots_adjust(left=0.22, right=0.98, top=1 - 1.25 / height, bottom=0.9 / height)
    path = os.path.join(out_dir, f"{sub}_{seizure}.png")
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    print(f"  {seizure}: saved {path}")
    return path


def audit_choice(sub, info, per, labels, montage):
    """The seizure to show for a patient: a usable seizure in which the SOZ starts (else the first usable, else the
    first seizure)."""
    inf = info[info["subject"] == sub]
    usable = list(inf.loc[inf.get(f"{montage}_usable", pd.Series(False, index=inf.index)).astype(bool), "seizure"])
    soz = set(labels.loc[(labels["subject"] == sub) & (labels["soz"] == 1), "contact"])
    for s in usable:
        t = per[(per["subject"] == sub) & (per["seizure"] == s) & (per["montage"] == montage) & per["onset_site"]]
        if soz & set(t["contact"]):
            return s
    if usable:
        return usable[0]
    return inf["seizure"].iloc[0] if len(inf) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None)
    ap.add_argument("--audit", action="store_true", help="one seizure of every patient, for the check before training")
    ap.add_argument("--seizure", nargs="*", default=None, help="seizure name(s) as in seizure_info.csv")
    ap.add_argument("--all", action="store_true", help="every seizure of the patient")
    ap.add_argument("--sites", type=int, default=16, help="number of sites to show (default 16)")
    args = ap.parse_args()
    if not args.audit and not args.subject:
        raise SystemExit("Give --audit or --subject, e.g.   python tools\\plot_seizure.py --audit")
    res = config.SOZ_RESULTS_DIR
    per = pd.read_csv(os.path.join(res, "soz_onset_per_seizure.csv"))
    info = pd.read_csv(os.path.join(res, "seizure_info.csv"))
    labels = pd.read_csv(os.path.join(res, "soz_labels.csv"))
    summary = pd.read_csv(os.path.join(res, "soz_summary.csv"))
    montage = str(summary["montage"].iloc[0]) if len(summary) else "bipolar"
    recs = pd.read_csv(os.path.join(res, "seizure_recordings.csv"), keep_default_na=False, dtype={"run": str})
    recs = recs[recs["kind"] == "ictal"]
    by_name = {(r.subject, f"{safe_name(r.task)}_run-{r.run}"): r for r in recs.itertuples()}
    out_dir = os.path.join(config.OUT_DIR, "seizure_plots", "audit" if args.audit else "")
    os.makedirs(out_dir, exist_ok=True)
    if args.audit:
        subs = sorted(summary["subject"])
        print(f"Audit: one seizure of each of {len(subs)} patients -> {out_dir}")
        for sub in subs:
            s = audit_choice(sub, info, per, labels, montage)
            if s is None or (sub, s) not in by_name:
                print(f"  {sub}: no seizure to show")
                continue
            print(f"{sub}:")
            plot_one(sub, s, by_name[(sub, s)], info[(info["subject"] == sub) & (info["seizure"] == s)].iloc[0],
                     per[per["subject"] == sub], labels[labels["subject"] == sub], montage, args.sites, out_dir)
        print(f"\nDone. Please zip the folder {out_dir} and send it to Claude (one picture per patient).")
        return
    sub = args.subject
    inf = info[info["subject"] == sub].set_index("seizure")
    if inf.empty:
        raise SystemExit(f"{sub}: no results in {res} - run SOZ\\02_compute_soz.py first.")
    names = sorted(inf.index)
    chosen = args.seizure or (names if args.all else names[:3])
    missing = [s for s in chosen if (sub, s) not in by_name or s not in inf.index]
    if missing:
        raise SystemExit(f"Unknown seizure(s) {', '.join(missing)}; this patient has: {', '.join(names)}")
    print(f"{sub}: plotting {len(chosen)} of {len(names)} seizures")
    for s in chosen:
        plot_one(sub, s, by_name[(sub, s)], inf.loc[s], per[per["subject"] == sub], labels[labels["subject"] == sub],
                 montage, args.sites, out_dir)


if __name__ == "__main__":
    with runlog.Log("plot_seizure"):
        main()
