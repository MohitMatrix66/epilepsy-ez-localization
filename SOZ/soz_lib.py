"""
soz_lib.py  -  building blocks used by the SOZ scripts (version 6.1):
    * reading the dataset's tables (channels.tsv, events.tsv, ieeg.json)
    * choosing which recorded channels are intracranial contacts (scalp EEG and ECG channels are ignored)
      and matching names that are written differently in different files ("LA01" = "LA1")
    * one seizure recording -> saved arrays (analyse_recording / analyse_signals): for every contact (common
      average reference) and every pair of neighbouring contacts on the same electrode (bipolar):
          - the 8 onset features over time (onset_lib.py: line length and the 8-feature detector), and
          - the energy ratio over time (for the Epileptogenicity Index, EI)
    * onset_tables(): where and when each seizure starts, with the method chosen by the expert benchmark:
          "ei+linelength" (version 6.1)  EI and line length TOGETHER: an onset site of either is an onset site
          "ei", "linelength", "multifeature"  one detector alone
    * ei_onset(): the EI with the version 6.1 safety rules (below); ei_tables(): the EI of version 5, unchanged
    * soz_sites(): from the per-seizure onset sites of a patient to its SOZ / uncertain sites
    * reading the electrode coordinates made by iEEG-recon and checking they fit the MRI

The EI in plain words (Bartolomei, Chauvel & Wendling, Brain 2008):
    For every site the "energy ratio" ER(t) = (beta + gamma energy) / (theta + alpha energy) is followed every
    0.25 s (1-s windows); a change-point detector (Page-Hinkley) finds WHEN it jumps up; sites that jump early and
    strongly get a high EI = (ER summed over the 5 s after the jump) / (delay after the earliest site + 1 s),
    scaled so the best site = 1. It sees low-voltage fast activity well, but fading slow waves raise the ratio
    without any seizure (sub-PID04, sub-PID27) and rhythmic slow onsets lower it (sub-PID25).
    Version 6.1 safety rules (ei_onset): a jump found before EI_EARLY_S (-5 s; no epileptologist onset on the
    benchmark was earlier than -2.4 s) or right at the start of the search is "already active" - not an onset;
    when that is true for EI_ACTIVE_SHARE (20%) or more of the sites that jump, the seizure had a state change
    and the EI is not used for it; the search ends EI_SEARCH_END_S (+10 s) after the marked onset.

From the onsets of every seizure to the SOZ of a patient (soz_sites, rule "repeated"):
    a site is SOZ when it is an onset site in at least SOZ_MIN_ONSETS (2) usable seizures AND in at least
    SOZ_MIN_SHARE (25%) of them - where the seizures USUALLY start; several foci in one patient all count.
    A site that is an onset site less often, or that was already active before the onset, is "uncertain": it is
    used neither as SOZ nor as "not SOZ". A patient is trusted when the SOZ (or a site next to it) is among the
    onset sites in at least SOZ_MIN_LED (30%) of the usable seizures.
This is still an automatic ESTIMATE of the clinical SOZ, not a clinician's reading.
"""
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage, signal

import onset_lib

VERSION = 6          # version of the saved per-seizure arrays

# --------------------------------------------------------------------------- EI parameters
BANDS = {"theta": (3.5, 7.4), "alpha": (7.4, 12.4), "beta": (12.4, 24.0), "gamma": (24.0, 97.0)}
WIN_S = 1.0          # spectral window length (s)
STEP_S = 0.25        # step between windows (s)
PH_BIAS = 1.0        # Page-Hinkley bias, in units of the site's own baseline energy ratio
PH_LAMBDA = 15.0     # Page-Hinkley alarm threshold (same units, summed over steps)
SUSTAIN_RATIO = 2.0  # a jump counts only if the energy ratio then STAYS >= this x its baseline:
CONFIRM_S = 1.0      #   - jump before the marked onset: in 75% of the time until CONFIRM_S after the onset
MIN_RUN_S = 2.0      #   - jump at / after the marked onset: for most (median) of the next MIN_RUN_S seconds
TAU_S = 1.0          # EI delay constant (s)
H_S = 5.0            # EI integration window after the detected change (s)
GUARD_S = 5.0        # the baseline ends at least this many seconds before the marked onset, and always 2 s
                     # before the search window starts (EI_LEAD_S + 2 s)
STRONG_EI = 0.7      # a "strong" onset site in one seizure (used by the multifocal SOZ rule)
MAX_PAIR_MM = 15.0   # neighbouring contacts further apart than this are not paired (e.g. grid row ends)
# version 6.1 EI (ei_onset)
EI_EARLY_S = -5.0         # a jump before this time (s from the marked onset) is "already active", not an onset
EI_ACTIVE_SHARE = 0.2     # ... and when this share of the jumping sites is already active, the EI is not used
EI_SEARCH_END_S = 10.0    # the EI search ends this long after the marked onset
EI_DEFAULTS = {"lead_s": 10.0, "threshold": 0.3, "early_s": EI_EARLY_S, "active_share": EI_ACTIVE_SHARE,
               "post_s": EI_SEARCH_END_S, "pre_s": 30.0}
METHOD_NAMES = {"ei+linelength": "EI + line length", "ei": "EI", "linelength": "line length",
                "multifeature": "8-feature (Imprint-style)"}

INTRACRANIAL_TYPES = {"SEEG", "ECOG", "DEPTH", "GRID", "STRIP", "IEEG"}
UNKNOWN_TYPES = {"", "N/A", "NA", "NAN", "NONE", "UNKNOWN"}
NON_NEURAL = re.compile(r"^(EKG|ECG|EMG|EOG|DC|TRIG|PHOTIC|RESP|SPO2|PULSE|EVENT|STATUS|MARK|OSAT|PR|PLETH)",
                        re.I)
# standard scalp electrode names (10-20 / 10-10 system). Only used for channels whose TYPE is EEG.
SCALP_NAMES = re.compile(r"^(FP[12Z]|AF[3478Z]|F[1-8Z]|F9|F10|FC[1-6Z]|FT[78]|FT9|FT10|C[1-6Z]|T[3-8]|T9|T10|"
                         r"TP[78]|TP9|TP10|CP[1-6Z]|P[1-8Z]|P9|P10|PO[3478Z]|O[12Z]|IZ|A[12]|M[12]|LOC|ROC)$")


# --------------------------------------------------------------------------- names
_PREFIX = re.compile(r"^(EEG|SEEG|ECOG|IEEG|POL)[\s_\-]+", re.I)
_SUFFIX = re.compile(r"[\s_\-]*(REF|REFERENCE|AVG|CAR)$", re.I)
_NUMBERED = re.compile(r"^([A-Z']*[A-Z'])0*(\d+)$")


def display_name(name):
    """Recorded name without 'EEG ' / '-Ref' decorations (kept as written otherwise)."""
    return _SUFFIX.sub("", _PREFIX.sub("", str(name).replace("\ufeff", "").strip().strip('"')))


def normalize_name(name):
    """'EEG LA01-Ref' -> 'LA1'. Used only to MATCH names between files; original names are kept."""
    s = display_name(name).upper()
    s = re.sub(r"[\s_\-]", "", s)
    m = _NUMBERED.match(s)
    return m.group(1) + m.group(2) if m else s


def split_numbered(norm):
    """'LA12' -> ('LA', 12); None for names without a trailing contact number."""
    m = _NUMBERED.match(norm)
    return (m.group(1), int(m.group(2))) if m else None


def apply_alias(norm, aliases):
    """Electrode-name fixes for one patient, e.g. {'G': 'LTG'} turns 'G12' into 'LTG12'."""
    if not aliases:
        return norm
    m = _NUMBERED.match(norm)
    if m and m.group(1) in aliases:
        return aliases[m.group(1)] + m.group(2)
    return aliases.get(norm, norm)


def hemisphere_from_name(name):
    """'L' / 'R' from names like 'LA3' or 'LA01-LA02'; '' when the name has no side prefix."""
    first = str(name).split("-")[0]
    m = re.match(r"^([LR])[A-Z]", normalize_name(first))
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- tables
def read_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_json(path):
    try:
        with open(path, encoding="utf-8-sig") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 - a missing or broken sidecar must not stop the analysis
        return {}


_SOZ_TEXT = re.compile(r"(?:\bsoz\b|seizure[\s_-]*onset|onset[\s_-]*zone)", re.I)
_TRUE = {"1", "true", "yes", "y", "soz"}


def clinical_soz_marks(channels):
    """Contacts a clinician marked as seizure onset zone in a channels.tsv table (if any).
    Returns {normalized name: 'column=value'}."""
    marks = {}
    if "name" not in channels.columns:
        return marks
    for col in channels.columns:
        if col == "name":
            continue
        is_soz_col = bool(_SOZ_TEXT.search(col))
        for name, val in zip(channels["name"], channels[col].astype(str)):
            v = val.strip()
            if (is_soz_col and v.lower() in _TRUE) or (not is_soz_col and _SOZ_TEXT.search(v)):
                marks[normalize_name(name)] = f"{col}={v}"
    return marks


_ONSET_WORDS = re.compile(r"(?:seizure|onset|ictal|\bsz\b|\beec\b|\bueo\b)", re.I)


def find_onset(events):
    """Seizure onset time (s from the start of the recording) from an events.tsv table.
    Returns (onset or None, description of where it came from)."""
    if events is None or events.empty or "onset" not in events.columns:
        return None, "no events in events.tsv"
    onset = pd.to_numeric(events["onset"], errors="coerce")
    text_cols = [c for c in events.columns if c not in ("onset", "duration", "sample")]
    labels = events[text_cols].astype(str).agg(" ".join, axis=1) if text_cols else pd.Series([""] * len(events))
    hit = labels.str.contains(_ONSET_WORDS) & ~labels.str.contains(r"inter|pre-?ictal|post-?ictal", case=False)
    hit &= onset.notna()
    if hit.any():
        i = onset[hit].idxmin()
        return float(onset[i]), f"events.tsv '{labels[i].strip()}'"
    other = labels.str.contains(r"inter|pre-?ictal|post-?ictal", case=False)
    if onset.notna().sum() == 1 and not other[onset.first_valid_index()]:
        i = onset.first_valid_index()
        return float(onset[i]), f"events.tsv only event '{labels[i].strip()}'"
    return None, "events.tsv has no seizure-onset event"


def power_line_frequency(sidecar):
    try:
        f = float(sidecar.get("PowerLineFrequency", 60))
        return f if f in (50.0, 60.0) else 60.0
    except (TypeError, ValueError):
        return 60.0   # "n/a" in this dataset; the recordings are from the USA (60 Hz)


# --------------------------------------------------------------------------- channels
def channel_kind(name, typ):
    """'intracranial', 'scalp', 'eeg' (type EEG but not a scalp name), 'unknown' or 'other' (ECG, ...).
    A scalp channel is type EEG AND named exactly like a 10-20 electrode as written ('P3', 'Cz', 'O1').
    Zero-padded names such as 'P03' or 'O01' are intracranial contacts (some are typed EEG in this dataset)."""
    t = (typ or "").strip().upper()
    if t in INTRACRANIAL_TYPES:
        return "intracranial"
    if t == "EEG":
        return "scalp" if SCALP_NAMES.match(display_name(name).upper()) else "eeg"
    if t in UNKNOWN_TYPES:
        return "unknown"
    return "other"   # ECG, EMG, EOG, MISC, TRIG, ...


def select_channels(edf_names, channels, coord_names, aliases=None):
    """Which EDF channels to analyse: intracranial contacts that have electrode coordinates (best),
    otherwise intracranial channel types. Never channels marked 'bad', ECG/EMG/trigger channels or
    scalp EEG channels (type EEG named exactly like a 10-20 electrode, e.g. 'P3' - these would otherwise be
    mixed up with intracranial contacts called 'P03'). Electrode-name fixes (aliases) are used only for
    channels whose own name has no coordinates and whose fixed name has.
    Returns (indices, notes, details); details['norms'] = matching names of the chosen channels."""
    coord_names = set(coord_names or ())
    base = [normalize_name(n) for n in edf_names]

    def resolve(nm):
        if aliases and nm not in coord_names:
            fixed = apply_alias(nm, aliases)
            if fixed != nm and (fixed in coord_names or not coord_names):
                return fixed
        return nm

    norm = [resolve(n) for n in base]
    exact, by_norm = {}, {}
    if channels is not None and "name" in channels.columns:
        for _, r in channels.iterrows():
            info = (str(r.get("type", "")).strip().upper(), str(r.get("status", "")).strip().lower())
            exact[str(r["name"]).replace("\ufeff", "").strip()] = info
            by_norm.setdefault(normalize_name(r["name"]), []).append(info)
    have_types = any(t for t, _ in exact.values())
    kinds, bad = [], []
    for name, b in zip(edf_names, base):
        info = exact.get(str(name).strip())
        if info is None:
            cands = by_norm.get(b, [])
            info = cands[0] if len(cands) == 1 else ("", "")
        kinds.append(channel_kind(name, info[0]) if have_types else "unknown")
        bad.append(info[1] == "bad")
    usable = [i for i in range(len(norm)) if kinds[i] in ("intracranial", "eeg", "unknown") and not bad[i]]
    if coord_names and len({norm[i] for i in usable} & coord_names) >= 3:
        keep = [i for i in usable if norm[i] in coord_names]
        how = "contacts with electrode coordinates"
    elif any(k == "intracranial" for k in kinds):
        keep = [i for i in usable if kinds[i] == "intracranial"]
        how = "intracranial channel types in channels.tsv"
    else:
        keep = [i for i in usable if not NON_NEURAL.match(norm[i])]
        how = "all channels except EKG/EMG/trigger-like names"
    rank = {"intracranial": 0, "unknown": 1, "eeg": 2}
    best = {}
    for i in keep:  # one channel per name; an intracranial-typed channel wins
        if norm[i] not in best or rank[kinds[i]] < rank[kinds[best[norm[i]]]]:
            best[norm[i]] = i
    uniq = sorted(best.values())
    n_scalp = sum(k == "scalp" for k in kinds)
    unmatched = sorted({norm[i] for i in usable if kinds[i] == "intracranial" and coord_names
                        and norm[i] not in coord_names})
    n_alias = sum(norm[i] != base[i] for i in uniq)
    notes = (f"{how}; {sum(bad)} marked bad in channels.tsv"
             + (f"; {n_scalp} scalp EEG channels ignored" if n_scalp else "")
             + (f"; {n_alias} names fixed with CHANNEL_ALIASES" if n_alias else ""))
    return uniq, notes, {"n_scalp": n_scalp, "unmatched": unmatched, "norms": [norm[i] for i in uniq]}


def bipolar_pairs(norms, coords=None, max_mm=MAX_PAIR_MM):
    """Neighbouring contacts on the same electrode (contact numbers n and n+1): list of (i, j) indices into
    `norms`. With coordinates ({name: xyz}), pairs further apart than max_mm are dropped (e.g. the last
    contact of one grid row and the first of the next)."""
    pos = {}
    for k, n in enumerate(norms):
        s = split_numbered(n)
        if s:
            pos[s] = k
    pairs = []
    for (prefix, num), i in sorted(pos.items()):
        j = pos.get((prefix, num + 1))
        if j is None:
            continue
        if coords:
            a, b = coords.get(norms[i]), coords.get(norms[j])
            if a is not None and b is not None and float(np.linalg.norm(np.subtract(a, b))) > max_mm:
                continue
        pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------- signal processing
def preprocess(x, fs, line_freq):
    """High-pass 1 Hz and remove power-line noise (and its harmonics). x: channels x samples."""
    x = signal.sosfiltfilt(signal.butter(4, 1.0, btype="highpass", fs=fs, output="sos"), x, axis=-1)
    for f0 in (line_freq, 2 * line_freq, 3 * line_freq):
        if f0 < fs / 2 - 2:
            b, a = signal.iirnotch(f0, Q=30.0, fs=fs)
            x = signal.filtfilt(b, a, x, axis=-1)
    return x


def bad_channels(x, factor=10.0):
    """Flat channels, and channels whose amplitude is more than `factor` times larger or smaller than
    the typical channel (broken contacts, heavy artifacts). Seizure activity is far below this."""
    sd = x.std(axis=1)
    med = np.median(sd[sd > 0]) if (sd > 0).any() else 0.0
    if med == 0:
        return np.ones(len(sd), bool)
    return (sd <= med / factor) | (sd >= med * factor)


def energy_ratio(x, fs):
    """ER(t) for every channel. Returns (times in s from the start of x, ER: channels x times)."""
    nper = int(round(WIN_S * fs))
    step = max(1, int(round(STEP_S * fs)))
    f, t, S = signal.spectrogram(x, fs=fs, window="hann", nperseg=nper, noverlap=nper - step,
                                 detrend="constant", scaling="density", mode="psd", axis=-1)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return S[:, m, :].sum(axis=1) if m.any() else np.zeros((S.shape[0], S.shape[2]))

    gamma_hi = min(BANDS["gamma"][1], fs / 2.0 - 1.0)
    fast = band(*BANDS["beta"]) + band(BANDS["gamma"][0], gamma_hi)
    slow = band(*BANDS["theta"]) + band(*BANDS["alpha"])
    floor = 1e-12 * max(float(np.median(slow)), 1e-300)
    return t, fast / (slow + floor)


def page_hinkley(er, bias=PH_BIAS, lam=PH_LAMBDA):
    """Index where a lasting increase of er starts (running-mean version), or -1 if there is none.
    Only used to estimate an onset time when events.tsv has none."""
    n = len(er)
    if n < 2:
        return -1
    running_mean = np.cumsum(er) / np.arange(1, n + 1)
    U = np.concatenate([[0.0], np.cumsum(er - running_mean - bias)])
    u_min = np.minimum.accumulate(U)
    alarm = np.flatnonzero(U - u_min > lam)
    if not len(alarm):
        return -1
    a = alarm[0]
    return int(np.flatnonzero(U[:a + 1] == u_min[a])[-1])


def detect_change(ern, start, stop, mu0, onset_frame, confirm_frame, bias=PH_BIAS, lam=PH_LAMBDA,
                  sustain=SUSTAIN_RATIO):
    """First LASTING increase of a normalised energy ratio (1 = baseline median) that starts in frames
    start..stop-1, found with Page-Hinkley (CUSUM) against the baseline level mu0.
    A change that starts BEFORE the marked onset (frame onset_frame) counts only if the energy ratio stays
    >= sustain x baseline during at least 75% of the time until confirm_frame (1 s after the marked onset):
    it must carry on into the seizure. A change at or after the marked onset counts if the energy ratio
    stays >= sustain for most (median) of the next MIN_RUN_S seconds. Changes that fail (short bursts,
    e.g. spikes before the seizure) are skipped and the search goes on. Returns the frame index or -1."""
    run = int(round(MIN_RUN_S / STEP_S))
    s = int(start)
    while s < stop:
        inc = ern[s:stop] - mu0 - bias
        U = np.concatenate([[0.0], np.cumsum(inc)])
        u_min = np.minimum.accumulate(U)
        alarm = np.flatnonzero(U - u_min > lam)
        if not len(alarm):
            return -1
        a = int(alarm[0])
        j = s + int(np.flatnonzero(U[:a + 1] == u_min[a])[-1])
        if j < onset_frame:
            ok = np.percentile(ern[j:max(confirm_frame, j + 4)], 25) >= sustain
        else:
            ok = np.median(ern[j:j + run]) >= sustain
        if ok:
            return j
        s += a
    return -1


def epileptogenicity_index(er, t, onset, pre_s, post_s, lead_s):
    """EI of every channel for one seizure (version 5).
    er: channels x times (raw energy ratio), t: times (s, same clock as onset).
    Returns (ei 0..1, detection time relative to onset in s (nan = no detection), baseline ok flag)."""
    energy, det_rel, valid, baseline_ok = epileptogenicity_parts(er, t, onset, pre_s, post_s, lead_s)
    ei = np.zeros(er.shape[0])
    if np.isfinite(det_rel).any():
        ok = np.isfinite(det_rel)
        ei[ok] = energy[ok] / (det_rel[ok] - np.nanmin(det_rel) + TAU_S)
        if ei.max() > 0:
            ei = ei / ei.max()
    ei[~valid] = np.nan
    return ei, det_rel, baseline_ok


def ei_scores(energy, det_rel, use):
    """EI (0-1) among the channels in `use` that have a jump; t0 = their earliest jump (nan if none)."""
    ok = np.asarray(use, bool) & np.isfinite(det_rel)
    ei = np.zeros(len(det_rel))
    if not ok.any():
        return ei, np.nan
    t0 = float(np.nanmin(det_rel[ok]))
    ei[ok] = energy[ok] / (det_rel[ok] - t0 + TAU_S)
    if ei.max() > 0:
        ei /= ei.max()
    return ei, t0


def ei_onset_from_parts(energy, det_rel, bad, params=None):
    """Version 6.1 EI onset of one seizure (one montage) from the EI parts.
    Returns a dict: start (jump time, s from the marked onset), edge (already active), score (EI among the sites
    not already active), onset (bool), t0, flags ([] = the EI gives an answer)."""
    p = {**EI_DEFAULTS, **(params or {})}
    good = ~np.asarray(bad, bool)
    det = np.where(good & (det_rel <= p["post_s"] + 1e-9), det_rel, np.nan)
    jumped = np.isfinite(det)
    edge = jumped & ((det < p["early_s"] - 1e-9) | (det <= -p["lead_s"] + STEP_S / 2))
    out = {"start": det, "edge": edge, "score": np.zeros(len(det)), "onset": np.zeros(len(det), bool),
           "t0": np.nan, "flags": []}
    if not jumped.any():
        out["flags"] = ["no onset"]
        return out
    if edge.sum() and edge.sum() >= p["active_share"] * jumped.sum():
        out["flags"] = ["state change before the onset (already active)"]
        return out
    score, t0 = ei_scores(energy, det, good & ~edge)
    out["score"], out["t0"] = score, t0
    if not np.isfinite(t0):
        out["flags"] = ["no onset"]
        return out
    out["onset"] = (score >= p["threshold"]) & good & ~edge & np.isfinite(det)
    return out


def epileptogenicity_parts(er, t, onset, pre_s, post_s, lead_s):
    """The EI's building blocks for one seizure: per channel the energy ratio summed over the 5 s after its jump,
    the jump time relative to the onset (s, nan = no jump), valid channels, and whether the baseline was long
    enough. er: channels x times (raw energy ratio), t: times (s, same clock as onset)."""
    n_ch = er.shape[0]
    base = (t >= onset - pre_s) & (t < onset - max(GUARD_S, lead_s + 2.0))
    baseline_ok = bool(base.sum() >= int(5.0 / STEP_S))
    if not baseline_ok:  # short pre-onset period: use whatever is before the search window (or before onset)
        base = t < onset - lead_s
        if base.sum() < int(2.0 / STEP_S):
            base = t < onset - 1.0
    if base.sum() < 4:
        raise ValueError("less than 2 s of recording before the seizure onset - cannot set a baseline")
    ref = np.median(er[:, base], axis=1)
    valid = np.isfinite(ref) & (ref > 0)
    ern = np.nan_to_num(er / np.where(valid, ref, 1.0)[:, None], nan=0.0, posinf=1e6, neginf=0.0)
    mu0 = ern[:, base].mean(axis=1)

    win = np.flatnonzero((t >= onset - lead_s) & (t <= onset + post_s))
    h = int(round(H_S / STEP_S))
    onset_frame = int(np.searchsorted(t, onset))
    confirm_frame = int(np.searchsorted(t, onset + CONFIRM_S))
    det = np.full(n_ch, np.nan)
    energy = np.zeros(n_ch)
    if len(win):
        for c in np.flatnonzero(valid):
            j = detect_change(ern[c], win[0], win[-1] + 1, mu0[c], onset_frame, confirm_frame)
            if j >= 0:
                det[c] = t[j]
                energy[c] = ern[c, j:j + h].sum()
    return energy, det - onset, valid, baseline_ok


def estimate_onset(er, t):
    """Seizure onset estimated from the data when events.tsv has none. Every contact's energy ratio
    (relative to the first part of the recording) goes through a change detector; the onset is the
    SECOND earliest detection, so one noisy contact cannot set it."""
    early = t < t[0] + max(10.0, 0.2 * (t[-1] - t[0]))
    ref = np.median(er[:, early], axis=1)
    ok = np.isfinite(ref) & (ref > 0)
    if not ok.any() or early.all():
        return None
    ern = er[ok] / ref[ok, None]
    after = np.flatnonzero(~early)
    det = []
    for row in ern:
        k = page_hinkley(row[after])
        if k >= 0:
            det.append(float(t[after[k]]))
    if not det:
        return None
    det.sort()
    return det[1] if len(det) > 1 else det[0]


# --------------------------------------------------------------------------- one recording
CLIP_BEFORE_S = 32.0   # seconds kept before the marked onset (the clips of this dataset start 30 s before it)
CLIP_AFTER_S = 31.0    # seconds kept after it (search to +20 s, a start must last up to 9 s, + filter margin)


def analyse_recording(edf_path, channels, events, sidecar, coord_names, coords, aliases, pre_s=30.0, post_s=20.0):
    """Read one seizure recording and make its arrays (see analyse_signals). Returns a dict for save_arrays()."""
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    fs = float(raw.info["sfreq"])
    duration = raw.n_times / fs
    idx, how, diag = select_channels(raw.ch_names, channels, coord_names, aliases)
    if len(idx) < 3:
        raise ValueError(f"only {len(idx)} usable channels ({how})")
    if fs < 100:
        raise ValueError(f"sampling rate {fs} Hz is too low (needs >= 100 Hz)")

    onset, src = find_onset(events)
    if onset is not None and not (0 <= onset <= duration):
        src += f" (onset {onset:.1f}s outside the {duration:.1f}s recording, ignored)"
        onset = None
    names = [raw.ch_names[i] for i in idx]
    norms = diag["norms"]
    n_edf = len(raw.ch_names)
    line = power_line_frequency(sidecar)
    raw.pick(names)
    if onset is None:   # no onset in events.tsv: estimate it from the whole recording first
        x = preprocess(raw.get_data(), fs, line)
        t_e, er_e = energy_ratio(x - x.mean(axis=0, keepdims=True), fs)
        onset = estimate_onset(er_e, t_e)
        src = "estimated from the recording (no onset in events.tsv)"
        if onset is None:
            raise ValueError("no seizure onset in events.tsv and none could be estimated")
    tmin = max(0.0, onset - max(CLIP_BEFORE_S, pre_s + 2.0))
    tmax = min(raw.times[-1], onset + max(CLIP_AFTER_S, post_s + H_S + 2.0))
    raw.crop(tmin=tmin, tmax=tmax)
    x = raw.get_data()  # volts, channels x samples
    del raw
    out = analyse_signals(x, fs, tmin, onset, names, norms, coords, line)
    out.update({"onset_source": src, "duration_s": round(duration, 1), "n_edf_channels": n_edf,
                "channel_selection": how, "n_scalp_ignored": diag["n_scalp"],
                "unmatched_edf": ";".join(diag["unmatched"])})
    return out


def analyse_signals(x, fs, tmin, onset, names, norms, coords=None, line_freq=60.0):
    """The arrays of one seizure from its raw signals.
    x: channels x samples (volts) starting at time tmin (s, recording clock); onset: marked onset (same clock);
    names / norms: recorded and matching names of the channels; coords: {norm: xyz} for pairing neighbours.
    Both montages - common average ('avg') and bipolar ('bip') - get: names, matching names, bad-channel flags,
    the energy ratio over time (EI) and the 8 onset features over time (onset_lib)."""
    x = preprocess(np.asarray(x, dtype=np.float64), fs, line_freq)
    pre = (np.arange(x.shape[1]) / fs + tmin) < onset - 1.0
    bad = bad_channels(x[:, pre] if pre.sum() > fs else x)
    good = ~bad
    if good.sum() < 3:
        raise ValueError("fewer than 3 good channels")
    xa = x - x[good].mean(axis=0, keepdims=True)                       # common average reference
    t, er_avg = energy_ratio(xa, fs)
    _, feat_avg = onset_lib.feature_series(xa, fs, line_freq)
    del xa
    pairs = [(i, j) for i, j in bipolar_pairs(norms, coords) if good[i] and good[j]]
    if pairs:
        xb = np.stack([x[i] - x[j] for i, j in pairs])
        bad_b = bad_channels(xb[:, pre] if pre.sum() > fs else xb)
        _, er_bip = energy_ratio(xb, fs)
        _, feat_bip = onset_lib.feature_series(xb, fs, line_freq)
        del xb
    else:
        bad_b, er_bip = np.zeros(0, bool), np.zeros((0, len(t)))
        feat_bip = np.zeros((len(onset_lib.FEATURES), 0, len(t)), np.float32)
    shown = [display_name(n) for n in names]
    return {
        "version": VERSION, "t": t + tmin, "onset": float(onset), "fs": float(fs), "line_freq": float(line_freq),
        "avg_names": shown, "avg_norm": list(norms), "avg_bad": bad, "avg_er": er_avg, "avg_feat": feat_avg,
        "bip_names": [f"{shown[i]}-{shown[j]}" for i, j in pairs],
        "bip_norm": [f"{norms[i]}-{norms[j]}" for i, j in pairs], "bip_bad": bad_b, "bip_er": er_bip,
        "bip_feat": feat_bip,
    }


_STR_LISTS = ("avg_names", "avg_norm", "bip_names", "bip_norm")


def save_arrays(path, d):
    """Save one recording's arrays (16-bit floats, compressed: ~0.5-1 MB per seizure). Written to a temporary file
    first, so an interrupted run never leaves a half-written result."""
    out = {}
    for k, v in d.items():
        if k in _STR_LISTS:
            out[k] = np.array([str(s) for s in v], dtype=str)
        elif k in ("avg_er", "bip_er"):
            out[k] = np.clip(np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0), 0, 6e4).astype(np.float16)
        elif k in ("avg_feat", "bip_feat"):
            a = np.asarray(v, dtype=np.float64)
            out[k] = np.where(np.isfinite(a), np.clip(a, -6e4, 6e4), np.nan).astype(np.float16)
        elif k == "t":
            out[k] = np.asarray(v, dtype=np.float64)
        elif k in ("avg_bad", "bip_bad"):
            out[k] = np.asarray(v, dtype=bool)
        else:
            out[k] = np.array(v)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k in list(d):
        if k in _STR_LISTS:
            d[k] = [str(s) for s in d[k]]
        elif d[k].ndim == 0:
            d[k] = d[k].item()
    d["t"] = d["t"].astype(np.float64)
    for k in ("avg_feat", "bip_feat"):
        if k in d:
            d[k] = d[k].astype(np.float32)
    return d


EI_DETECT_POST_S = 20.0   # the EI's change-point search runs this long after the onset; jumps after EI_SEARCH_END_S
                          # are then not accepted (ei_onset_from_parts)


COMBO_LL_DEFAULTS = {"method": "linelength", "t_from": -5.0, "z": 4.0, "persist_s": 9.0, "onset_oi": 0.3,
                     "global_q": 0.5}


def method_params(method, overrides=None, diffuse_share=None):
    """Complete settings of an onset method: its defaults updated with `overrides` (e.g. the benchmark's choice)."""
    o = dict(overrides or {})
    o.pop("method", None)
    ds = {} if diffuse_share is None else {"diffuse_share": diffuse_share}
    if method == "ei+linelength":
        return {"method": method, "ei": {**EI_DEFAULTS, **o.get("ei", {})},
                "ll": {**onset_lib.DEFAULT_PARAMS, **COMBO_LL_DEFAULTS, **o.get("ll", {}), **ds, "method": "linelength"}}
    if method == "ei":
        return {**EI_DEFAULTS, **o, "method": "ei"}
    if method in onset_lib.METHODS:
        return {**onset_lib.DEFAULT_PARAMS, **o, **ds, "method": method}
    raise ValueError(f"unknown onset method {method!r} (ei+linelength, ei, linelength or multifeature)")


def describe_method(p, montage=""):
    """One line in plain words: the method and its settings."""
    m = f", {montage}" if montage else ""

    def ei_text(e):
        return (f"EI: search {-e['lead_s']:+g} to {e['post_s']:+g} s, a jump before {e['early_s']:+g} s = already "
                f"active, onset site = EI >= {e['threshold']:g}")

    def ll_text(q):
        name = "line length" if q["method"] == "linelength" else "8 features"
        return (f"{name}: search {q['t_from']:+g} to {q['t_to']:+g} s, z >= {q['z']:g} lasting {q['frac']:.0%} of "
                f"{q['persist_s']:g} s, onset site = onset index >= {q['onset_oi']:g}")

    if p["method"] == "ei+linelength":
        return f"EI + line length{m} (an onset site of either) - {ei_text(p['ei'])}; {ll_text(p['ll'])}"
    if p["method"] == "ei":
        return f"EI alone{m} - {ei_text(p)}"
    return f"{METHOD_NAMES[p['method']]} alone{m} - {ll_text(p)}"


def ei_parts(arr, key, lead_s, pre_s=30.0, post_s=EI_DETECT_POST_S):
    """Energy after the jump and jump time (s from the marked onset) of every site of one montage ('avg'/'bip')."""
    bad = np.asarray(arr[f"{key}_bad"], bool)
    er = arr[f"{key}_er"]
    energy, det = np.zeros(len(bad)), np.full(len(bad), np.nan)
    good = ~bad
    if good.any() and er.shape[1]:
        e, d, _, _ = epileptogenicity_parts(np.asarray(er[good], np.float64), arr["t"], arr["onset"], pre_s, post_s,
                                            lead_s)
        energy[good], det[good] = e, d
    return energy, det


def short_flag(flags):
    """'state change', 'already active', 'no onset', ... - why one detector gave no answer (short form)."""
    if not flags:
        return "no onset site"
    f = flags[0]
    for key, short in (("state change", "state change"), ("already active", "already active"),
                       ("diffuse", "diffuse onset"), ("short baseline", "short baseline"), ("no sites", "no sites")):
        if key in f:
            return short
    return f.split(" (")[0]


def combine_onsets(e, ll):
    """The version 6.1 rule: a site is an onset site when the EI OR line length marks it (each only when it gives
    an answer for this seizure). e: ei_onset_from_parts() result, ll: dict start/edge/score/onset/flags."""
    n = len(e["start"])
    use_e, use_l = not e["flags"], not ll["flags"]
    eo = e["onset"] if use_e else np.zeros(n, bool)
    lo = ll["onset"] if use_l else np.zeros(n, bool)
    onset = eo | lo
    es, ls = e["start"], ll["start"]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        start = np.where(eo & lo, np.fmin(es, ls), np.where(eo, es, np.where(lo, ls, np.fmin(es, ls))))
        score = np.fmax(e["score"] if use_e else np.zeros(n), ll["score"] if use_l else np.zeros(n))
    flags = [] if onset.any() else [f"EI {short_flag(e['flags'])} + line length {short_flag(ll['flags'])}"]
    return {"start": start, "edge": e["edge"] | ll["edge"], "score": score, "onset": onset, "flags": flags,
            "t0": float(np.nanmin(start[onset])) if onset.any() else np.nan}


def site_onsets(arr, key, params):
    """Onset of one saved seizure, one montage ('avg' / 'bip'), with the method in params['method']:
    'ei+linelength' (params 'ei' and 'll'), 'ei', 'linelength' or 'multifeature'. Returns a dict of per-site arrays
    start, edge (already active), score (EI / OI), onset, plus flags ([] = usable), t0 and the parts of each
    detector (ei_start, ei_score, ll_start, ll_score, ei_flags, ll_flags)."""
    method = (params or {}).get("method", "multifeature")
    params = method_params(method, params)
    bad = np.asarray(arr[f"{key}_bad"], bool)
    n = len(bad)
    nan = np.full(n, np.nan)
    empty = {"start": nan, "edge": np.zeros(n, bool), "score": np.zeros(n), "onset": np.zeros(n, bool), "t0": np.nan}
    e = ll = None
    if method in ("ei", "ei+linelength"):
        pe = params["ei"] if method == "ei+linelength" else params
        energy, det = ei_parts(arr, key, pe["lead_s"], pe["pre_s"])
        e = ei_onset_from_parts(energy, det, bad, pe) if n else {**empty, "flags": ["no sites"]}
    if method in ("linelength", "multifeature", "ei+linelength"):
        pl = params["ll"] if method == "ei+linelength" else params
        ll = {**empty, "flags": ["no sites"]}
        if n and arr[f"{key}_feat"].shape[2]:
            try:
                d = onset_lib.detect(arr[f"{key}_feat"], arr["t"] - arr["onset"], pl, bad=bad)
                ll = {"start": d["start"], "edge": d["edge"] & ~bad, "score": d["oi"],
                      "onset": onset_lib.onset_sites(d, pl) & ~bad, "t0": d["t0"],
                      "flags": onset_lib.seizure_flags(d, int((~bad).sum()), pl)}
            except onset_lib.BaselineError as err:
                ll = {**empty, "flags": [f"short baseline ({err})"]}
    if method == "ei+linelength":
        out = combine_onsets(e, ll)
    elif method == "ei":
        out = dict(e)
    else:
        out = dict(ll)
    out["onset"] = out["onset"] & ~bad & (not out["flags"])
    out["ei_onset"] = (e["onset"] & ~bad) if e and not e["flags"] else np.zeros(n, bool)
    out["ll_onset"] = (ll["onset"] & ~bad) if ll and not ll["flags"] else np.zeros(n, bool)
    out["ei_start"] = e["start"] if e else nan
    out["ei_score"] = e["score"] if e else np.zeros(n)
    out["ei_flags"] = "; ".join(e["flags"]) if e else ""
    out["ll_start"] = ll["start"] if ll else nan
    out["ll_score"] = ll["score"] if ll else np.zeros(n)
    out["ll_flags"] = "; ".join(ll["flags"]) if ll else ""
    return out


def onset_tables(arr, params):
    """Where and when one saved seizure starts, with the method of params (both montages).
    Returns (sites table, seizure table): per site start_s (relative to the marked onset), edge (already active),
    oi (EI or onset index), onset_site, and the parts of each detector; per montage the seizure's flags
    ('' = usable), earliest onset t0_s and number of onset sites."""
    if "bip_feat" not in arr:
        raise ValueError("these saved results have no onset features (made by an older version) - run step 2 again")
    sites, seiz = [], []
    for key, montage in (("avg", "average"), ("bip", "bipolar")):
        names, norms, bad = arr[f"{key}_names"], arr[f"{key}_norm"], np.asarray(arr[f"{key}_bad"], bool)
        r = site_onsets(arr, key, params)
        on = r["onset"]
        sites.append(pd.DataFrame({"montage": montage, "contact": names, "contact_norm": norms, "bad_channel": bad,
                                   "start_s": np.round(r["start"], 3), "edge": r["edge"],
                                   "oi": np.round(r["score"], 4), "onset_site": on,
                                   "ei_onset": r["ei_onset"], "ll_onset": r["ll_onset"],
                                   "ei_start_s": np.round(r["ei_start"], 3), "ei_score": np.round(r["ei_score"], 4),
                                   "ll_start_s": np.round(r["ll_start"], 3), "ll_score": np.round(r["ll_score"], 4)}))
        seiz.append({"montage": montage, "flags": "; ".join(r["flags"]), "usable": not r["flags"], "t0_s": r["t0"],
                     "n_onset": int(on.sum()), "n_started": int(np.isfinite(r["start"]).sum()),
                     "n_analysed": int((~bad).sum()), "ei_flags": r["ei_flags"], "ll_flags": r["ll_flags"]})
    return pd.concat(sites, ignore_index=True), pd.DataFrame(seiz)


def ei_tables(arr, pre_s, post_s, lead_s):
    """EI of every contact (average reference) and every bipolar pair for one saved recording."""
    out = []
    for key, montage in (("avg", "average"), ("bip", "bipolar")):
        names, norms, bad, er = arr[f"{key}_names"], arr[f"{key}_norm"], arr[f"{key}_bad"], arr[f"{key}_er"]
        ei = np.full(len(names), np.nan)
        det = np.full(len(names), np.nan)
        good = ~np.asarray(bad, bool)
        baseline_ok = True
        if good.any() and er.shape[1]:
            e, d, baseline_ok = epileptogenicity_index(er[good].astype(np.float64), arr["t"], arr["onset"],
                                                       pre_s, post_s, lead_s)
            ei[good], det[good] = e, d
        df = pd.DataFrame({"montage": montage, "contact": names, "contact_norm": norms,
                           "bad_channel": np.asarray(bad, bool), "ei": ei, "detect_s": det})
        df["n_detected"] = int(np.isfinite(det).sum())
        df["n_analysed"] = int(good.sum())
        df["baseline_ok"] = baseline_ok
        out.append(df)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- SOZ of one patient
def neighbourhood(norms):
    """Contacts of the given sites plus their direct neighbours on the same electrode (n-1, n+1)."""
    out = set()
    for site in norms:
        for c in str(site).split("-"):
            s = split_numbered(c)
            out |= {c} if not s else {f"{s[0]}{s[1] - 1}", c, f"{s[0]}{s[1] + 1}"}
    return out


def soz_sites(p, rule="multifocal", onset_ei=0.3, mean_ei=0.3, min_onsets=2, strong_ei=STRONG_EI):
    """SOZ label of every site of ONE patient (one montage).
    p: per-seizure EI rows of the patient's informative seizures (seizures with a detected onset), bad channels
       removed; columns seizure, contact_norm, contact, ei, detect_s.
    rule "multifocal" (version 5): SOZ = strong onset site (EI >= strong_ei) in >= min_onsets seizures, OR the
         classic rule below; uncertain = strong onset site in some seizure but not SOZ.
    rule "classic" (version 4): SOZ = mean EI over all seizures >= mean_ei and EI >= onset_ei in >= min_onsets
         seizures (a focus that starts only some of the seizures is averaged away); no uncertain sites.
    rule "repeated" (version 6): SOZ = onset site (score >= strong_ei) in >= min_onsets seizures (the multifocal rule
         without its classic part); uncertain = onset site in some seizure but not SOZ. The score column is called
         'ei' here but holds the onset index (OI) in version 6.
    (With a single seizure, min_onsets becomes 1.)
    Returns (sites table indexed by contact_norm, share of seizures started at the SOZ):
       multifocal: a SOZ site or its neighbour is a strong onset site in the seizure;
       classic: the seizure's strongest site is a SOZ site or its neighbour."""
    q = p.assign(ei0=p["ei"].fillna(0.0))
    n = int(q["seizure"].nunique())
    g = (q.groupby("contact_norm")
          .agg(contact=("contact", "first"), n_seizures=("seizure", "nunique"),
               n_onset=("ei0", lambda v: int((v >= onset_ei).sum())),
               n_strong=("ei0", lambda v: int((v >= strong_ei).sum())),
               ei_mean=("ei0", "mean"), ei_median=("ei0", "median"), ei_max=("ei0", "max"),
               detect_s_median=("detect_s", "median")))
    need = min(min_onsets, max(n, 1))
    classic = (g["ei_mean"] >= mean_ei) & (g["n_onset"] >= need)
    if rule == "classic":
        soz = classic
        g["uncertain"] = 0
    elif rule == "multifocal":
        soz = classic | (g["n_strong"] >= need)
        g["uncertain"] = ((g["ei_max"] >= strong_ei) & ~soz).astype(int)
    elif rule == "repeated":
        soz = g["n_strong"] >= need
        g["uncertain"] = ((g["ei_max"] >= strong_ei) & ~soz).astype(int)
    else:
        raise ValueError(f"unknown SOZ rule {rule!r} (use 'repeated', 'multifocal' or 'classic')")
    g["soz_ei"] = soz.astype(int)
    if n == 0 or not soz.any():
        return g, np.nan
    near = neighbourhood(g.index[soz])
    if rule == "classic":
        tops = q.loc[q.groupby("seizure")["ei0"].idxmax(), "contact_norm"]
        started = np.mean([bool(set(str(t).split("-")) & near) for t in tops])
    else:
        strong = q[q["ei0"] >= strong_ei]
        hit = strong.groupby("seizure")["contact_norm"].apply(
            lambda s: any(set(str(x).split("-")) & near for x in s))
        started = float(hit.reindex(sorted(q["seizure"].unique()), fill_value=False).mean())
    return g, round(float(started), 2)


def label_quality(n_informative, n_soz, started, min_started, rule="repeated"):
    """How far a patient's SOZ labels can be trusted (only 'ok' and 'few seizures' are used for training)."""
    if n_informative == 0:
        return "no usable seizure"
    if n_soz == 0:
        return "no repeated onset" if rule in ("multifocal", "repeated") else "no consistent onset"
    if not started >= min_started:
        return "SOZ rarely leads"
    if n_informative == 1:
        return "single seizure"
    return "few seizures" if n_informative == 2 else "ok"


# --------------------------------------------------------------------------- coordinates
def _tokens(line):
    return [t for t in re.split(r"[,\s;]+", line.strip()) if t]


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_names(path):
    names = []
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            toks = _tokens(line)
            if toks and not line.lstrip().startswith("#"):
                names += [t for t in toks if not _is_num(t)] if len(toks) > 1 else toks
    return names


def read_coordinate_file(path):
    """Rows of (name or None, x, y, z) from an iEEG-recon coordinate file.
    Accepts 'x y z', 'name x y z', 'x y z 1' (homogeneous), 'index x y z', with comma / space / tab
    separators and an optional header line."""
    names, nums = [], []
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            toks = _tokens(line)
            if sum(_is_num(t) for t in toks) < 3:
                continue  # header line
            if _is_num(toks[0]):
                names.append(None)
                nums.append([float(t) for t in toks if _is_num(t)])
            else:
                names.append(toks[0])
                nums.append([float(t) for t in toks[1:] if _is_num(t)])
    if not nums:
        return []
    width = min(len(r) for r in nums)
    m = np.array([r[:width] for r in nums], dtype=float)
    if width >= 4:
        n = len(m)
        if np.allclose(m[:, 3], 1.0):            # homogeneous coordinates: x y z 1
            m = m[:, :3]
        elif np.array_equal(m[:, 0], np.arange(n)) or np.array_equal(m[:, 0], np.arange(1, n + 1)):
            m = m[:, 1:4]                          # index x y z
        else:
            m = m[:, :3]
    return [(nm, *xyz) for nm, xyz in zip(names, m[:, :3])]


def coordinate_candidates(names_path, mm_path, vox_path, t1_img):
    """Possible scanner-RAS (mm) coordinates of the contacts: the mm file as it is ('mm'), and the voxel
    file through the reference MRI header, counting voxels from 0 ('vox0') or from 1 ('vox1').
    Returns ({label: DataFrame contact, contact_norm, x, y, z}, info)."""
    import nibabel as nib
    names = read_names(names_path) if names_path else []

    def with_names(rows, label):
        if not rows:
            return None, None
        if all(r[0] for r in rows):
            return [r[0] for r in rows], np.array([r[1:] for r in rows], dtype=float)
        if len(names) == len(rows):
            return names, np.array([r[1:] for r in rows], dtype=float)
        raise ValueError(f"{label}: {len(rows)} coordinates but {len(names)} names - cannot pair them")

    def table(nm, xyz):
        df = pd.DataFrame({"contact": nm, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        df["contact_norm"] = df["contact"].map(normalize_name)
        return df.drop_duplicates("contact_norm", keep="first").reset_index(drop=True)

    cands, info = {}, {}
    mm_names, mm_xyz = with_names(read_coordinate_file(mm_path) if mm_path else [], "mm file")
    vox_names, vox_ijk = with_names(read_coordinate_file(vox_path) if vox_path else [], "voxel file")
    if mm_xyz is not None:
        cands["mm"] = table(mm_names, mm_xyz)
    if vox_ijk is not None and t1_img is not None:
        w0 = nib.affines.apply_affine(t1_img.affine, vox_ijk)
        w1 = nib.affines.apply_affine(t1_img.affine, vox_ijk - 1.0)
        cands["vox0"], cands["vox1"] = table(vox_names, w0), table(vox_names, w1)
        if mm_xyz is not None and mm_names == vox_names:
            info["mm_vs_vox_0based_mm"] = round(float(np.median(np.linalg.norm(mm_xyz - w0, axis=1))), 2)
            info["mm_vs_vox_1based_mm"] = round(float(np.median(np.linalg.norm(mm_xyz - w1, axis=1))), 2)
    return cands, info


def electrode_coordinates(names_path, mm_path, vox_path, t1_img):
    """Electrode contacts with scanner-RAS coordinates (mm) in the space of the reference MRI: the mm file
    when it agrees with the voxel file + MRI header (or when there is nothing to check it against),
    otherwise the voxel file + header. Returns (DataFrame contact, contact_norm, x, y, z; info).
    (SOZ/03 makes the final choice with the FreeSurfer brain when the two disagree.)"""
    cands, info = coordinate_candidates(names_path, mm_path, vox_path, t1_img)
    agree = min(info.get("mm_vs_vox_0based_mm", np.inf), info.get("mm_vs_vox_1based_mm", np.inf)) < 1.5
    if "mm" in cands and ("vox0" not in cands or agree):
        src = "mm file (agrees with voxel file + MRI header)" if agree else \
              "mm file (not cross-checked: no voxel file or MRI)"
        df = cands["mm"]
    elif "vox0" in cands:
        src, df = "voxel file + MRI header", cands["vox0"]
    else:
        raise ValueError("no electrode coordinates found")
    info["coord_source"] = src
    return df, info


def space_check(fs_img, fs_labels, ref_img, n=20000, shift_mm=4.0, seed=0):
    """Is the reference MRI (that the coordinates belong to) in the same scanner space as the
    FreeSurfer MRI? Correlates the two images inside the brain at the same scanner coordinates,
    and compares with the same test after shifting by +/- shift_mm in x, y and z.
    Aligned images give the highest correlation with no shift."""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    a = np.asanyarray(fs_img.dataobj).astype(np.float32)
    b = np.asanyarray(ref_img.dataobj).astype(np.float32)
    if b.ndim > 3:
        b = b[..., 0]
    ijk = np.argwhere(np.asanyarray(fs_labels.dataobj) > 0)
    ijk = ijk[rng.choice(len(ijk), size=min(n, len(ijk)), replace=False)]
    world = nib.affines.apply_affine(fs_img.affine, ijk)
    va = a[tuple(ijk.T)]
    inv = np.linalg.inv(ref_img.affine)

    def corr(offset):
        q = nib.affines.apply_affine(inv, world + offset)
        inside = np.all((q >= 0) & (q <= np.array(b.shape) - 1), axis=1)
        if inside.sum() < 100:
            return np.nan, inside.mean()
        vb = ndimage.map_coordinates(b, q[inside].T, order=1)
        return float(np.corrcoef(va[inside], vb)[0, 1]), float(inside.mean())

    c0, inside0 = corr(np.zeros(3))
    shifted = [corr(np.eye(3)[k] * s)[0] for k in range(3) for s in (-shift_mm, shift_mm)]
    best_shifted = float(np.nanmax(shifted)) if np.isfinite(shifted).any() else np.nan
    aligned = bool(np.isfinite(c0) and c0 > 0.3 and inside0 > 0.9 and (not np.isfinite(best_shifted) or c0 >= best_shifted))
    return {"corr_same_position": round(c0, 3), "corr_best_shifted": round(best_shifted, 3),
            "overlap": round(inside0, 3), "aligned": aligned}
