"""
onset_lib.py  -  WHERE and WHEN does a seizure start? The line-length / 8-feature onset detector (version 6.1).

In version 6.1 this detector is used TOGETHER with the Epileptogenicity Index (EI, soz_lib.ei_onset): a site is an
onset site of a seizure when either of the two marks it (soz_lib.combined_onset). On 40 seizures marked by
epileptologists the two find different parts of the onset - the EI the low-voltage fast activity, line length the
rhythmic discharges - and together they agree with the experts clearly better than either alone
(tools/benchmark_onset.py).

What this detector does (based on "Imprint", Gascoigne et al. 2024), for every site (bipolar pair or contact) and
every 0.25 s (1-s windows):
    1. measures line length - or 8 things: line length, energy and the power of 6 frequency bands
       (delta 1-4, theta 4-8, alpha 8-13, beta 13-30, low gamma 30-55, high gamma 65-100 Hz; the power-line
       frequency and its harmonics are left out),
    2. compares each with the site's OWN normal period before the seizure (median and spread, so a few spikes
       in the normal period do not matter): z = how many "spreads" above normal,
    3. counts only INCREASES, never a ratio - fading slow waves cannot create a start,
    4. removes a rise that happens at many sites at the same moment (a whole-brain change such as muscle
       activity), but never a whole-brain fall (removing falls would create fake starts),
    5. a site "starts" when it is above the threshold and STAYS abnormal (>= 80% of the next few seconds),
       searched from t_from to t_to (+10 s: on the benchmark no expert onset came later, and every later start
       was wrong),
    6. a site already abnormal in the first second of the search was "already active": it is not an onset site
       (step 2 makes it 'uncertain'); when that is true for many of the sites that start (>= active_share), the
       whole seizure is "already active" and this detector is not used for it,
    7. onset index (OI), like the EI, over the sites that were not already active: early AND strong sites score
       high; the first, strongest site = 1
           OI = (mean abnormality over the 5 s after the start) / (delay after the earliest site + 1 s)
This detector gives no answer for a seizure when: no site starts ("no onset"), the seizure was already active
("already active"), or too many sites start together ("diffuse onset").
"""
import warnings

import numpy as np
from scipy import signal

VERSION = 6
WIN_S = 1.0           # window length (s)
STEP_S = 0.25         # step between windows (s)
FEATURES = ("linelength", "energy", "delta", "theta", "alpha", "beta", "lowgamma", "highgamma")
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
         "lowgamma": (30.0, 55.0), "highgamma": (65.0, 100.0)}
ENERGY_BAND = (1.0, 100.0)
LINE_GAP_HZ = 3.0     # frequency bins this close to the power-line frequency (or a harmonic) are not used
TOP_FRACTION = 0.45   # bands stop at 0.45 x the sampling rate (the recording filter cuts above that)
MAD_FLOOR = 0.05      # smallest spread of a feature (log10 units, = x1.12) so that a very flat baseline cannot
                      # turn a tiny change into a huge z
MIN_BASELINE_S = 8.0  # a seizure needs at least this much normal recording before the search window
MIN_AVAIL_S = 2.0     # near the end of a recording a start must still be seen for at least this long
FIRST_S = 1.0         # ... and the first second after a start must be abnormal for >= 75% of it, so a single
FIRST_FRAC = 0.75     #     chance window just before the real start cannot become the start
TAU_S = 1.0           # onset index delay constant (s), as in the EI
H_S = 5.0             # onset index: abnormality averaged over this long after the start (s)
METHODS = ("multifeature", "linelength")

# Settings used when tools/benchmark_onset.py has not chosen any (the published Imprint values where they exist)
DEFAULT_PARAMS = {"method": "multifeature", "z": 3.0, "persist_s": 9.0, "frac": 0.8, "t_from": -5.0, "t_to": 10.0,
                  "onset_oi": 0.5, "base_from": -30.0, "base_gap_s": 2.0, "diffuse_share": 0.4, "global_q": 0.5,
                  "active_zone_s": 1.0, "active_share": 0.2}

_TINY = 1e-24


class BaselineError(ValueError):
    """Not enough normal recording before the seizure to compare with."""


def feature_series(x, fs, line_freq=60.0):
    """The 8 features of every site over time.
    x: sites x samples (already high-pass filtered and notch filtered, any unit), fs: sampling rate (Hz).
    Returns (t: window centres in s from the first sample of x, F: float32 array features x sites x windows of
    log10 values; NaN for a band the sampling rate cannot measure)."""
    x = np.asarray(x, dtype=np.float64)
    nper = int(round(WIN_S * fs))
    step = max(1, int(round(STEP_S * fs)))
    if x.ndim != 2 or x.shape[1] < nper + step:
        raise ValueError(f"recording too short for the onset detector ({x.shape[-1] / fs:.1f} s)")
    f, t, S = signal.spectrogram(x, fs=fs, window="hann", nperseg=nper, noverlap=nper - step,
                                 detrend="constant", scaling="density", mode="psd", axis=-1)
    keep = np.ones(len(f), bool)
    if line_freq:
        for h in range(1, int(f[-1] // line_freq) + 2):
            keep &= np.abs(f - h * line_freq) > LINE_GAP_HZ
    top = TOP_FRACTION * fs
    df = f[1] - f[0]

    def band(lo, hi):
        m = keep & (f >= lo) & (f < min(hi, top))
        if m.sum() < 1 or min(hi, top) - lo < 2 * df:
            return None
        return S[:, m, :].sum(axis=1) * df

    n_fr = S.shape[2]
    F = np.full((len(FEATURES), x.shape[0], n_fr), np.nan, np.float32)
    # line length of the same windows (segment k covers samples k*step ... k*step + nper - 1), measured on the
    # signal low-passed at the top of the energy band, so it does not depend on the sampling rate
    lp = min(ENERGY_BAND[1], top)
    xl = signal.sosfiltfilt(signal.butter(4, lp, btype="lowpass", fs=fs, output="sos"), x, axis=-1) \
        if lp < 0.49 * fs else x
    d = np.abs(np.diff(xl, axis=1))
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(d, axis=1)], axis=1)
    s0 = np.arange(n_fr) * step
    F[0] = np.log10(c[:, s0 + nper - 1] - c[:, s0] + _TINY)
    e = band(*ENERGY_BAND)
    if e is not None:
        F[1] = np.log10(e + _TINY)
    for i, name in enumerate(FEATURES[2:], start=2):
        b = band(*BANDS[name])
        if b is not None:
            F[i] = np.log10(b + _TINY)
    return t, F


def baseline_mask(t_rel, params):
    """Windows of the normal period: from base_from to base_gap_s before the search window starts."""
    return (t_rel >= params["base_from"]) & (t_rel < params["t_from"] - params["base_gap_s"])


def robust_z(F, base):
    """z = (feature - median of the normal period) / spread (1.4826 x median absolute deviation, >= MAD_FLOOR)."""
    if base.sum() * STEP_S < MIN_BASELINE_S:
        raise BaselineError(f"only {base.sum() * STEP_S:.1f} s of normal recording before the seizure "
                            f"(needs {MIN_BASELINE_S:g} s)")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows (bands a low sampling rate cannot see)
        B = F[:, :, base]
        med = np.nanmedian(B, axis=2, keepdims=True)
        mad = 1.4826 * np.nanmedian(np.abs(B - med), axis=2, keepdims=True)
        return (F - med) / np.maximum(mad, MAD_FLOOR)


def activation(Z, method="multifeature", remove_global_rise=True, global_q=0.5):
    """How abnormal every site is in every window: the largest z over the features of the method (sites x windows).
    A rise shared by many sites is subtracted: the global_q quantile over the sites (0.5 = the median: a rise at more
    than half of the sites; 0.75: at more than a quarter), only when it is > 0 - a shared fall is left alone."""
    if method == "linelength":
        Zm = Z[[0]]
    elif method == "multifeature":
        Zm = Z
    else:
        raise ValueError(f"unknown onset method {method!r} (use one of {METHODS})")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if remove_global_rise:
            g = np.nanquantile(Zm, global_q, axis=1, keepdims=True)
            Zm = Zm - np.maximum(np.nan_to_num(g, nan=0.0), 0.0)
        A = np.nanmax(np.where(np.isnan(Zm), -np.inf, Zm), axis=0)
    A[~np.isfinite(A)] = np.nan
    return A.astype(np.float32)


def start_times(A, t_rel, params):
    """First window in [t_from, t_to] from which a site is abnormal (A >= z) in >= frac of the next persist_s
    seconds (fewer at the end of the recording, but at least MIN_AVAIL_S) and in >= FIRST_FRAC of the first
    FIRST_S seconds. Returns (start in s or NaN, edge = the site was already abnormal in the first active_zone_s
    seconds of the search: 'already active')."""
    ab = np.nan_to_num(A, nan=-np.inf) >= params["z"]
    n_fr = ab.shape[1]
    k = max(1, int(round(params["persist_s"] / STEP_S)))
    cs = np.concatenate([np.zeros((ab.shape[0], 1)), np.cumsum(ab, axis=1)], axis=1)
    i = np.arange(n_fr)
    end = np.minimum(i + k, n_fr)
    share = (cs[:, end] - cs[:, i]) / (end - i)
    k1 = max(1, int(round(FIRST_S / STEP_S)))
    end1 = np.minimum(i + k1, n_fr)
    share1 = (cs[:, end1] - cs[:, i]) / (end1 - i)
    ok = ab & (share >= params["frac"]) & (share1 >= FIRST_FRAC) & ((end - i) * STEP_S >= MIN_AVAIL_S)
    win = np.flatnonzero((t_rel >= params["t_from"] - 1e-9) & (t_rel <= params["t_to"] + 1e-9))
    start = np.full(ab.shape[0], np.nan)
    edge = np.zeros(ab.shape[0], bool)
    if not len(win):
        return start, edge
    sub = ok[:, win]
    has = sub.any(axis=1)
    first = np.argmax(sub, axis=1)
    start[has] = t_rel[win[first[has]]]
    zone = max(float(params.get("active_zone_s", STEP_S)), STEP_S / 2)
    edge[has] = start[has] < params["t_from"] + zone - 1e-9
    return start, edge


def onset_index(A, t_rel, start):
    """OI of every site (0-1): mean abnormality over the H_S seconds after its start, divided by its delay after
    the earliest site + TAU_S; the best site of the seizure = 1. Returns (oi, t0 = earliest start or NaN)."""
    oi = np.zeros(A.shape[0])
    ok = np.isfinite(start)
    if not ok.any():
        return oi, np.nan
    t0 = float(np.nanmin(start))
    h = max(1, int(round(H_S / STEP_S)))
    for c in np.flatnonzero(ok):
        j = int(np.searchsorted(t_rel, start[c] - 1e-9))
        seg = np.nan_to_num(A[c, j:j + h], nan=0.0)
        oi[c] = np.clip(seg, 0, None).mean() / (start[c] - t0 + TAU_S) if len(seg) else 0.0
    if oi.max() > 0:
        oi = oi / oi.max()
    return oi, t0


def from_activation(A, t_rel, params):
    """Starts, 'already active' sites and OI from an abnormality map (the part of detect() after the features)."""
    start, edge = start_times(A, t_rel, params)
    oi, t0 = onset_index(A, t_rel, np.where(edge, np.nan, start))     # OI among the sites not already active
    return {"start": start, "edge": edge, "oi": oi, "t0": t0, "A": A, "params": params}


def detect(F, t_rel, params, bad=None):
    """Onset detection for one seizure from its features (F: features x sites x windows, t_rel: window times in s
    relative to the marked onset). bad: sites to ignore. Returns a dict with per-site arrays start, edge (already
    active), oi and the abnormality map A (sites x windows), plus t0 (earliest start of a site not already active)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    Z = robust_z(F, baseline_mask(t_rel, p))
    A = activation(Z, p["method"], remove_global_rise=True, global_q=p.get("global_q", 0.5))
    if bad is not None:
        A[np.asarray(bad, bool)] = np.nan
    return from_activation(A, t_rel, p)


def seizure_flags(det, n_sites, params=None):
    """Why this detector gives no answer for a seizure ([] = it gives one). onset sites = OI >= onset_oi."""
    p = {**DEFAULT_PARAMS, **(params or det.get("params", {}))}
    start, edge = det["start"], det["edge"]
    started = np.isfinite(start)
    if not started.any():
        return ["no onset"]
    n_edge = int((edge & started).sum())
    if n_edge and n_edge >= p.get("active_share", 0.2) * started.sum():
        return [f"already active at {p['t_from']:+g} s"]
    if not (started & ~edge).any():
        return ["no onset"]
    onset = onset_sites(det, p)
    if n_sites and onset.sum() > max(3, p["diffuse_share"] * n_sites):
        return ["diffuse onset"]
    return []


def onset_sites(det, params=None):
    """Boolean per site: an onset site of this seizure (start found, not already active, OI >= onset_oi)."""
    p = {**DEFAULT_PARAMS, **(params or det.get("params", {}))}
    return np.isfinite(det["start"]) & ~np.asarray(det["edge"], bool) & (det["oi"] >= p["onset_oi"])
