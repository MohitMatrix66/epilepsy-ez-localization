"""
onset_selftest.py  -  quick tests of the onset detector (SOZ/onset_lib.py) on fake signals with a known answer,
at the sampling rates of the real data (256, 500, 512, 1024, 2048 Hz). Run by tools/smoke_test.py (about 30 s);
can also be run on its own:  python tools/onset_selftest.py
    fast    : 3 contacts start fast activity (the classic onset)
    slow    : big slow waves everywhere fade 15 s before the seizure, then 3 contacts start big rhythmic slow
              waves (the real sub-PID04 - the version 5 EI fails here)
    muscle  : muscle activity on 24 of 30 contacts around the onset; the seizure starts on 3 others
    pops    : one contact with electrode pops around the onset; the seizure starts on 3 others
"""
import os
import sys

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "SOZ"))
import onset_lib  # noqa: E402
import soz_lib  # noqa: E402

PRE, POST = 32.0, 31.0


def _pink(rng, n_ch, n, fs):
    w = rng.normal(size=(n_ch, n))
    f = np.fft.rfftfreq(n, 1 / fs)
    x = np.fft.irfft(np.fft.rfft(w, axis=1) / np.sqrt(np.maximum(f, 1.0)), n=n, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _band(rng, n, fs, lo, hi):
    sos = signal.butter(4, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, rng.normal(size=n))
    return x / x.std()


def scenario(kind, fs, seed=0, n_ch=30):
    """Fake signals (volts, 30 sites) from 32 s before to 31 s after the marked onset, and the planted onset sites."""
    rng = np.random.default_rng(seed)
    n = int((PRE + POST) * fs)
    t = np.arange(n) / fs - PRE
    x = 40 * _pink(rng, n_ch, n, fs) + 8 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (n_ch, 1)))
    x += 6 * np.sin(2 * np.pi * 60 * t)
    if kind == "fast":
        truth = [0, 1, 2]
        for c in truth:
            x[c] += 45 * np.clip((t - 1.0 - rng.uniform(0, 0.3)) / 1.0, 0, 1) * _band(rng, n, fs, 25, 90)
        for c in range(3, 6):
            x[c] += 45 * np.clip((t - 6.0 - rng.uniform(0, 0.5)) / 1.0, 0, 1) * _band(rng, n, fs, 25, 90)
    elif kind == "slow":
        truth = [0, 1, 2]
        fade = np.where(t < -15, 1.0, np.where(t < -13, 1 - 0.65 * (t + 15) / 2, 0.35))
        x += 90 * np.stack([_band(rng, n, fs, 0.8, 3.5) for _ in range(n_ch)]) * fade
        for c in truth:
            env = np.clip((t - 2.0 - rng.uniform(0, 0.3)) / 1.5, 0, 1)
            x[c] += 160 * env * signal.sawtooth(2 * np.pi * 4.5 * t + rng.uniform(0, 6), 0.15)
        for c in range(3, 9):
            x[c] += 120 * np.clip((t - 7.0 - rng.uniform(0, 1)) / 1.5, 0, 1) * np.sin(2 * np.pi * 5 * t)
    elif kind == "muscle":
        truth = [10, 11, 12]
        env = ((t > -2) & (t < 7)).astype(float)
        for c in [c for c in range(n_ch) if c not in truth][:24]:
            x[c] += 50 * env * _band(rng, n, fs, 30, 220)
        for c in truth:
            x[c] += 100 * np.clip((t - 3.0 - rng.uniform(0, 0.3)) / 1.0, 0, 1) * np.sin(2 * np.pi * 6 * t) \
                * (1 + 0.5 * _band(rng, n, fs, 1, 3))
    elif kind == "pops":
        truth = [0, 1, 2]
        for p in np.sort(rng.uniform(-3, 12, 7)):
            x[20] += 600 * np.exp(-np.clip(t - p, 0, None) / 0.05) * (t >= p)
        for c in truth:
            x[c] += 45 * np.clip((t - 2.0 - rng.uniform(0, 0.3)) / 1.0, 0, 1) * _band(rng, n, fs, 25, 90)
    else:
        raise ValueError(kind)
    return x * 1e-6, t, truth


def run(kind, fs, params=None, seed=0):
    x, t, truth = scenario(kind, fs, seed)
    x = soz_lib.preprocess(x, fs, 60.0)
    tt, F = onset_lib.feature_series(x, fs, 60.0)
    det = onset_lib.detect(F, tt + t[0], params or {})
    return truth, list(np.flatnonzero(onset_lib.onset_sites(det))), onset_lib.seizure_flags(det, x.shape[0])


def main(verbose=True):
    problems = []
    for fs in (256, 500, 512, 1024, 2048):
        for kind in ("fast", "slow", "muscle", "pops"):
            truth, found, flags = run(kind, fs)
            good = not flags and set(found) <= set(truth) and len(found) >= 2
            if verbose:
                print(f"  {fs:5d} Hz  {kind:7s} planted {truth}  found {[int(f) for f in found]}  "
                      f"{'ok' if good else 'WRONG'}" + (f"  flags {flags}" if flags else ""))
            if not good:
                problems.append(f"{kind} at {fs} Hz: planted {truth}, found {found}, flags {flags}")
    return problems


if __name__ == "__main__":
    p = main()
    print("ONSET DETECTOR SELF-TEST " + ("PASSED" if not p else "FAILED:\n  " + "\n  ".join(p)))
