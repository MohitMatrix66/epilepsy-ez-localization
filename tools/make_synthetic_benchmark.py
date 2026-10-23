"""
make_synthetic_benchmark.py  -  a FAKE expert benchmark for the smoke test: the same tables as benchmark/
(seizures.csv + sidecars) and EDF recordings (with an EDF Annotations signal, like the real ones) that
tools/range_server.py serves like the public server, so tools/get_benchmark.py and tools/benchmark_onset.py
can be tested without the internet. Half of the fake seizures start with fast activity (the EI sees them), half
with big rhythmic slow waves after the slow waves of the normal period fade (the EI cannot see them) - like
the real sub-PID04. The fake "epileptologists" mark the planted onset contacts (with small disagreements).

Run:  python tools/make_synthetic_benchmark.py --out synthetic_data
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edf_writer import write_edf  # noqa: E402

DUR, ONSET = 160.0, 120.0
ELECTRODES = ["LA", "LB", "LC", "RA", "RB"]
N_CONTACTS = 8


def _pink(rng, n_ch, n, fs):
    w = rng.normal(size=(n_ch, n))
    f = np.fft.rfftfreq(n, 1 / fs)
    x = np.fft.irfft(np.fft.rfft(w, axis=1) / np.sqrt(np.maximum(f, 1.0)), n=n, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _band(rng, n, fs, lo, hi):
    sos = signal.butter(4, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, rng.normal(size=n))
    return x / x.std()


def make_seizure(rng, fs, kind, onset_idx, spread_idx):
    n = int(DUR * fs)
    t = np.arange(n) / fs - ONSET
    n_ch = len(ELECTRODES) * N_CONTACTS
    x = 40 * _pink(rng, n_ch, n, fs) + 8 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (n_ch, 1)))
    x += 5 * np.sin(2 * np.pi * 60 * t)
    if kind == "slow":                         # slow waves of sleep that fade 15 s before the seizure
        fade = np.where(t < -15, 1.0, np.where(t < -13, 1 - 0.65 * (t + 15) / 2, 0.35))
        x += 90 * np.stack([_band(rng, n, fs, 0.8, 3.5) for _ in range(n_ch)]) * fade
    for c in onset_idx:
        st = 0.5 + rng.uniform(0, 0.3)
        env = np.clip((t - st) / 1.0, 0, 1) * (t < 60)
        if kind == "lvfa":
            x[c] += 50 * env * _band(rng, n, fs, 25, 90)
        else:
            x[c] += 400 * env * signal.sawtooth(2 * np.pi * 4.5 * t + rng.uniform(0, 6), 0.15)
    for c in spread_idx:
        st = 6.0 + rng.uniform(0, 2)
        env = np.clip((t - st) / 1.0, 0, 1) * (t < 60)
        x[c] += (50 * _band(rng, n, fs, 25, 90) if kind == "lvfa" else 160 * np.sin(2 * np.pi * 5 * t)) * env
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--patients", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    bdir = os.path.join(args.out, "benchmark")
    sdir = os.path.join(args.out, "benchmark_server", "nm000402", "objects")
    os.makedirs(sdir, exist_ok=True)
    names = [f"{e}{c:02d}" for e in ELECTRODES for c in range(1, N_CONTACTS + 1)]
    rows = []
    for p in range(1, args.patients + 1):
        pat = f"FAKE{p:02d}"
        sub = f"sub-{pat}"
        fs = 512 if p <= args.patients // 2 else 1024
        d = os.path.join(bdir, "sidecars", sub)
        os.makedirs(d, exist_ok=True)
        e_on = int(rng.integers(len(ELECTRODES)))
        c0 = int(rng.integers(1, N_CONTACTS - 3))
        onset_idx = [e_on * N_CONTACTS + c0 + k for k in range(3)]
        spread_idx = [((e_on + 1) % len(ELECTRODES)) * N_CONTACTS + k for k in range(2, 5)]
        xyz = []
        for e in range(len(ELECTRODES)):
            origin = rng.uniform(-50, 50, 3)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            xyz += [origin + direction * 3.5 * c for c in range(N_CONTACTS)]
        el = pd.DataFrame({"name": names, "x": [v[0] for v in xyz], "y": [v[1] for v in xyz],
                           "z": [v[2] for v in xyz], "size": "n/a"})
        el.to_csv(os.path.join(d, f"{sub}_ses-postimplant_space-Other_electrodes.tsv"), sep="\t", index=False)
        for k, kind in enumerate(("lvfa", "slow")):
            task = f"ictal{int(rng.integers(10000, 900000))}"
            stem = f"{sub}_ses-postimplant_task-{task}_run-00"
            x = make_seizure(rng, fs, kind, onset_idx, spread_idx)
            ekg = np.zeros(x.shape[1])
            ekg[(np.arange(0, DUR, 0.8) * fs).astype(int)] = 900.0
            ekg = np.convolve(ekg, np.hanning(9), mode="same")
            scalp = 30 * _pink(rng, 1, x.shape[1], fs)
            data = np.vstack([x, scalp, ekg[None]])
            all_names = names + ["FP1", "EKG1"]
            tmp = os.path.join(sdir, "tmp.edf")
            write_edf(tmp, data, fs, all_names, annotations=True)
            blob = open(tmp, "rb").read()
            key = f"SHA256E-s{len(blob)}--{hashlib.sha256(blob).hexdigest()}.edf"
            os.replace(tmp, os.path.join(sdir, key))
            pd.DataFrame({"name": all_names, "type": ["SEEG"] * len(names) + ["EEG", "ECG"],
                          "units": "V", "low_cutoff": "0.0", "high_cutoff": str(fs / 2), "description": "n/a",
                          "sampling_frequency": str(float(fs)), "status": "good",
                          "status_description": "n/a"}).to_csv(os.path.join(d, f"{stem}_channels.tsv"), sep="\t",
                                                               index=False, encoding="utf-8-sig")
            with open(os.path.join(d, f"{stem}_events.tsv"), "w", encoding="utf-8-sig") as fh:
                fh.write(f"onset\tduration\ttrial_type\tvalue\tsample\n{ONSET}\t40.0\tictal\t1\t{int(ONSET * fs)}\n")
            with open(os.path.join(d, f"{stem}_ieeg.json"), "w") as fh:
                json.dump({"SamplingFrequency": fs, "RecordingDuration": DUR, "PowerLineFrequency": "n/a"}, fh)
            cons = [names[i] for i in onset_idx]
            each = [cons, cons + [names[onset_idx[-1] + 1]], cons[:2]]
            approx = float(task[5:]) + 0.37
            rows.append({"subject": sub, "patient": pat, "site": "HUP", "task": task, "run": "00", "stem": stem,
                         "annex_key": key, "edf_bytes": len(blob), "fs": float(fs), "duration_s": DUR,
                         "n_channels": len(all_names), "events_onset_s": ONSET, "approximate_onset": approx,
                         "ueo_time_consensus": approx + 0.6, "ueo_rel_s": 0.6, "ueo_consensus": json.dumps(cons),
                         "ueo_each": json.dumps(each), "clinicians": json.dumps(["Clin 1", "Clin 2", "Clin 3"]),
                         "n_clinicians": 3, "sec_consensus": json.dumps(cons), "lvfa": "yes" if kind == "lvfa" else "no"})
    pd.DataFrame(rows).to_csv(os.path.join(bdir, "seizures.csv"), index=False)
    print(f"Fake expert benchmark: {len(rows)} seizures of {args.patients} patients in {bdir} "
          f"(recordings in {os.path.dirname(os.path.dirname(sdir))})")


if __name__ == "__main__":
    main()
