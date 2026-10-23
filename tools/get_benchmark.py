"""
tools/get_benchmark.py  -  download the EXPERT BENCHMARK: seizures whose onset channels were marked by
epileptologists (the test that decides which onset method step 2 uses - see tools/benchmark_onset.py).

The seizures come from "Multi-Expert Seizure Annotation" (Penn + CHOP, CC-BY 4.0; NEMAR nm000402 =
Pennsieve dataset 530, Ojemann et al. 2026): 3-5 board-certified epileptologists marked, for every seizure, the
time and the channels of the unequivocal onset. The project folder benchmark/ already holds their marks and the
small tables (benchmark/README.txt); this script downloads only the recordings, and of each recording only the
63 seconds that are needed (32 s before to 31 s after the marked onset, like our own clips) - about 0.8 GB for
the 40 adult (HUP) seizures instead of 4 GB.

Run from the project folder:
    python tools\\get_benchmark.py --test      quick connection test (downloads a few kB)
    python tools\\get_benchmark.py             the 40 adult (HUP) seizures, about 0.8 GB, 10-40 minutes
    python tools\\get_benchmark.py --all       also the 3 children's (CHOP) seizures
Stop it at any time; started again it continues where it stopped. Every file is checked after download.
Saved in CACHE_DIR\\benchmark\\clips\\ (config.py; C:\\Users\\<you>\\ez_data\\benchmark\\clips on Windows).
"""
import argparse
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SOZ"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402

VERSION = 6
BASE_URL = "https://nemar.s3.us-east-2.amazonaws.com/nm000402/objects/"   # public copy of the recordings
OTHER_URLS = ["https://nemar.s3.amazonaws.com/nm000402/objects/",          # the same files, other addresses
              "https://s3.us-east-2.amazonaws.com/nemar/nm000402/objects/"]
BEFORE_S, AFTER_S = 32.0, 31.0
CHUNK = 4 * 1024 * 1024
UNIT_TO_V = {"uv": 1e-6, "\u00b5v": 1e-6, "mv": 1e-3, "v": 1.0, "nv": 1e-9}


def bench_dir():
    return config.BENCHMARK_DIR


def clips_dir():
    return os.path.join(config.CACHE_DIR, "benchmark", "clips")


def _opener(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost"):   # the smoke test's local server: never through a proxy
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def get_range(url, start, end, retries=6, timeout=90):
    """Bytes start..end (inclusive) of a file on a web server (HTTP range request), with retries."""
    want = end - start + 1
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                                       "User-Agent": "ez-localization-starter/6"})
            with _opener(url).open(req, timeout=timeout) as r:
                status = getattr(r, "status", 200)
                data = r.read(want + 1) if status == 206 else r.read()
            if status == 206 and len(data) == want:
                return data
            if status == 200 and len(data) > end:        # the server ignored the range: cut it out
                return data[start:end + 1]
            last = IOError(f"server sent {len(data)} bytes (status {status}), expected {want}")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 416):
                raise IOError(f"HTTP {e.code} {e.reason} for {url}") from None
            last = e
        except (urllib.error.URLError, OSError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(min(30, 2 ** attempt))
    raise IOError(f"download failed after {retries} tries: {last}")


def read_header(url, retries=6):
    """EDF header of a recording on the server."""
    h0 = get_range(url, 0, 255, retries=retries)
    if not h0[:1] == b"0":
        raise IOError("not an EDF file (first byte is not '0')")
    n_header = int(h0[184:192].decode("ascii").strip())
    ns = int(h0[252:256].decode("ascii").strip())
    if n_header != 256 + 256 * ns:
        raise IOError(f"EDF header size {n_header} does not fit {ns} signals")
    h = h0 + get_range(url, 256, n_header - 1, retries=retries)

    def field(off, width):
        return [h[off + i * width: off + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]

    o = 256
    labels = field(o, 16); o += 16 * ns
    o += 80 * ns                                            # transducer type
    dims = field(o, 8); o += 8 * ns
    pmin = [float(v) for v in field(o, 8)]; o += 8 * ns
    pmax = [float(v) for v in field(o, 8)]; o += 8 * ns
    dmin = [float(v) for v in field(o, 8)]; o += 8 * ns
    dmax = [float(v) for v in field(o, 8)]; o += 8 * ns
    o += 80 * ns                                            # prefiltering
    nsamp = [int(v) for v in field(o, 8)]
    return {"n_header": n_header, "ns": ns, "n_records": int(h0[236:244].decode("ascii").strip()),
            "record_s": float(h0[244:252].decode("ascii").strip()), "reserved": h0[192:236].decode("latin-1").strip(),
            "labels": labels, "dims": dims, "pmin": pmin, "pmax": pmax, "dmin": dmin, "dmax": dmax, "nsamp": nsamp}


def fetch_clip(url, size, onset, before=BEFORE_S, after=AFTER_S, progress=None):
    """The EDF records around the onset, decoded. Returns (dict for np.savez, info text)."""
    hd = read_header(url)
    if hd["reserved"].upper().startswith("EDF+D"):
        raise IOError("discontinuous EDF+D recording - not supported")
    rec_bytes = 2 * sum(hd["nsamp"])
    n_rec = hd["n_records"]
    if n_rec < 0:
        n_rec = (size - hd["n_header"]) // rec_bytes
    if size and hd["n_header"] + n_rec * rec_bytes != size:
        raise IOError(f"file size {size} does not match its header ({hd['n_header']} + {n_rec} x {rec_bytes})")
    rs = hd["record_s"]
    k0 = max(0, int(math.floor((onset - before) / rs)))
    k1 = min(n_rec, int(math.ceil((onset + after) / rs)))
    if k1 <= k0:
        raise IOError("the onset is outside the recording")
    first, last = hd["n_header"] + k0 * rec_bytes, hd["n_header"] + k1 * rec_bytes - 1
    parts, pos = [], first
    while pos <= last:
        end = min(last, pos + CHUNK - 1)
        parts.append(get_range(url, pos, end))
        pos = end + 1
        if progress:
            progress(pos - first, last - first + 1)
    raw = np.frombuffer(b"".join(parts), dtype="<i2").reshape(k1 - k0, sum(hd["nsamp"]))
    offs = np.concatenate([[0], np.cumsum(hd["nsamp"])])
    is_sig = [not lb.upper().startswith("EDF ANNOTATION") for lb in hd["labels"]]
    common = pd.Series([n for n, s in zip(hd["nsamp"], is_sig) if s]).mode().iloc[0]
    keep = [i for i in range(hd["ns"]) if is_sig[i] and hd["nsamp"][i] == common]
    data = np.stack([raw[:, offs[i]:offs[i + 1]].reshape(-1) for i in keep])
    gain, offset = [], []
    for i in keep:
        g = (hd["pmax"][i] - hd["pmin"][i]) / ((hd["dmax"][i] - hd["dmin"][i]) or 1.0)
        unit = UNIT_TO_V.get(hd["dims"][i].strip().lower(), 1e-6)
        gain.append(g * unit)
        offset.append((hd["pmin"][i] - hd["dmin"][i] * g) * unit)
    out = {"version": VERSION, "data": data.astype(np.int16), "gain_v": np.array(gain), "offset_v": np.array(offset),
           "names": np.array([hd["labels"][i] for i in keep], dtype=str), "fs": float(common / rs),
           "t_start": float(k0 * rs), "onset": float(onset), "records": np.array([k0, k1, n_rec]),
           "record_s": rs}
    skipped = hd["ns"] - len(keep)
    return out, f"{len(keep)} channels at {common / rs:g} Hz" + (f" ({skipped} other signals skipped)" if skipped else "")


def load_clip(path):
    """A downloaded clip: (x volts channels x samples, names, fs, t_start, onset) - times on the recording clock."""
    with np.load(path, allow_pickle=False) as z:
        x = z["data"].astype(np.float64) * z["gain_v"][:, None] + z["offset_v"][:, None]
        return x, [str(n) for n in z["names"]], float(z["fs"]), float(z["t_start"]), float(z["onset"])


def clip_ok(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return int(z["version"]) == VERSION and z["data"].shape[1] > 0
    except Exception:  # noqa: BLE001
        return False


def working_base(first_key, base):
    """The first server address that answers (the default one, then the alternatives). Returns (base, header)."""
    tried = []
    for b in [base] + ([u for u in OTHER_URLS if u != base] if base == BASE_URL else []):
        try:
            return b, read_header(b + first_key, retries=2)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{urllib.parse.urlparse(b).netloc}: {type(e).__name__}: {e}")
    raise IOError("no server address answered:\n    " + "\n    ".join(tried))


def seizure_table(include_children=False):
    m = pd.read_csv(os.path.join(bench_dir(), "seizures.csv"), dtype={"run": str})
    return m if include_children else m[m["site"] == "HUP"].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="only check the connection (a few kB)")
    ap.add_argument("--all", action="store_true", help="also the children's (CHOP) seizures")
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)   # the smoke test's local server
    args = ap.parse_args()
    m = seizure_table(args.all)
    out_dir = clips_dir()
    os.makedirs(out_dir, exist_ok=True)
    base = args.base_url.rstrip("/") + "/"

    r0 = m.iloc[0]
    print(f"Connection test: reading the header of {r0['stem']} ...")
    try:
        base, hd = working_base(r0["annex_key"], base)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"CONNECTION FAILED: {e}\nCheck your internet connection (a university network or proxy "
                         "may block it) and try again; if it keeps failing, send me outputs\\logs\\get_benchmark.txt")
    print(f"CONNECTION OK ({urllib.parse.urlparse(base).netloc}) - {hd['ns']} signals, {hd['n_records']} records of "
          f"{hd['record_s']:g} s.")
    if args.test:
        print("Now run:  python tools\\get_benchmark.py")
        return

    total = float((m["edf_bytes"] * (BEFORE_S + AFTER_S) / m["duration_s"]).sum())
    print(f"{len(m)} expert-marked seizures from {m['patient'].nunique()} patients; about {total / 1e9:.2f} GB to "
          f"download (only {BEFORE_S + AFTER_S:g} s of each recording). Saved in {out_dir}")
    rows, t_all = [], time.time()
    for i, r in m.iterrows():
        dest = os.path.join(out_dir, f"{r['stem']}.npz")
        tag = f"[{i + 1}/{len(m)}] {r['subject']} {r['task']}"
        if os.path.exists(dest) and clip_ok(dest):
            print(f"  {tag}: already downloaded")
            rows.append({"stem": r["stem"], "result": "ok (already there)"})
            continue
        t0 = time.time()

        def progress(done, size, tag=tag):
            print(f"\r  {tag}: {done / 1e6:6.1f} / {size / 1e6:.1f} MB", end="", flush=True)

        try:
            clip, info = fetch_clip(base + r["annex_key"], int(r["edf_bytes"]), float(r["events_onset_s"]),
                                    progress=progress)
            names = {soz_lib.normalize_name(n) for n in clip["names"]}
            ch = pd.read_csv(os.path.join(bench_dir(), "sidecars", r["subject"], f"{r['stem']}_channels.tsv"),
                             sep="\t", encoding="utf-8-sig", dtype=str, keep_default_na=False)
            missing = [n for n in ch["name"] if soz_lib.normalize_name(n) not in names]
            if len(missing) > 0.1 * len(ch):
                raise IOError(f"{len(missing)} of the {len(ch)} channels of channels.tsv are not in the recording")
            tmp = dest[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, **clip)
            os.replace(tmp, dest)
            if not clip_ok(dest):
                raise IOError("the saved file could not be read back")
            print(f"\r  {tag}: OK - {info}, {time.time() - t0:.0f} s" + " " * 20)
            rows.append({"stem": r["stem"], "result": "ok"})
        except KeyboardInterrupt:
            print("\nStopped. Run the same command again to continue.")
            raise SystemExit(1)
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the others
            print(f"\r  {tag}: FAILED - {type(e).__name__}: {e}" + " " * 10)
            rows.append({"stem": r["stem"], "result": f"failed: {type(e).__name__}: {e}"})
    log = pd.DataFrame(rows)
    log.to_csv(os.path.join(os.path.dirname(out_dir), "download_log.csv"), index=False)
    ok = log["result"].str.startswith("ok").sum()
    print(f"\n{ok} of {len(m)} seizures ready ({(time.time() - t_all) / 60:.1f} min).")
    if ok < len(m):
        print("Some downloads failed - run the same command again (finished ones are kept). If they keep failing, "
              "send me the log in outputs\\logs.")
    else:
        print("BENCHMARK DOWNLOAD COMPLETE. Next:  python tools\\benchmark_onset.py")


if __name__ == "__main__":
    with runlog.Log("get_benchmark"):
        main()
