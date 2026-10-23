"""
tools/check_install.py  -  is everything in place before you start? (version 9)

Run from the project folder:
    python tools\\check_install.py              packages, project files, folders, disk space, downloaded files
    python tools\\check_install.py --network    also tests the connection to the dataset and the expert benchmark
    python tools\\check_install.py --fix        also deletes downloaded recordings that are damaged (they are then
                                               downloaded again automatically)
It ends with INSTALL OK or with a list of what to fix.
"""
import argparse
import hashlib
import importlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SOZ"))
sys.path.insert(0, os.path.join(HERE, "tools"))

PACKAGES = [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("mne", "mne"), ("nibabel", "nibabel"),
            ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl"), ("boto3", "boto3"),
            ("torch", "torch")]
MANIFEST = os.path.join(HERE, "tools", "FILES.sha256")
VERSION = "9"
EDITABLE = {"config.py"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def edf_problem(path):
    """'' when the EDF file is complete, otherwise what is wrong (size against its own header)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            h = fh.read(256)
            if len(h) < 256 or h[:1] != b"0":
                return "not an EDF file"
            n_header = int(h[184:192].decode("ascii").strip())
            ns = int(h[252:256].decode("ascii").strip())
            n_rec = int(h[236:244].decode("ascii").strip())
            fh.seek(256 + 216 * ns)
            nsamp = [int(fh.read(8).decode("ascii").strip()) for _ in range(ns)]
        if n_rec < 0:
            return ""
        want = n_header + n_rec * 2 * sum(nsamp)
        if size != want:
            return f"size {size} bytes, its header says {want}"
        with open(path, "rb") as fh:
            fh.seek(n_header)
            sample = fh.read(1 << 20)
        if sample.count(b"\xef\xbf\xbd") > 1000:
            return "damaged (converted as text)"
        return ""
    except Exception as e:  # noqa: BLE001
        return f"cannot be read ({type(e).__name__}: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true", help="also test the internet connections")
    ap.add_argument("--fix", action="store_true", help="delete damaged downloaded recordings")
    args = ap.parse_args()
    problems, notes = [], []
    print("=" * 78)
    print(f"CHECKING THE INSTALLATION (version {VERSION})")
    print("=" * 78)
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    if sys.version_info < (3, 9):
        problems.append("Python 3.9 or newer is needed")

    print("\nPackages:")
    missing = []
    for mod, pip_name in PACKAGES:
        try:
            m = importlib.import_module(mod)
            print(f"  {pip_name:<14} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {pip_name:<14} MISSING ({type(e).__name__})")
            missing.append(pip_name)
    if missing:
        problems.append("missing packages - run:  python -m pip install " + " ".join(missing))

    print("\nProject files:")
    if not os.path.exists(MANIFEST):
        problems.append("tools/FILES.sha256 is missing - the project folder is incomplete (unzip it again)")
    else:
        bad, changed, n = [], [], 0
        with open(MANIFEST, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                digest, rel = line.strip().split("  ", 1)
                n += 1
                path = os.path.join(HERE, *rel.split("/"))
                if not os.path.exists(path):
                    bad.append(f"{rel} (missing)")
                elif sha256(path) != digest:
                    (changed if rel in EDITABLE else bad).append(rel)
        print(f"  {n - len(bad)} of {n} files are the version {VERSION} files"
              + (f"; changed by you: {', '.join(changed)}" if changed else ""))
        if bad:
            problems.append(f"files missing or not version {VERSION}: " + ", ".join(bad[:12]) + (" ..." if len(bad) > 12 else "")
                            + f"  -> unzip the version {VERSION} zip into a NEW, empty folder")
        old = [f for f in ("SOZ/results/ei_arrays",) if os.path.exists(os.path.join(HERE, *f.split("/")))]
        if old:
            notes.append(f"old version 5 results found ({', '.join(old)}) - they are not used by version {VERSION}")

    import config
    print(f"\nFolders:\n  project  {HERE}\n  data     {config.CACHE_DIR}\n  results  {config.SOZ_RESULTS_DIR}")
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        test = os.path.join(config.CACHE_DIR, ".write_test")
        with open(test, "w") as fh:
            fh.write("ok")
        os.remove(test)
    except OSError as e:
        problems.append(f"cannot write to the data folder {config.CACHE_DIR}: {e}")
    free = shutil.disk_usage(config.CACHE_DIR).free / 1e9
    print(f"  free disk space: {free:.1f} GB")
    if free < 6:
        notes.append(f"only {free:.1f} GB free - the project needs about 4-6 GB (recordings, MRI, benchmark)")

    print("\nDownloaded files:")
    edfs = []
    for root, _, files in os.walk(os.path.join(config.CACHE_DIR, "files")):
        edfs += [os.path.join(root, f) for f in files if f.lower().endswith(".edf")]
    damaged = [(p, edf_problem(p)) for p in edfs]
    damaged = [(p, why) for p, why in damaged if why]
    print(f"  seizure recordings: {len(edfs)} ({len(edfs) - len(damaged)} complete)")
    for p, why in damaged[:10]:
        print(f"    DAMAGED: {os.path.basename(p)}: {why}")
    if damaged:
        if args.fix:
            for p, _ in damaged:
                os.remove(p)
            print(f"  deleted {len(damaged)} damaged recordings - they are downloaded again in step 2")
        else:
            problems.append(f"{len(damaged)} damaged recordings - run:  python tools\\check_install.py --fix")
    clips = os.path.join(config.CACHE_DIR, "benchmark", "clips")
    if os.path.isdir(clips):
        import get_benchmark
        npz = [f for f in os.listdir(clips) if f.endswith(".npz") and not f.endswith(".tmp.npz")]
        ok = sum(get_benchmark.clip_ok(os.path.join(clips, f)) for f in npz)
        print(f"  expert benchmark seizures: {ok} of {len(npz)} readable")
        if ok < len(npz):
            notes.append("some benchmark seizures are unreadable - delete them and run tools\\get_benchmark.py again")
    else:
        print("  expert benchmark seizures: none yet (tools\\get_benchmark.py)")

    if args.network:
        print("\nInternet connections:")
        try:
            import aws_data
            files, _ = aws_data.listing("")
            print(f"  Penn 3T-7T dataset on AWS: OK ({len(files)} files at the top level)")
        except Exception as e:  # noqa: BLE001
            problems.append(f"cannot reach the dataset on AWS: {type(e).__name__}: {e}")
        try:
            import get_benchmark
            r = get_benchmark.seizure_table().iloc[0]
            base, hd = get_benchmark.working_base(r["annex_key"], get_benchmark.BASE_URL)
            print(f"  expert benchmark server: OK ({get_benchmark.urllib.parse.urlparse(base).netloc}; "
                  f"{hd['ns']} signals in the first recording)")
        except Exception as e:  # noqa: BLE001
            problems.append(f"cannot reach the expert benchmark server: {e}")

    print("\n" + "=" * 78)
    for n in notes:
        print("NOTE: " + n)
    if problems:
        print("INSTALL PROBLEMS:")
        for p in problems:
            print("  - " + p)
        raise SystemExit(1)
    print("INSTALL OK" + ("" if args.network else "   (add --network to also test the internet connections)"))


if __name__ == "__main__":
    import runlog
    with runlog.Log("check_install"):
        main()
