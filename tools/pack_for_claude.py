"""
tools/pack_for_claude.py  -  pack the PROCESSED results for a check by Claude (no raw recordings, no MRI).

Run from the project folder:
    python tools\\pack_for_claude.py
Output: outputs\\for_claude\\review_part01.zip, review_part02.zip, ...   (each at most 24 MB, each complete on its own)
Upload ALL the review_part*.zip files to Claude (you can select them all at once).

What goes in:
    SOZ\\results\\                  every table and report of the SOZ steps, and seizure_arrays\\ (the numbers step 2
                                   computed from each seizure: about 0.5-1 MB per seizure)
    <data folder>\\benchmark\\arrays  the same numbers for the 40 expert-marked benchmark seizures
    outputs\\logs\\                 what every script printed
    metadata_3T_7T.xlsx            the dataset's clinical table (for the side / lobe check)
"""
import hashlib
import os
import sys
import zipfile


def project_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(here)):
        if os.path.exists(os.path.join(d, "config.py")):
            return d
    raise SystemExit("Put this file in the project's tools folder and run it from the project folder:\n"
                     "    python tools\\pack_for_claude.py")


HERE = project_dir()
sys.path.insert(0, HERE)
import config  # noqa: E402
import runlog  # noqa: E402

LIMIT = 24 * 1000 * 1000
OUT = os.path.join(config.OUT_DIR, "for_claude")


def walk(folder, prefix, skip_dirs=()):
    out = []
    if not os.path.isdir(folder):
        return out
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
        for f in sorted(files):
            if f.endswith(".tmp.npz"):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, folder).replace(os.sep, "/")
            out.append((p, f"{prefix}/{rel}"))
    return out


def find_metadata():
    for root, _, files in os.walk(config.CACHE_DIR):
        if "metadata_3T_7T.xlsx" in files:
            return os.path.join(root, "metadata_3T_7T.xlsx")
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    items = walk(config.SOZ_RESULTS_DIR, "SOZ_results")
    bench = os.path.join(config.CACHE_DIR, "benchmark")
    items += walk(os.path.join(bench, "arrays"), "benchmark_arrays")
    if os.path.exists(os.path.join(bench, "download_log.csv")):
        items.append((os.path.join(bench, "download_log.csv"), "benchmark_arrays/download_log.csv"))
    items += walk(os.path.join(config.OUT_DIR, "logs"), "logs")
    meta = find_metadata()
    if meta:
        items.append((meta, "metadata/metadata_3T_7T.xlsx"))
    n_arr = sum(1 for _, a in items if a.startswith("SOZ_results/seizure_arrays/") and a.endswith(".npz"))
    n_bench = sum(1 for _, a in items if a.startswith("benchmark_arrays/") and a.endswith(".npz"))
    print(f"Found {len(items)} files: {n_arr} seizure arrays of your patients, {n_bench} benchmark seizure arrays"
          f"{', the clinical table' if meta else ' (clinical table not found)'}.")
    if n_arr == 0:
        print("WARNING: no seizure arrays found in " + os.path.join(config.SOZ_RESULTS_DIR, "seizure_arrays"))
    if n_bench == 0:
        print("WARNING: no benchmark arrays found in " + os.path.join(bench, "arrays"))

    # files go into parts in order, a new part when the next file would make the part bigger than LIMIT
    parts, cur, size = [], [], 0
    for p, arc in items:
        s = os.path.getsize(p)
        if cur and size + s > LIMIT:
            parts.append(cur)
            cur, size = [], 0
        cur.append((p, arc, s))
        size += s
    if cur:
        parts.append(cur)

    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.startswith("review_part") and f.endswith(".zip"):
            os.remove(os.path.join(OUT, f))
    manifest = ["part\tbytes\tsha256\tfile"]
    for k, part in enumerate(parts, 1):
        for p, arc, s in part:
            manifest.append(f"{k}\t{s}\t{sha256(p)}\t{arc}")
    text = "\n".join(manifest) + "\n"
    total = 0
    for k, part in enumerate(parts, 1):
        path = os.path.join(OUT, f"review_part{k:02d}.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("MANIFEST.tsv", f"# {len(parts)} parts\n" + text)
            for p, arc, s in part:
                z.write(p, arc)
        mb = os.path.getsize(path) / 1e6
        total += mb
        print(f"  review_part{k:02d}.zip  {mb:5.1f} MB  ({len(part)} files)")
        if max(s for _, _, s in part) > LIMIT:
            print("    (larger than 24 MB because one file is that big - try to upload it anyway)")
    print(f"\nDone: {len(parts)} files, {total:.0f} MB in total, in {OUT}")
    print("Please upload ALL the review_part*.zip files to Claude.")


if __name__ == "__main__":
    with runlog.Log("pack_for_claude"):
        main()
