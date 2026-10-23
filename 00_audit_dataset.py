"""
00_audit_dataset.py  -  OPTIONAL: list what is inside a dataset folder that is already on your computer
(by default the download cache). For the dataset on AWS use SOZ/01_scan_dataset.py instead.

Run:   python 00_audit_dataset.py            (uses DATA_ROOT from config.py)
       python 00_audit_dataset.py /some/folder

It writes outputs/audit_report.txt.
"""
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

import config
import runlog

KEYWORDS = ["soz", "onset", "seizure", "electrode", "coord", "channels", "contact", "recon",
            "ieeg", "edf", "mef", "events", "mask", "resect", "lesion", "engel", "ilae", "outcome",
            "clinical", "participants", "metadata", "ct"]
FS_FILES = ["mri/aparc+aseg.mgz|mri/aparc_aseg.mgz", "mri/norm.mgz", "mri/T2.mgz", "stats/lh.aparc.stats"]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else config.DATA_ROOT
    if not os.path.isdir(root):
        sys.exit(f"Folder not found: {root}\n-> set DATA_ROOT in config.py")

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    ext_count, ext_size = Counter(), Counter()
    subjects = set()
    hits = defaultdict(list)
    tables = []
    n_files = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            n_files += 1
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root)
            ext = f.lower().split(".", 1)[1] if "." in f else "(none)"
            ext_count[ext] += 1
            try:
                ext_size[ext] += os.path.getsize(path)
            except OSError:
                pass
            subjects.update(re.findall(r"sub-[A-Za-z0-9]+", rel))
            low = rel.lower()
            for kw in KEYWORDS:
                if kw == "ct":
                    if re.search(r"(^|[_/\-])ct([_.\-]|$)", low):
                        hits[kw].append(rel)
                elif kw in low and "/freesurfer" not in low:
                    hits[kw].append(rel)
            if f.lower().endswith((".tsv", ".csv", ".xlsx", ".xls")) and "/freesurfer" not in low:
                tables.append(path)

    say(f"Dataset root: {root}")
    say(f"Total files: {n_files}")
    say(f"Subjects found ({len(subjects)}): {', '.join(sorted(subjects))}")
    say("")
    say("File types (count, total size):")
    for ext, c in ext_count.most_common(25):
        say(f"  .{ext:<22} {c:>7}   {ext_size[ext] / 1e9:8.2f} GB")

    say("")
    say("FreeSurfer outputs per subject (3T / 7T):")
    for sub in sorted(subjects):
        row = []
        for field in ("3T", "7T"):
            base = config.FS_DIR_TEMPLATE.format(root=root, sub=sub, field=field)
            have = ["Y" if any(os.path.exists(os.path.join(base, g)) for g in f.split("|")) else "-"
                    for f in FS_FILES]
            row.append(f"{field}: " + "".join(have))
        say(f"  {sub:<12} " + "   ".join(row) + "   (aparc+aseg, norm, T2, stats)")

    say("")
    say("Files whose path mentions a keyword (outside FreeSurfer folders, max 15 each):")
    for kw in KEYWORDS:
        if hits[kw]:
            say(f"  [{kw}]  {len(hits[kw])} files")
            for rel in hits[kw][:15]:
                say(f"      {rel}")

    say("")
    say("Table files (first rows) - look here for SOZ / electrode / outcome columns:")
    for path in tables[:40]:
        rel = os.path.relpath(path, root)
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                sheets = pd.read_excel(path, sheet_name=None, nrows=5)
                for name, df in sheets.items():
                    say(f"  {rel}  [sheet: {name}]  columns: {list(df.columns)}")
                    say("    " + df.head(3).to_string().replace("\n", "\n    "))
            else:
                sep = "\t" if path.lower().endswith(".tsv") else ","
                df = pd.read_csv(path, sep=sep, nrows=5)
                say(f"  {rel}  columns: {list(df.columns)}")
                say("    " + df.head(3).to_string().replace("\n", "\n    "))
        except Exception as e:  # noqa: BLE001 - we only want a readable report
            say(f"  {rel}  (could not read: {e})")

    out = os.path.join(config.OUT_DIR, "audit_report.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    with runlog.Log("00_audit_dataset"):
        main()
