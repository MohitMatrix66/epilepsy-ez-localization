"""
tools/check_dataset_extras.py  -  what else is in the dataset that could make the model better? (version 8.1)

The dataset paper (Imtiaz et al., Scientific Data 2026, "A multimodal epilepsy dataset of paired 3-Tesla and 7-Tesla MRI
and intracranial EEG") describes more than this project uses so far. This tool reads only file LISTS of the dataset (and
the first few kilobytes of each FreeSurfer norm.mgz, for its voxel size) - nothing big is downloaded - and answers:
  for every patient
    * raw MRI scans: 3T T1w / T2w / FLAIR and 7T T1w / T2w (the paper: a 3D T2w at 3T, and at 7T a T2w of the
      hippocampus with 0.4 x 0.4 mm pixels) - can the T2 channel be given to EVERY patient?
    * FreeSurfer 3T and 7T: voxel size of norm.mgz (the paper: two 3T T1 protocols, 0.8 mm and 1 mm), T2.mgz,
      hippocampal-subfield files
    * iEEG-recon: brain-shift files, the atlas tables of module3
  for the whole dataset
    * the files at the top level, with the columns of every small table: the paper reports Engel surgical outcomes
      for 15 patients, but metadata_3T_7T.xlsx has no outcome column - is there another table?

Run from the project folder (in 'aws' mode it needs the internet; 1-3 minutes):
    python tools\\check_dataset_extras.py
Output: outputs/dataset_extras.csv (one row per patient) and outputs/dataset_extras.txt  <- send the .txt to Claude
"""
import os
import re
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402

import aws_data  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402

VERSION = "8.1"
T1_LIKE = {"T1w", "UNIT1", "MP2RAGE", "T1map", "inv-1", "inv-2"}
OUTCOME_WORDS = re.compile(r"outcome|engel|ilae|surg|resect|ablat|clinic|particip|phenotype|metadata", re.I)
TABLE_EXT = (".xlsx", ".xls", ".csv", ".tsv")


def mgh_voxel_size(raw):
    """Voxel size (mm) and dimensions from the first bytes of a FreeSurfer .mgz / .mgh file."""
    data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    if len(data) < 42:
        raise ValueError("file too short for an MGH header")
    _version, w, h, d, _frames, _type, _dof = struct.unpack(">7i", data[:28])
    good_ras = struct.unpack(">h", data[28:30])[0]
    zooms = struct.unpack(">3f", data[30:42]) if good_ras else (1.0, 1.0, 1.0)
    return [round(float(z), 3) for z in zooms], [w, h, d]


def fs_rel(sub, field, fname):
    return f"derivatives/{sub}/freesurfer{field}/{sub}/mri/{fname}"


def scan_name(rel):
    """(session, suffix, acq) of a raw scan path, e.g. ('ses-research3T', 'T2w', '3D')."""
    parts = rel.split("/")
    ses = next((p for p in parts if p.startswith("ses-")), "")
    name = re.sub(r"\.nii(\.gz)?$", "", parts[-1])
    suffix = name.split("_")[-1]
    acq = re.search(r"_acq-([A-Za-z0-9]+)", name)
    return ses, suffix, acq.group(1) if acq else ""


def one_patient(sub):
    row = {"subject": sub}
    try:
        files, _ = aws_data.listing(f"sourcedata/{sub}/", recursive=True)
        scans = [scan_name(f) + (f,) for f, _ in files if "/anat/" in f and re.search(r"\.nii(\.gz)?$", f)]
        ct = [f for f, _ in files if "/ct/" in f and re.search(r"\.nii(\.gz)?$", f)]

        def pick(ses_part, kinds):
            return [f"{acq + ' ' if acq else ''}{suf}" for ses, suf, acq, _ in scans
                    if ses_part in ses and suf in kinds]

        row["raw_3T_T1w"] = "; ".join(pick("research3T", T1_LIKE))
        row["raw_3T_T2w"] = "; ".join(pick("research3T", {"T2w"}))
        row["raw_3T_FLAIR"] = "; ".join(pick("research3T", {"FLAIR"}))
        row["raw_7T_T1w"] = "; ".join(pick("research7T", T1_LIKE))
        row["raw_7T_T2w"] = "; ".join(pick("research7T", {"T2w"}))
        known = T1_LIKE | {"T2w", "FLAIR"}
        row["raw_other"] = "; ".join(sorted({f"{ses.replace('ses-', '')}:{suf}" for ses, suf, _, _ in scans
                                             if suf not in known}))
        row["clinical_mri"] = sum("clinical" in ses for ses, _, _, _ in scans)
        row["ct"] = len(ct)
        for field in ("3T", "7T"):
            fs, _ = aws_data.listing(f"derivatives/{sub}/freesurfer{field}/{sub}/mri/")
            names = [f.split("/")[-1] for f, _ in fs]
            row[f"fs{field}"] = bool(names)
            row[f"fs{field}_T2"] = "T2.mgz" in names
            row[f"fs{field}_FLAIR"] = "FLAIR.mgz" in names
            row[f"fs{field}_hippocampal_subfields"] = "; ".join(sorted(n for n in names if re.search(r"hippo|HBT", n)))
            row[f"fs{field}_voxel_mm"] = ""
            if "norm.mgz" in names:
                raw = aws_data.read_head(fs_rel(sub, field, "norm.mgz"))
                if raw:
                    zooms, dims = mgh_voxel_size(raw)
                    row[f"fs{field}_voxel_mm"] = zooms[0] if max(zooms) - min(zooms) < 0.01 else "x".join(map(str, zooms))
                    row[f"fs{field}_dims"] = "x".join(map(str, dims))
        rec, _ = aws_data.listing(f"derivatives/{sub}/recon/", recursive=True)
        rec = [f for f, _ in rec]
        row["recon_files"] = len(rec)
        row["recon_brainshift"] = "; ".join(sorted(f.split("/")[-1] for f in rec if "brainshift" in f.lower()))
        row["recon_module3"] = "; ".join(sorted(f.split("/")[-1] for f in rec if "/module3/" in f))[:300]
    except Exception as e:  # noqa: BLE001 - one patient must not stop the overview
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def table_columns(rel, size):
    """Columns and number of rows of a small table file at the top level ('' if it cannot be read)."""
    if size > 5e6:
        return "(large file - not opened)"
    path = aws_data.fetch(rel, missing_ok=True)
    if not path:
        return "(could not be downloaded)"
    try:
        if rel.lower().endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None)
            return " | ".join(f"sheet '{k}': {len(v)} rows; columns: {', '.join(map(str, v.columns))}"
                              for k, v in sheets.items())
        df = pd.read_csv(path, sep="\t" if rel.lower().endswith(".tsv") else ",")
        return f"{len(df)} rows; columns: {', '.join(map(str, df.columns))}"
    except Exception as e:  # noqa: BLE001
        return f"(could not be read: {type(e).__name__})"


def names_of(df, col):
    return ", ".join(s.replace("sub-", "") for s in df.loc[df[col], "subject"]) if col in df and df[col].any() else "-"


def main():
    where = f"AWS s3://{config.S3_BUCKET}/{config.S3_PREFIX}" if config.DATA_SOURCE == "aws" else config.DATA_ROOT
    print(f"Reading the dataset's file lists from {where} ...")
    if config.DATA_SOURCE != "aws":
        print("NOTE: DATA_SOURCE is 'local' - only the files of your local copy can be seen.")
    subs = aws_data.subjects()
    if config.DATA_SOURCE == "aws":
        aws_data._s3()      # one client for all threads
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(one_patient, subs))
    else:
        rows = [one_patient(s) for s in subs]
    df = pd.DataFrame(rows)
    for c in ("raw_3T_T1w", "raw_3T_T2w", "raw_3T_FLAIR", "raw_7T_T1w", "raw_7T_T2w", "fs3T_voxel_mm", "fs7T_voxel_mm",
              "fs3T_hippocampal_subfields", "fs7T_hippocampal_subfields", "recon_brainshift", "raw_other"):
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    for c in ("fs3T", "fs7T", "fs3T_T2", "fs7T_T2", "fs3T_FLAIR", "fs7T_FLAIR"):
        df[c] = df[c].fillna(False).astype(bool) if c in df else False
    training = set()
    if os.path.exists(config.CONTACTS_CSV):
        training = set(pd.read_csv(config.CONTACTS_CSV, usecols=["subject"])["subject"])
    df["training_patient"] = df["subject"].isin(training)
    for c in ("raw_3T_T2w", "raw_3T_FLAIR", "raw_7T_T2w"):
        df["has_" + c] = df[c] != ""
    df["t2_could_be_added"] = ~df["fs3T_T2"] & df["has_raw_3T_T2w"]
    df["no_t2_at_all"] = ~df["fs3T_T2"] & ~df["has_raw_3T_T2w"]

    files, folders = aws_data.listing("")
    src_files, _ = aws_data.listing("sourcedata/")
    der_files, der_folders = aws_data.listing("derivatives/")
    top = [(f, s) for f, s in files + src_files + der_files]
    other_der = [f for f in der_folders if not f.rstrip("/").split("/")[-1].startswith("sub-")]

    out = []
    say = out.append
    n = len(df)
    say("=" * 100)
    say(f"DATASET EXTRAS (version {VERSION}) - what the dataset holds besides what the model uses   [{where}]")
    say("=" * 100)
    say(f"{n} patients; {int(df['training_patient'].sum())} are training patients (contacts.csv).")
    vs = df["fs3T_voxel_mm"].replace("", "?")
    say("\n1. FreeSurfer 3T voxel size (norm.mgz) - the paper: two 3T T1 protocols, 0.8 mm and 1 mm:")
    for v, g in df.groupby(vs):
        tr = g[g["training_patient"]]
        say(f"   {v} mm: {len(g)} patients ({', '.join(s.replace('sub-', '') for s in g['subject'])})"
            + (f"; training: {', '.join(s.replace('sub-', '') for s in tr['subject'])}" if len(tr) else ""))
    say("   (version 8.1 cuts every cube in mm, so both sizes give the same 32 mm cubes)")
    say(f"\n2. T2: FreeSurfer T2.mgz for {int(df['fs3T_T2'].sum())} of {n}; raw 3T T2w scan for "
        f"{int(df['has_raw_3T_T2w'].sum())} of {n}.")
    say(f"   no T2.mgz but a raw 3T T2w (a T2 channel for them would need a T2 -> T1 registration): "
        f"{names_of(df, 't2_could_be_added')}")
    say(f"   no T2 at all: {names_of(df, 'no_t2_at_all')}")
    tr_missing = df[df["training_patient"] & ~df["fs3T_T2"]]
    if len(tr_missing):
        say(f"   training patients without T2.mgz: {', '.join(s.replace('sub-', '') for s in tr_missing['subject'])}"
            " - the reason the model uses T1 only")
    say(f"\n3. FLAIR at 3T (the sequence radiologists use for sclerosis and dysplasia): raw scan for "
        f"{int(df['has_raw_3T_FLAIR'].sum())} of {n} ({names_of(df, 'has_raw_3T_FLAIR')}); FreeSurfer FLAIR.mgz "
        f"for {int(df['fs3T_FLAIR'].sum())}.")
    say(f"\n4. 7T: FreeSurfer 7T for {int(df['fs7T'].sum())} of {n}; raw 7T T2w for {int(df['has_raw_7T_T2w'].sum())} "
        f"({names_of(df, 'has_raw_7T_T2w')}); 7T voxel sizes: "
        + (", ".join(f"{v or '?'} mm: {k}" for v, k in df.loc[df['fs7T'], 'fs7T_voxel_mm'].value_counts().items())
           or "-"))
    kinds = sorted({k for v in df["raw_7T_T1w"] for k in v.split("; ") if k})
    say(f"   7T T1-type scans found: {', '.join(kinds) or '-'}")
    for field in ("3T", "7T"):
        col = f"fs{field}_hippocampal_subfields"
        have = df[df[col] != ""]
        say(f"\n5{'a' if field == '3T' else 'b'}. Hippocampal subfields in FreeSurfer {field}: {len(have)} of {n} patients"
            + (f" (e.g. {have[col].iloc[0][:150]})" if len(have) else ""))
    bs = df[df["recon_brainshift"] != ""]
    say(f"\n6. iEEG-recon: brain-shift files for {len(bs)} patients ({', '.join(s.replace('sub-', '') for s in bs['subject']) or '-'}); "
        f"module3 (atlas) files for {int((df.get('recon_module3', pd.Series('', index=df.index)).fillna('') != '').sum())}.")
    if len(bs):
        say(f"   e.g. {bs['subject'].iloc[0]}: {bs['recon_brainshift'].iloc[0][:200]}")
    say("\n7. Top-level files (outcome / surgery information?):")
    for f, s in top:
        name = f.split("/")[-1]
        flag = "  <- name suggests clinical information" if OUTCOME_WORDS.search(name) else ""
        say(f"   {f}  ({s / 1e3:.0f} kB){flag}")
        if name.lower().endswith(TABLE_EXT):
            say(f"      {table_columns(f, s)}")
    if other_der:
        say(f"   other folders in derivatives/: {', '.join(other_der)}")
    say("   (The paper reports Engel outcomes for 15 patients: 11 Engel I, 4 Engel II-IV. If no table above has an "
        "outcome column, the per-patient outcomes are not in the public files.)")
    errors = df[df["error"].notna()] if "error" in df else df.iloc[0:0]
    if len(errors):
        say("\nERRORS: " + "; ".join(f"{r.subject}: {r.error}" for r in errors.itertuples()))
    say("\nPer patient: outputs/dataset_extras.csv.   Send this text (outputs/dataset_extras.txt) to Claude.")

    text = "\n".join(out)
    print("\n" + text)
    df.to_csv(os.path.join(config.OUT_DIR, "dataset_extras.csv"), index=False)
    with open(os.path.join(config.OUT_DIR, "dataset_extras.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    with runlog.Log("check_dataset_extras"):
        main()
