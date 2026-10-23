"""
aws_data.py  -  the map of the Penn 3T-7T dataset, and the code that fetches files from it.

The dataset is public on AWS (Registry of Open Data): no account, no cost.
    s3://edots-prod-aod-discover-publish50-use1/662/files/

What is in it, and what this project uses:
    derivatives/sub-PIDxx/freesurfer3T/sub-PIDxx/mri/...    FreeSurfer of the research 3T MRI   USED (MRI input)
        aparc_aseg.mgz (= the usual "aparc+aseg.mgz"; '+' is written '_' in this dataset), norm.mgz,
        T2.mgz (missing for some patients). There is no stats/ folder (no thickness tables).
    derivatives/sub-PIDxx/freesurfer7T/sub-PIDxx/...        FreeSurfer of the 7T MRI            later (FIELD = "7T")
    derivatives/sub-PIDxx/recon/ieeg_recon/module2/...      electrode coordinates (iEEG-recon)  USED
        reference MRI: MRI_RAS/..._space-T00mri_T1w_defaced.nii.gz (newer patients) or
                       ..._space-T00mri_T1w_ras_defaced.nii.gz directly in module2 (older patients)
    derivatives/sub-PIDxx/fmriprep3T, fmriprep7T            fMRI processing                     skipped
    sourcedata/sub-PIDxx/ses-clinical01/ieeg/*task-ictal*   seizure recordings + their tables   USED (SOZ labels)
    sourcedata/sub-PIDxx/ses-clinical01/ieeg/*interictal*   recordings without seizures         skipped
    sourcedata/sub-PIDxx/ses-clinical01/anat, ct            clinical MRI, post-implant CT       skipped
    sourcedata/sub-PIDxx/ses-research3T, ses-research7T     raw MRI and fMRI                    skipped
    metadata_3T_7T.xlsx, Imaging_Metadata.xlsx              patient / scan information          read

Files are downloaded into CACHE_DIR (config.py) with the same folder structure, only when a script
needs them, and reused afterwards. With DATA_SOURCE = "local" nothing is downloaded and nothing is
ever deleted: the files are read from LOCAL_DATASET_DIR.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

import config

_RUN_RE = re.compile(r"_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>[0-9]+)_"
                     r"(?P<suffix>ieeg\.edf|channels\.tsv|events\.tsv|ieeg\.json)$")
# FreeSurfer files used by the project, with the other spellings found in datasets ('+' -> '_')
FS_ALTERNATIVES = {"aparc+aseg.mgz": ["aparc+aseg.mgz", "aparc_aseg.mgz"]}
# reference MRI of the electrode coordinates, most specific pattern first: (pattern, exclude)
_T1_PATTERNS = [
    (r"/MRI_RAS/[^/]*T1w[^/]*\.nii(\.gz)?$", r"electrode_spheres|space-T01ct"),
    (r"/module2/[^/]*space-T00mri_T1w(_ras)?(_defaced)?\.nii(\.gz)?$", None),
    (r"/MRI_RAS/[^/]*\.nii(\.gz)?$", r"electrode_spheres|space-T01ct"),
]

_client = None
_missing = set()


# --------------------------------------------------------------------------- low level
def _s3():
    global _client
    if _client is None:
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError:
            raise SystemExit("boto3 is not installed. Run:  python -m pip install boto3")
        _client = boto3.client(
            "s3", region_name="us-east-1", endpoint_url=config.S3_ENDPOINT or None,
            config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10, "mode": "standard"},
                          connect_timeout=30, read_timeout=300, max_pool_connections=16))
    return _client


def local_path(rel):
    """Where a dataset file (path relative to the dataset's 'files' folder) lives on this computer."""
    return os.path.join(config.DATA_ROOT, *rel.replace("\\", "/").strip("/").split("/"))


def listing(rel_prefix, recursive=False):
    """Files and sub-folders under rel_prefix (e.g. 'derivatives/sub-PID02/').
    Returns (files, folders): files = [(rel_path, size_bytes)], folders = [rel_path ending in '/']."""
    rel_prefix = rel_prefix.replace("\\", "/")
    files, folders = [], []
    if config.DATA_SOURCE == "aws":
        kwargs = {"Bucket": config.S3_BUCKET, "Prefix": config.S3_PREFIX + rel_prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"
        n = len(config.S3_PREFIX)
        for page in _s3().get_paginator("list_objects_v2").paginate(**kwargs):
            files += [(o["Key"][n:], o["Size"]) for o in page.get("Contents", [])]
            folders += [p["Prefix"][n:] for p in page.get("CommonPrefixes", [])]
        return files, folders
    base = local_path(rel_prefix)
    if not os.path.isdir(base):
        return files, folders
    if recursive:
        for dirpath, _, names in os.walk(base):
            for f in names:
                p = os.path.join(dirpath, f)
                files.append((os.path.relpath(p, config.DATA_ROOT).replace(os.sep, "/"), os.path.getsize(p)))
    else:
        for entry in sorted(os.scandir(base), key=lambda e: e.name):
            rel = os.path.relpath(entry.path, config.DATA_ROOT).replace(os.sep, "/")
            if entry.is_dir():
                folders.append(rel + "/")
            else:
                files.append((rel, entry.stat().st_size))
    return files, folders


def fetch(rel, missing_ok=False, quiet=True):
    """Make sure one dataset file is on this computer and return its local path.
    In 'aws' mode it is downloaded the first time (to CACHE_DIR); afterwards the local copy is used.
    Returns None for a file that does not exist when missing_ok=True."""
    rel = rel.replace("\\", "/").strip("/")
    dest = local_path(rel)
    if os.path.exists(dest):
        return dest
    if config.DATA_SOURCE != "aws" or rel in _missing:
        if missing_ok:
            return None
        raise FileNotFoundError(f"Not found in the dataset: {rel}")
    from botocore.exceptions import ClientError
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        _s3().download_file(config.S3_BUCKET, config.S3_PREFIX + rel, tmp)
    except ClientError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        code = str(e.response.get("Error", {}).get("Code"))
        if code in ("404", "NoSuchKey", "NotFound") or (missing_ok and code in ("403", "AccessDenied", "Forbidden")):
            _missing.add(rel)   # (a public bucket answers 403 instead of 404 when listing is not allowed)
            if missing_ok:
                return None
            raise FileNotFoundError(f"Not found in the dataset: {rel}")
        raise
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, dest)
    if not quiet:
        print(f"    downloaded {rel.split('/')[-1]} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


def read_head(rel, n_bytes=65536):
    """The first n_bytes of a dataset file, to read a header without downloading the whole file: from the local copy
    if there is one, otherwise a small ranged download from AWS (nothing is saved). None if the file does not exist."""
    rel = rel.replace("\\", "/").strip("/")
    dest = local_path(rel)
    if os.path.exists(dest):
        with open(dest, "rb") as fh:
            return fh.read(n_bytes)
    if config.DATA_SOURCE != "aws":
        return None
    from botocore.exceptions import ClientError
    try:
        obj = _s3().get_object(Bucket=config.S3_BUCKET, Key=config.S3_PREFIX + rel, Range=f"bytes=0-{n_bytes - 1}")
        return obj["Body"].read()
    except ClientError as e:
        if str(e.response.get("Error", {}).get("Code")) in ("404", "NoSuchKey", "NotFound", "403", "AccessDenied"):
            return None
        raise


def fetch_many(rels, missing_ok=True, workers=8):
    """Fetch many small files in parallel. Returns {rel: local path or None}."""
    rels = list(dict.fromkeys(r for r in rels if r))
    if config.DATA_SOURCE != "aws" or len(rels) < 2:
        return {r: fetch(r, missing_ok=missing_ok) for r in rels}
    _s3()  # create the client once, before the threads use it
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(lambda r: fetch(r, missing_ok=missing_ok), rels))
    return dict(zip(rels, paths))


def remove_downloaded(rel):
    """Delete a downloaded copy (only in 'aws' mode - your own local data is never deleted)."""
    if config.DATA_SOURCE != "aws":
        return
    p = local_path(rel)
    try:
        if os.path.exists(p):
            os.remove(p)
    except OSError as e:
        print(f"    (could not delete {p}: {e})")


# --------------------------------------------------------------------------- dataset map
def subjects():
    _, f1 = listing("derivatives/")
    _, f2 = listing("sourcedata/")
    names = {f.strip("/").split("/")[-1] for f in f1 + f2}
    return sorted(n for n in names if n.startswith("sub-"))


def _pick(files, pattern, exclude=None):
    hits = [f for f, _ in files if re.search(pattern, f) and not (exclude and re.search(exclude, f))]
    return sorted(hits)[0] if hits else ""


def recon_files(sub):
    """iEEG-recon files of one patient: electrode names, mm and voxel coordinates, reference MRI."""
    files, _ = listing(f"derivatives/{sub}/recon/ieeg_recon/module2/", recursive=True)
    out = {"names": _pick(files, r"_electrode_names\.txt$"),
           "coords_mm": _pick(files, r"_space-T00mri_desc-mm_electrodes\.txt$", exclude=r"brainshift"),
           "coords_vox": _pick(files, r"_space-T00mri_desc-vox_electrodes\.txt$", exclude=r"brainshift"),
           "t1": ""}
    for pattern, exclude in _T1_PATTERNS:
        out["t1"] = _pick(files, pattern, exclude)
        if out["t1"]:
            break
    return out


def freesurfer_files(sub, field=None):
    """Which FreeSurfer files exist for one patient: {'aparc': file name or '', 'norm': bool, 't2': bool}."""
    field = field or config.FIELD
    files, _ = listing(f"derivatives/{sub}/freesurfer{field}/{sub}/mri/")
    names = {f.split("/")[-1] for f, _ in files}
    aparc = next((n for n in FS_ALTERNATIVES["aparc+aseg.mgz"] if n in names), "")
    return {"aparc": aparc, "norm": "norm.mgz" in names, "t2": "T2.mgz" in names}


def describe_subject(sub):
    """Everything this project needs to know about one patient's files."""
    _, deriv = listing(f"derivatives/{sub}/")
    deriv = {d.strip("/").split("/")[-1] for d in deriv}
    _, sessions = listing(f"sourcedata/{sub}/")
    sessions = sorted(s.strip("/").split("/")[-1] for s in sessions)

    runs = {}
    for ses in [s for s in sessions if s.startswith("ses-clinical")]:
        files, _ = listing(f"sourcedata/{sub}/{ses}/ieeg/", recursive=True)
        for rel, size in files:
            m = _RUN_RE.search(rel)
            if not m:
                continue
            key = (ses, m.group("task"), m.group("run"))
            r = runs.setdefault(key, {"subject": sub, "session": ses, "task": m.group("task"),
                                      "run": m.group("run"), "edf": "", "edf_bytes": 0,
                                      "channels": "", "events": "", "json": ""})
            suffix = m.group("suffix")
            if suffix == "ieeg.edf":
                r["edf"], r["edf_bytes"] = rel, size
            elif suffix == "channels.tsv":
                r["channels"] = rel
            elif suffix == "events.tsv":
                r["events"] = rel
            else:
                r["json"] = rel
    for r in runs.values():
        task = r["task"].lower()
        r["kind"] = "ictal" if task.startswith("ictal") else ("interictal" if "interictal" in task else "other")

    recon = recon_files(sub) if "recon" in deriv else {"names": "", "coords_mm": "", "coords_vox": "", "t1": ""}
    has_fs = f"freesurfer{config.FIELD}" in deriv
    fs = freesurfer_files(sub) if has_fs else {"aparc": "", "norm": False, "t2": False}

    ictal = [r for r in runs.values() if r["kind"] == "ictal" and r["edf"]]
    row = {
        "subject": sub,
        "has_freesurfer3T": "freesurfer3T" in deriv,
        "has_freesurfer7T": "freesurfer7T" in deriv,
        f"fs{config.FIELD}_aparc": fs["aparc"],
        f"fs{config.FIELD}_norm": fs["norm"],
        f"fs{config.FIELD}_T2": fs["t2"],
        "sessions": " ".join(sessions),
        "n_seizure_recordings": len(ictal),
        "n_interictal_recordings": sum(r["kind"] == "interictal" for r in runs.values()),
        "seizure_data_GB": round(sum(r["edf_bytes"] for r in ictal) / 1e9, 3),
        "has_electrode_coords": bool(recon["coords_mm"] or recon["coords_vox"]),
        "recon_names": recon["names"], "recon_coords_mm": recon["coords_mm"],
        "recon_coords_vox": recon["coords_vox"], "recon_t1": recon["t1"],
    }
    missing = []
    if not has_fs:
        missing.append(f"no FreeSurfer {config.FIELD}")
    elif not (fs["aparc"] and fs["norm"]):
        missing.append(f"FreeSurfer {config.FIELD} without aparc+aseg / norm.mgz")
    if not ictal:
        missing.append("no seizure (iEEG) recordings")
    if not row["has_electrode_coords"]:
        missing.append("no electrode coordinates")
    elif not recon["t1"] and not recon["coords_mm"]:
        missing.append("no reference MRI for the voxel coordinates")
    row["usable"] = not missing
    row["reason"] = "ok" if not missing else "; ".join(missing)
    return row, sorted(runs.values(), key=lambda r: (r["kind"], r["task"], r["run"]))


def inventory(verbose=True):
    """Look at every patient. Returns (inventory DataFrame, recordings DataFrame)."""
    import pandas as pd
    subs = subjects()
    if not subs:
        raise SystemExit(f"No patients found under {'AWS ' + config.S3_BUCKET if config.DATA_SOURCE == 'aws' else config.DATA_ROOT}")
    if verbose:
        where = f"AWS s3://{config.S3_BUCKET}/{config.S3_PREFIX}" if config.DATA_SOURCE == "aws" else config.DATA_ROOT
        print(f"Reading the dataset map from {where} ({len(subs)} patients) ...")
    rows, recs = [], []
    if config.DATA_SOURCE == "aws":
        _s3()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(describe_subject, subs))
    else:
        results = [describe_subject(s) for s in subs]
    for row, runs in results:
        rows.append(row)
        recs += runs
    inv = pd.DataFrame(rows)
    rec = pd.DataFrame(recs, columns=["subject", "session", "task", "run", "kind", "edf", "edf_bytes",
                                      "channels", "events", "json"])
    return inv, rec


def freesurfer_rel(sub, fname, field=None):
    field = field or config.FIELD
    return f"derivatives/{sub}/freesurfer{field}/{sub}/{fname}"


def metadata_table():
    """The patients' clinical table (metadata_3T_7T.xlsx) as a DataFrame, or None if not available."""
    import pandas as pd
    path = fetch("metadata_3T_7T.xlsx", missing_ok=True)
    if not path:
        return None
    try:
        return pd.read_excel(path)
    except Exception:  # noqa: BLE001 - the clinical table is only used for a sanity check
        return None
