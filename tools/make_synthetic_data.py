"""
make_synthetic_data.py  -  builds FAKE patients laid out exactly like the Penn 3T-7T dataset on AWS
(as seen in the first real run), so the whole project can be tested in minutes without real data.

    synthetic_data/files/
        metadata_3T_7T.xlsx                      clinical table (seizure onset side / lobe)
        sourcedata/participants.json
        sourcedata/sub-PIDxx/ses-clinical01/ieeg/sub-PIDxx_ses-clinical01_task-ictalNNN_run-0K_ieeg.edf
                                                  ..._channels.tsv  ..._events.tsv  ..._ieeg.json
        derivatives/sub-PIDxx/freesurfer3T/sub-PIDxx/mri/{aparc_aseg,norm,T2}.mgz      (no stats/ folder)
        derivatives/sub-PIDxx/recon/ieeg_recon/module2/   names, mm and voxel coordinates, reference MRI in
                    MRI_RAS/ (newer patients) or as ..._T1w_ras_defaced.nii.gz in module2 (older patients)
    synthetic_data/truth_contacts.csv    the planted answer (which contacts are SOZ)

What the fake data contains (each item mirrors something in the real dataset):
    * recordings start 30 s before the marked onset; events.tsv 'ictal 1' with a byte-order mark
    * EDF names 'LA01' but coordinate names 'LA1'; an ECG channel
    * a short burst of fast activity 2.5 s before onset on a non-SOZ contact (like interictal spikes)
    * PID01-03: old iEEG-recon layout (+ decoy MRIs);  PID04: no T2 image;  PID03: clinician SOZ marks
    * PID02: the SOZ electrode is 'RG' in the recordings but 'G' in the coordinate files (fixed by the
      CHANNEL_ALIASES entry for sub-PID02 in config.py, like the real PID02 grid)
    * PID05: every seizure starts somewhere else, and one contact always joins 4 s later (it is reached by every
      seizure but never starts one) -> no repeated onset, must be left out of training
    * PID06: only 2 seizures;  PID07: a 4x4 grid (rows must not be paired across);  PID08: names without
      L/R, scalp EEG channels 'P3', 'O1', ... recorded next to intracranial 'P03', and intracranial
      contacts O01-O08 typed EEG (must be kept)
    * PID09: one seizure that starts with rhythmic SLOW waves only (no fast activity: the version 5 EI could not see
      it, version 6 must find its start at the SOZ), clinical side written wrongly, and 4 electrodes running out
      through the skull (outer contacts outside the brain, patient must be kept)
    * PID10: seizures but no coordinates;  PID11, PID12: MRI only
Version 8.1 cases (the dataset paper: two 3T T1 protocols, 0.8 mm and 1 mm):
    * PID07, PID08, PID09, PID11, PID12: FreeSurfer volumes at 0.8 mm (320^3), the others at 1 mm (256^3) - the
      same brains, so a cube must show the same millimetres in both
    * raw scans in sourcedata/sub-PIDxx/ses-research3T/anat and ses-research7T/anat (tiny files, only their names
      matter): 3T T1w for all, 3T T2w for all but PID05 (so PID04 has a raw T2w but no FreeSurfer T2.mgz), FLAIR for
      PID11, 7T T1w for all, a 7T hippocampal T2w for PID01 and PID11
Version 9 cases (the side model and the sclerosis check):
    * the clinical table has the MRI report columns; PID11 and PID12 (MRI only, so never trained on) have
      'Mesial temporal sclerosis' on their side, and their hippocampus on that side is planted 'sick'
    * FreeSurfer surface files (surf/?h.thickness, label/?h.aparc.annot, fake vertices): the temporal cortex is a
      little thinner on the epileptic side (drawn from a separate random generator, so the rest of the data is
      unchanged)
Version 5 cases (several foci, early starts, one-off starts):
    * PID04: two foci - 5 of 7 seizures start at the planted SOZ, 2 at a second focus (role 'focus_b', also SOZ);
      both must be found
    * PID06: every seizure starts 4 s BEFORE the marked onset, and 3 contacts ('spread') join 4 s later: only the
      real SOZ may become SOZ
    * PID01: one extra seizure starts at 3 other contacts ('sporadic'): they must be 'uncertain', not SOZ
Version 6 cases (the onset detector):
    * PID04: 2 of its seizures look like the real sub-PID04: the big slow waves of the normal period fade 15 s
      before the seizure, and the seizure starts with big rhythmic slow waves (the version 5 EI fails on both)
    * PID07: one seizure has muscle activity at most contacts from 3 s before to 5 s after the onset
    * PID08: one seizure has a contact with electrode 'pops' (sudden jumps) around the onset
The fake brains are ellipsoids; results on this data mean NOTHING clinically.

Run:  python tools/make_synthetic_data.py --out synthetic_data --n 12
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common  # noqa: E402
from edf_writer import write_edf  # noqa: E402

VERSION = "9"
N = 256
TKR = np.array([[-1, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128], [0, 0, 0, 1]], dtype=float)
CORTEX_IDX = [k for k in range(1, 36) if k != 4]          # 34 Desikan-Killiany regions
SUBCORT = {  # name: (left id, right id, centre offset (di, dj, dk), radii)
    "hippocampus": (17, 53, (30, 22, -8), (6, 6, 14)),
    "amygdala": (18, 54, (28, 20, 12), (6, 6, 6)),
    "thalamus": (10, 49, (10, 2, -10), (8, 8, 10)),
    "caudate": (11, 50, (14, -12, 15), (5, 6, 8)),
    "putamen": (12, 51, (24, 0, 8), (5, 8, 10)),
    "pallidum": (13, 52, (18, 3, 5), (3, 4, 5)),
    "accumbens": (26, 58, (9, 9, 22), (3, 3, 3)),
}
TEMPORAL_LIKE = [6, 7, 9, 15, 16, 30, 33, 34]
FS = 256          # sampling rate of the fake recordings (Hz)
PRE = 30.0        # recording starts this long before the marked onset (like the real clips)
DUR = 60.0        # seconds per recording
ONSET = PRE
SCALP = ["P3", "P4", "O1", "C3", "CZ"]
OLD_LAYOUT, NO_T2, MARKED, MULTIFOCAL, TWO_SEIZURES, GRID, NO_PREFIX, SLOW_SEIZURE = (1, 2, 3), 4, 3, 5, 6, 7, 8, 9
ALIAS_PATIENT, SKULL_CONTACTS = 2, 9
HIRES, HIRES_MM = (7, 8, 9, 11, 12), 0.8      # version 8.1: FreeSurfer volumes at 0.8 mm for these patients
RAW_NO_T2, RAW_FLAIR, RAW_7T_T2 = (5,), (11,), (1, 11)
FORCE_MTS = (11, 12)     # version 9: hippocampal sclerosis on the MRI report (and a 'sick' hippocampus planted)
TWO_FOCI, EARLY_ONSET, SPORADIC = 4, 6, 1   # version 5 cases (see the list above)
STATE_CHANGE, MUSCLE, POPS = 4, 7, 8          # version 6 cases


def build_atlas():
    i, j, k = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij", sparse=True)
    di, dj, dk = (i - 128) / 68.0, (j - 128) / 60.0, (k - 128) / 85.0
    d = np.sqrt(di ** 2 + dj ** 2 + dk ** 2)
    labels = np.zeros((N, N, N), dtype=np.int32)
    left = np.broadcast_to(i >= 128, labels.shape)
    brain = d < 1.0
    labels[brain & left] = 2          # left white matter
    labels[brain & ~left] = 41        # right white matter
    shell = brain & (d >= 0.86)
    az = np.broadcast_to(np.arctan2(dk, np.abs(di) + 1e-6), labels.shape)
    el = np.broadcast_to(np.arctan2(-dj, np.sqrt(di ** 2 + dk ** 2) + 1e-6), labels.shape)
    a_bin = np.clip(((az + np.pi / 2) / np.pi * 7).astype(int), 0, 6)
    e_bin = np.clip(((el + np.pi / 2) / np.pi * 5).astype(int), 0, 4)
    cell = np.minimum(a_bin * 5 + e_bin, 33)
    parcel = np.array(CORTEX_IDX)[cell]
    labels[shell & left] = 1000 + parcel[shell & left]
    labels[shell & ~left] = 2000 + parcel[shell & ~left]
    for lid, rid, (oi, oj, ok), (ri, rj, rk) in SUBCORT.values():
        for lab, sign in ((lid, +1), (rid, -1)):
            ci, cj, ck = 128 + sign * oi, 128 + oj, 128 + ok
            blob = ((i - ci) / ri) ** 2 + ((j - cj) / rj) ** 2 + ((k - ck) / rk) ** 2 < 1
            labels[blob] = lab
    return labels


def make_mri(s, sub, labels, rng, root):
    """FreeSurfer-like files + planted 'epileptic' regions. Returns (affine, norm volume, soz regions, side)."""
    present = [r for r in common.REGION_IDS if (labels == r).any()]
    side = str(rng.choice(["L", "R"]))
    pool = ([17, 18] if side == "L" else [53, 54]) + [(1000 if side == "L" else 2000) + t for t in TEMPORAL_LIKE]
    soz = {r for r in rng.choice(pool, size=rng.integers(1, 3), replace=False).tolist() if r in present}
    if s in FORCE_MTS:           # (after the random draws, so every other patient stays exactly as before)
        soz.add(17 if side == "L" else 53)

    t1 = np.zeros(labels.shape, np.float32)
    t2 = np.zeros(labels.shape, np.float32)
    wm = np.isin(labels, [2, 41])
    cortex = labels >= 1000
    deep = (labels > 0) & ~wm & ~cortex
    t1[wm], t1[cortex], t1[deep] = 110, 72, 82
    t2[wm], t2[cortex], t2[deep] = 60, 92, 86
    off1 = np.zeros(labels.max() + 1, np.float32)
    off2 = np.zeros(labels.max() + 1, np.float32)
    off1[common.REGION_IDS] = rng.normal(0, 3, len(common.REGION_IDS))
    off2[common.REGION_IDS] = rng.normal(0, 3, len(common.REGION_IDS))
    for r in soz:
        off1[r] -= 3
        off2[r] += 4
    t1 += off1[labels]
    t2 += off2[labels]
    brain = labels > 0
    t1[brain] += rng.normal(0, 5, int(brain.sum())).astype(np.float32)
    t2[brain] += rng.normal(0, 5, int(brain.sum())).astype(np.float32)

    affine = TKR.copy()
    affine[:3, 3] += rng.normal(0, 8, 3)  # scanner space differs from tkr space
    mri = os.path.join(root, "derivatives", sub, "freesurfer3T", sub, "mri")
    os.makedirs(mri, exist_ok=True)
    norm = np.clip(t1, 0, 255).astype(np.uint8)
    lab_out, t1_out, t2_out, aff_out = labels, t1, t2, affine
    if s in HIRES:   # the same brain in 0.8 mm voxels (like the real patients of the 0.8 mm T1 protocol)
        lab_out, t1_out, t2_out = to_hires(labels, 0), to_hires(t1, 1), to_hires(t2, 1)
        aff_out = affine @ np.diag([HIRES_MM, HIRES_MM, HIRES_MM, 1.0])
    nib.save(nib.MGHImage(lab_out, aff_out), os.path.join(mri, "aparc_aseg.mgz"))      # '+' written as '_'
    nib.save(nib.MGHImage(np.clip(t1_out, 0, 255).astype(np.uint8), aff_out), os.path.join(mri, "norm.mgz"))
    if s != NO_T2:
        nib.save(nib.MGHImage(np.clip(t2_out, 0, 255).astype(np.uint8), aff_out), os.path.join(mri, "T2.mgz"))
    return affine, norm, soz, present, side


def to_hires(vol, order):
    """The 1 mm volume in 0.8 mm voxels over the same 256 mm (320^3): voxel i of the new volume is at 0.8 * i of the
    old one, so with the affine (old affine x 0.8) every point keeps its scanner coordinates."""
    from scipy import ndimage
    n = int(round(N / HIRES_MM))
    out = np.int32 if order == 0 else np.float32
    return ndimage.affine_transform(np.asarray(vol, dtype=out), np.full(3, HIRES_MM), offset=0.0, output_shape=(n, n, n),
                                    order=order, mode="constant", cval=0.0, output=out)


def make_surfaces(s, sub, side, root):
    """FreeSurfer surface files with fake vertices: thickness (temporal cortex 0.15 mm thinner on the epileptic side)
    and the Desikan-Killiany annotation. Only per-vertex values matter for the project, not the geometry."""
    r = np.random.default_rng(1000 + s)
    names = ["unknown"] + [n for n in common._DK if n]
    ctab = np.array([[(37 * i) % 256, (91 * i) % 256, (53 * i) % 256, 0, 0] for i in range(len(names))], dtype=np.int32)
    base = os.path.join(root, "derivatives", sub, "freesurfer3T", sub)
    os.makedirs(os.path.join(base, "surf"), exist_ok=True)
    os.makedirs(os.path.join(base, "label"), exist_ok=True)
    for hemi in ("lh", "rh"):
        vlab = np.arange(3000) % len(names)
        thick = 2.5 + r.normal(0, 0.2, len(vlab))
        thick[vlab == 0] = 0.0                                   # the medial wall has no thickness
        epileptic = (hemi == "lh") == (side == "L")
        temporal = np.isin(vlab, [names.index(n) for n in common.LOBES["temporal"]])
        thick[temporal & epileptic] -= 0.15
        nib.freesurfer.write_morph_data(os.path.join(base, "surf", f"{hemi}.thickness"), thick.astype(np.float32))
        nib.freesurfer.write_annot(os.path.join(base, "label", f"{hemi}.aparc.annot"), vlab.astype(np.int32), ctab,
                                   names, fill_ctab=True)


def make_raw_scans(s, sub, root):
    """Raw scans of the research sessions, laid out like the real dataset (tiny images: only the names matter here)."""
    tiny = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4))
    files = [("ses-research3T", "acq-3D_T1w"), ("ses-research7T", "T1w")]
    if s not in RAW_NO_T2:
        files.append(("ses-research3T", "acq-3D_T2w"))
    if s in RAW_FLAIR:
        files.append(("ses-research3T", "acq-3D_FLAIR"))
    if s in RAW_7T_T2:
        files.append(("ses-research7T", "acq-hippocampus_T2w"))
    for ses, suffix in files:
        d = os.path.join(root, "sourcedata", sub, ses, "anat")
        os.makedirs(d, exist_ok=True)
        nib.save(tiny, os.path.join(d, f"{sub}_{ses}_{suffix}.nii.gz"))


def _depth(labels, tgt, rng, n=8, spacing=3.5):
    vox = np.argwhere(labels == tgt)
    start = vox[rng.integers(len(vox))].astype(float)
    centre = np.array([128.0, 128.0, 128.0])
    direction = start - centre if tgt < 1000 else centre - start   # deep: go outwards; cortex: go inwards
    direction /= np.linalg.norm(direction) + 1e-9
    return [start + direction * spacing * c + rng.normal(0, 0.3, 3) for c in range(n)]


def _grid(labels, rng):
    """4 x 4 grid, 10 mm spacing, on a patch of cortex (numbered row by row)."""
    vox = np.argwhere(labels >= 1000)
    centre = np.array([128.0, 128.0, 128.0])
    while True:
        p = vox[rng.integers(len(vox))].astype(float)
        if abs(p[1] - 128) < 30 and abs(p[0] - 128) > 45:   # lateral surface, away from the midline
            break
    normal = (p - centre) / np.linalg.norm(p - centre)
    u = np.cross(normal, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return [p + (c - 1.5) * 10 * u + (r - 1.5) * 10 * v for r in range(4) for c in range(4)]


def _outward(labels, rng, n=8, spacing=3.5):
    """An electrode whose deepest contact is in the cortex and which runs OUT through the skull, so its outer
    contacts lie outside the brain - like the superficial end of real stereo-EEG electrodes."""
    vox = np.argwhere(labels >= 1000)
    start = vox[rng.integers(len(vox))].astype(float)
    d = start - np.array([128.0, 128.0, 128.0])
    d /= np.linalg.norm(d)
    return [start + d * spacing * c + rng.normal(0, 0.3, 3) for c in range(n)]


def make_contacts(s, sub, labels, affine, soz, present, rng):
    """7 electrodes (2 aimed at the SOZ regions); PID07 has a grid instead of the last electrode; PID09 has 4
    extra electrodes running out through the skull. 'contact' = name in the coordinate files, 'edf_name' =
    name in the recordings (zero-padded; for PID02 the SOZ electrode is 'RG' in the recordings but 'G' in
    the coordinate files, like the real PID02 grid)."""
    targets = list(soz) + list(rng.choice([r for r in present if r not in soz], 7 - len(soz), replace=False))
    targets += [None] * (4 if s == SKULL_CONTACTS else 0)
    rows = []
    for e_no, tgt in enumerate(targets):
        if tgt is None:
            pts, letter, kind = _outward(labels, rng), chr(65 + e_no), "SEEG"
        elif s == GRID and e_no == 6:
            pts, letter, kind = _grid(labels, rng), "G", "ECOG"
        else:
            pts, letter, kind = _depth(labels, tgt, rng), chr(65 + e_no), "SEEG"
        if s == NO_PREFIX and e_no == 0:
            letter = "P"   # an intracranial electrode called P, recorded next to the scalp channel P3
        if s == NO_PREFIX and e_no == 1:
            letter, kind = "O", "EEG"   # intracranial contacts O01-O08 typed EEG (like the real PID02)
        if s == ALIAS_PATIENT:
            letter = {0: "G", 6: "Q"}.get(e_no, letter)
        hemi = "" if s == NO_PREFIX else ("L" if pts[0][0] >= 128 else "R")
        for c, p in enumerate(pts):
            q = np.clip(np.round(p).astype(int), 0, N - 1)
            lab = int(labels[tuple(q)])
            xyz = nib.affines.apply_affine(affine, p)
            recon = f"{letter}{c + 1}" if (s == ALIAS_PATIENT and e_no == 0) else f"{hemi}{letter}{c + 1}"
            rows.append({"subject": sub, "contact": recon, "edf_name": f"{hemi}{letter}{c + 1:02d}",
                         "electrode": e_no, "type": kind, "x": xyz[0], "y": xyz[1], "z": xyz[2],
                         "soz": int(lab in soz)})
    return pd.DataFrame(rows)


def make_recon(s, sub, contacts, affine, norm, root, rng):
    """iEEG-recon files: names, mm and voxel coordinates ('x y z', 3 columns like the real files) in a reference
    MRI that is the same scan as FreeSurfer's, stored in RAS orientation."""
    d = os.path.join(root, "derivatives", sub, "recon", "ieeg_recon", "module2")
    os.makedirs(d, exist_ok=True)
    ras = nib.as_closest_canonical(nib.Nifti1Image(norm, affine))
    stem = f"{sub}_ses-research3T_acq-3D_space-T00mri_T1w"
    if s in OLD_LAYOUT:
        nib.save(ras, os.path.join(d, stem + "_ras_defaced.nii.gz"))
        decoy = nib.Nifti1Image(np.asarray(ras.dataobj)[::-1].copy(), ras.affine)          # wrong-space decoys
        nib.save(decoy, os.path.join(d, f"{sub}_ses-research3T_acq-3D_space-T01ct_T1w_defaced.nii.gz"))
        nib.save(decoy, os.path.join(d, stem + "_electrode_spheres.nii.gz"))
    else:
        os.makedirs(os.path.join(d, "MRI_RAS"), exist_ok=True)
        nib.save(ras, os.path.join(d, "MRI_RAS", stem + "_defaced.nii.gz"))
        nib.save(ras, os.path.join(d, stem + "_electrode_spheres.nii.gz"))
    xyz = contacts[["x", "y", "z"]].values
    ijk = nib.affines.apply_affine(np.linalg.inv(ras.affine), xyz)
    with open(os.path.join(d, f"{sub}_electrode_names.txt"), "w") as fh:
        fh.write("\n".join(contacts["contact"]) + "\n")
    np.savetxt(os.path.join(d, f"{sub}_ses-research3T_space-T00mri_desc-mm_electrodes.txt"), xyz)
    np.savetxt(os.path.join(d, f"{sub}_ses-research3T_space-T00mri_desc-vox_electrodes.txt"), ijk)
    if s in OLD_LAYOUT:
        np.savetxt(os.path.join(d, f"{sub}_ses-research3T_space-T00mri_desc-vox_electrodes_before_brainshift.txt"),
                   ijk + rng.normal(0, 3, ijk.shape))


def assign_roles(s, contacts):
    """Column 'role' for the special cases: 'soz', 'focus_b' (second SOZ, PID04), 'spread' (PID06), 'sporadic'
    (PID01), 'pop' (electrode pops, PID08), '' = other contacts."""
    c = contacts.copy().reset_index(drop=True)
    e, soz = c["electrode"].to_numpy(), c["soz"].to_numpy()
    role = np.where(soz == 1, "soz", "").astype(object)
    if s == TWO_FOCI:
        b = np.flatnonzero((e == 1) & (soz == 1))
        if len(b) == 0:
            b = np.flatnonzero(e == 1)[:3]
        role[b] = "focus_b"
        c.loc[b, "soz"] = 1
    if s == EARLY_ONSET:
        role[np.flatnonzero((e == 5) & (soz == 0))[:3]] = "spread"
    if s == SPORADIC:
        role[np.flatnonzero((e == 4) & (soz == 0))[:3]] = "sporadic"
    if s == POPS:
        role[np.flatnonzero((e == 3) & (soz == 0))[2:3]] = "pop"
    c["role"] = role
    return c


def _pink(rng, n_ch, n):
    white = rng.normal(size=(n_ch, n))
    f = np.fft.rfftfreq(n, 1 / FS)
    shape = 1 / np.sqrt(np.maximum(f, 1.0))
    x = np.fft.irfft(np.fft.rfft(white, axis=1) * shape, n=n, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _fast(rng, n):
    b, a = signal.butter(4, [25, 90], btype="bandpass", fs=FS)
    x = signal.filtfilt(b, a, rng.normal(size=n))
    return x / x.std()


def _band(rng, n, lo, hi):
    b, a = signal.butter(4, [lo, min(hi, 0.45 * FS)], btype="bandpass", fs=FS)
    x = signal.filtfilt(b, a, rng.normal(size=n))
    return x / x.std()


def make_recording(contacts, rng, starters, slow=False, burst=None, followers=(), lead=0.0, spread=(),
                   state_change=False, muscle=False, pops=None):
    """Fake iEEG (microvolts): pink background + alpha + 60 Hz line noise. In a seizure the `starters` begin
    fast (25-90 Hz) activity `lead` seconds before the marked onset, `followers` 4 s after the onset, `spread`
    at the onset, and 6 other contacts join 8-12 s after it; slow=True gives a seizure with only rhythmic slow
    activity (no fast activity). burst = contact with a 1-s burst of fast activity 2.5 s before onset.
    state_change=True: big slow waves everywhere that fade 15 s before the onset, and the starters begin big rhythmic
    slow waves (like the real sub-PID04). muscle=True: muscle activity at 70% of the contacts from -3 s to +5 s.
    pops = a contact with sudden jumps (electrode pops) around the onset."""
    n = int(DUR * FS)
    t = np.arange(n) / FS
    names = list(contacts["contact"])
    x = 40.0 * _pink(rng, len(names), n)
    x += 8.0 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi, (len(names), 1)))
    x += 6.0 * np.sin(2 * np.pi * 60 * t)
    if state_change:
        tr = t - ONSET
        fade = np.where(tr < -15, 1.0, np.where(tr < -13, 1 - 0.65 * (tr + 15) / 2, 0.35))
        x += 90.0 * np.stack([_band(rng, n, 0.8, 3.5) for _ in names]) * fade
    if slow:
        for c in starters:
            # a rhythmic discharge with sharp waves, big like the real ones (sub-PID25): line length sees it,
            # the EI does not (no fast activity)
            x[c] += 350.0 * (t > ONSET) * signal.sawtooth(2 * np.pi * 3 * t + rng.uniform(0, 6), 0.15)
    else:
        others = [c for c in range(len(names)) if c not in set(starters) | set(followers) | set(spread)]
        late = rng.choice(others, size=min(6, len(others)), replace=False)
        for c in list(starters) + list(followers) + list(spread) + list(late):
            start = ONSET + (-lead + rng.uniform(0, 0.4) if c in starters else
                             4.0 + rng.uniform(0, 0.2) if c in followers else
                             0.0 + rng.uniform(0, 0.2) if c in spread else rng.uniform(8, 12))
            ramp = np.clip((t - start) / 1.0, 0, 1) * (t < start + 25)
            if state_change and c in starters:
                x[c] += 400.0 * ramp * signal.sawtooth(2 * np.pi * 4.5 * t + rng.uniform(0, 6), 0.15)
            else:
                x[c] += 45.0 * ramp * _fast(rng, n)
    if muscle:
        emg = [c for c in range(len(names)) if c not in set(starters)]
        emg = rng.choice(emg, size=int(0.7 * len(names)), replace=False)
        env = ((t > ONSET - 3) & (t < ONSET + 5)).astype(float)
        for c in emg:
            x[c] += 50.0 * env * _band(rng, n, 30, 110)
    if pops is not None:
        for p in np.sort(rng.uniform(ONSET - 3, ONSET + 12, 7)):
            x[pops] += 600.0 * np.exp(-np.clip(t - p, 0, None) / 0.05) * (t >= p)
    if burst is not None:
        env = np.where((t >= ONSET - 2.5) & (t < ONSET - 1.5), np.sin(np.pi * (t - ONSET + 2.5)) ** 2, 0.0)
        x[burst] += 120.0 * env * _fast(rng, n)
    return x


def make_ieeg(s, sub, contacts, root, rng):
    d = os.path.join(root, "sourcedata", sub, "ses-clinical01", "ieeg")
    os.makedirs(d, exist_ok=True)
    # recorded names are zero-padded (LA01), coordinate names are not (LA1) - like the real dataset
    pad = list(contacts["edf_name"])
    soz_idx = np.flatnonzero(contacts["soz"].values == 1)
    # PID05: every seizure starts on a different electrode, and one other contact always joins 1 s later
    # (like the real PID20: a "SOZ" by the mean-EI rule that is never where a seizure starts)
    follower = [int(np.flatnonzero(contacts["electrode"].values == 6)[2])] if s == MULTIFOCAL else []
    role = contacts["role"].to_numpy() if "role" in contacts.columns else np.array([""] * len(contacts), dtype=object)
    non_soz = np.array([c for c in np.flatnonzero(contacts["soz"].values == 0) if c not in follower and not role[c]])
    bad_i = int(rng.choice(non_soz))
    n_seiz = 2 if s == TWO_SEIZURES else 7 if s == TWO_FOCI else 4 if s == SPORADIC else 3
    focus_a, focus_b = np.flatnonzero(role == "soz"), np.flatnonzero(role == "focus_b")
    spread, sporadic = np.flatnonzero(role == "spread"), np.flatnonzero(role == "sporadic")
    tasks = [(f"ictal{rng.integers(1000, 400000)}", f"0{4 + k % 2}") for k in range(n_seiz)]
    for k, (task, run) in enumerate(tasks):
        base = os.path.join(d, f"{sub}_ses-clinical01_task-{task}_run-{run}")
        if s == MULTIFOCAL:
            e = (k * 2 + 1) % 7
            starters = list(np.flatnonzero(contacts["electrode"].values == e)[:3])
        elif s == TWO_FOCI:
            starters = list(focus_a if k < 5 else focus_b)
        elif s == SPORADIC and k == 3:
            starters = list(sporadic)
        else:
            starters = list(soz_idx)
        slow = s == SLOW_SEIZURE and k == 2
        burst = int(rng.choice([c for c in non_soz if c != bad_i and c not in follower])) if k < 2 else None
        pop = int(np.flatnonzero(role == "pop")[0]) if (s == POPS and k == 1 and (role == "pop").any()) else None
        x = make_recording(contacts, rng, starters, slow=slow, burst=burst, followers=follower,
                           lead=4.0 if s == EARLY_ONSET else 0.0, spread=spread if s == EARLY_ONSET else (),
                           state_change=(s == STATE_CHANGE and k in (1, 3)), muscle=(s == MUSCLE and k == 0),
                           pops=pop)
        x[bad_i] = 0.0                                      # a dead channel, marked 'bad'
        names, types, data = list(pad), list(contacts["type"]), [x]
        if s == NO_PREFIX:   # scalp EEG channels recorded FIRST, with names colliding with P01-P08 / O01...
            scalp = 30.0 * _pink(rng, len(SCALP), x.shape[1])
            names, types, data = SCALP + names, ["EEG"] * len(SCALP) + types, [scalp] + data
        ekg = np.zeros(x.shape[1])
        ekg[(np.arange(0, DUR, 0.85) * FS).astype(int)] = 800.0
        ekg = np.convolve(ekg, np.hanning(9), mode="same")
        names, types = names + ["EKG1"], types + ["ECG"]
        data = np.vstack(data + [ekg[None]])
        write_edf(base + "_ieeg.edf", data, FS, names)
        soz_names = {pad[i] for i in soz_idx}
        ch = pd.DataFrame({
            "name": names, "type": types, "units": ["uV"] * len(names),
            "low_cutoff": ["0.0"] * len(names), "high_cutoff": [str(FS / 2)] * len(names),
            "description": ["n/a"] * len(names), "sampling_frequency": [str(float(FS))] * len(names),
            "status": ["bad" if nm == pad[bad_i] else "good" for nm in names],
            "status_description": [("soz" if (s == MARKED and nm in soz_names) else "n/a") for nm in names],
        })
        ch.to_csv(base + "_channels.tsv", sep="\t", index=False, encoding="utf-8-sig")
        if s == 2:
            ev = "onset\tduration\ttrial_type\n"                     # no onset given -> estimated from the data
        elif s == TWO_SEIZURES:
            ev = f"onset\tduration\ttrial_type\n{ONSET:.3f}\t20.000\tseizure onset\n"
        else:
            ev = f"onset\tduration\ttrial_type\tvalue\tsample\n{ONSET:.1f}\t{DUR - ONSET:.1f}\tictal\t1\t{int(ONSET * FS)}\n"
        with open(base + "_events.tsv", "w", encoding="utf-8-sig") as fh:
            fh.write(ev)
        with open(base + "_ieeg.json", "w") as fh:
            json.dump({"TaskName": task, "Manufacturer": "n/a", "PowerLineFrequency": "n/a",
                       "SamplingFrequency": FS, "SoftwareFilters": "n/a", "RecordingDuration": DUR,
                       "iEEGReference": "n/a"}, fh, indent=1)
    # an interictal recording (must be skipped)
    task = f"interictal{rng.integers(1000, 400000)}"
    x = make_recording(contacts, rng, [], slow=True)
    write_edf(os.path.join(d, f"{sub}_ses-clinical01_task-{task}_run-01_ieeg.edf"), x, FS, pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    labels = build_atlas()
    root = os.path.join(args.out, "files")
    os.makedirs(os.path.join(root, "sourcedata"), exist_ok=True)
    truth, meta = [], []
    for s in range(1, args.n + 1):
        sub = f"sub-PID{s:02d}"
        affine, norm, soz, present, side = make_mri(s, sub, labels, rng, root)
        make_raw_scans(s, sub, root)
        make_surfaces(s, sub, side, root)
        contacts = assign_roles(s, make_contacts(s, sub, labels, affine, soz, present, rng))
        has_ieeg = s <= args.n - 2
        has_recon = s <= args.n - 3
        if has_recon:
            make_recon(s, sub, contacts, affine, norm, root, rng)
        if has_ieeg:
            make_ieeg(s, sub, contacts, root, rng)
            truth.append(contacts.assign(multifocal=(s == MULTIFOCAL)))
        lat = {"L": "Left only", "R": "Right only"}[side]
        if s == SLOW_SEIZURE:
            lat = {"L": "Right only", "R": "Left only"}[side]          # a clinical table that disagrees
        if s == MULTIFOCAL:
            lat = "Bilateral independent Seizures"
        mts = s in FORCE_MTS
        meta.append({"Subject ID": sub, "Sex assigned at birth": "n/a",
                     "Seizure Onset Localization: ": "Multi-focal" if s == MULTIFOCAL else "Temporal",
                     "Seizure Onset Lateralization: ": lat,
                     "MRI Result:": "Potential Epileptogenic Lesion" if mts else "Normal",
                     "MRI Lesion Type:": "Mesial temporal sclerosis" if mts else "",
                     "MRI Lesion Laterality:": {"L": "Left", "R": "Right"}[side] if mts else ""})
        print(f"{sub}: SOZ regions = {[common.REGIONS[r] for r in sorted(soz)]}, "
              f"{int(contacts['soz'].sum())} SOZ contacts" + ("" if has_ieeg else "  (MRI only)")
              + ("" if has_recon or not has_ieeg else "  (no coordinates)"))
    pd.concat(truth).to_csv(os.path.join(args.out, "truth_contacts.csv"), index=False)
    pd.DataFrame(meta).to_excel(os.path.join(root, "metadata_3T_7T.xlsx"), index=False)
    with open(os.path.join(root, "sourcedata", "participants.json"), "w") as fh:
        json.dump({"participant_id": {"Description": "fake"}}, fh)
    with open(os.path.join(args.out, ".version"), "w") as fh:
        fh.write(VERSION)
    print(f"Synthetic dataset written to {args.out}/")


if __name__ == "__main__":
    main()
