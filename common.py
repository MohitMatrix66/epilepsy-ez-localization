"""
common.py  -  small helpers shared by all scripts (brain-region names, file loading,
coordinate conversion, evaluation metrics). You normally do not need to edit this file.
"""
import os

import numpy as np
import nibabel as nib
from sklearn.metrics import roc_auc_score, average_precision_score

import config

# ---------------------------------------------------------------------------
# Brain regions we use (FreeSurfer "aparc+aseg" = Desikan-Killiany atlas + deep structures)
# ---------------------------------------------------------------------------
_DK = ["bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", None,  # 4 = corpus callosum, unused
       "cuneus", "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
       "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
       "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
       "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
       "postcentral", "posteriorcingulate", "precentral", "precuneus",
       "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
       "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
       "temporalpole", "transversetemporal", "insula"]

REGIONS = {}
for _k, _name in enumerate(_DK, start=1):
    if _name:
        REGIONS[1000 + _k] = "ctx-lh-" + _name
        REGIONS[2000 + _k] = "ctx-rh-" + _name

_SUBCORT_PAIRS = {10: 49, 11: 50, 12: 51, 13: 52, 17: 53, 18: 54, 26: 58}
REGIONS.update({10: "Left-Thalamus", 11: "Left-Caudate", 12: "Left-Putamen", 13: "Left-Pallidum",
                17: "Left-Hippocampus", 18: "Left-Amygdala", 26: "Left-Accumbens-area",
                49: "Right-Thalamus", 50: "Right-Caudate", 51: "Right-Putamen", 52: "Right-Pallidum",
                53: "Right-Hippocampus", 54: "Right-Amygdala", 58: "Right-Accumbens-area"})
REGION_IDS = np.array(sorted(REGIONS), dtype=np.int32)
# Version 9: deep grey matter (thalamus, caudate, putamen, pallidum, accumbens) - almost never touched by electrodes, so
# the model has no labels there. 04 adds 'not SOZ' cubes from them; 08 maps only the ASSESSED regions (cortex,
# hippocampus, amygdala) and shows the deep structures as 'not assessed'.
DEEP_IDS = np.array([10, 11, 12, 13, 26, 49, 50, 51, 52, 58], dtype=np.int32)
ASSESSED_IDS = np.array([r for r in REGION_IDS if r not in set(DEEP_IDS.tolist())], dtype=np.int32)


def homolog(rid):
    """Same region in the other hemisphere (used for left-right asymmetry features)."""
    if 1000 < rid < 1100:
        return rid + 1000
    if 2000 < rid < 2100:
        return rid - 1000
    for left, right in _SUBCORT_PAIRS.items():
        if rid == left:
            return right
        if rid == right:
            return left
    return None


def hemisphere(rid):
    if 1000 < rid < 1100 or rid in _SUBCORT_PAIRS:
        return "L"
    return "R"


LOBES = {
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                "paracentral", "frontalpole"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                 "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "cingulate": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate"],
    "insula": ["insula"],
}
_LOBE_OF = {name: lobe for lobe, names in LOBES.items() for name in names}


def lobe(rid):
    """Lobe of a region: frontal / temporal (incl. hippocampus, amygdala) / parietal / occipital /
    cingulate / insula / deep (thalamus, basal ganglia); '' for no region."""
    name = REGIONS.get(int(rid), "")
    if name.startswith("ctx-"):
        return _LOBE_OF.get(name[7:], "")
    if "Hippocampus" in name or "Amygdala" in name:
        return "temporal"
    return "deep" if name else ""


_LEFT_LABELS = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31]
_RIGHT_LABELS = [41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63]


def side_codes(labels):
    """Hemisphere of every voxel of a FreeSurfer aparc+aseg volume: 1 = left, 2 = right, 0 = midline/none."""
    lab = np.asarray(labels)
    out = np.zeros(lab.shape, np.int8)
    out[np.isin(lab, _LEFT_LABELS) | ((lab >= 1000) & (lab < 2000)) | ((lab >= 3000) & (lab < 4000))] = 1
    out[np.isin(lab, _RIGHT_LABELS) | ((lab >= 2000) & (lab < 3000)) | ((lab >= 4000) & (lab < 5000))] = 2
    return out


def nearest_code(codes, ijk, radius_vox):
    """Value of the nearest non-zero voxel of `codes` within radius_vox of the point ijk (0 if none)."""
    centre = np.round(ijk).astype(int)
    r = int(np.ceil(radius_vox))
    lo = np.maximum(centre - r, 0)
    hi = np.minimum(centre + r + 1, codes.shape)
    if np.any(lo >= hi):
        return 0
    cube = codes[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    nz = np.argwhere(cube > 0)
    if not len(nz):
        return 0
    d = np.linalg.norm(nz + lo - np.asarray(ijk), axis=1)
    k = int(np.argmin(d))
    return int(cube[tuple(nz[k])]) if d[k] <= radius_vox else 0


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def subject_dir(sub):
    return os.path.normpath(config.FS_DIR_TEMPLATE.format(root=config.DATA_ROOT, sub=sub, field=config.FIELD))


def ensure_local(path):
    """With DATA_SOURCE = 'aws', download a dataset file the first time it is needed.
    Returns the local path (which may not exist if the dataset does not have that file)."""
    if config.DATA_SOURCE == "aws" and not os.path.exists(path):
        import aws_data
        rel = os.path.relpath(path, config.DATA_ROOT).replace(os.sep, "/")
        if not rel.startswith(".."):
            aws_data.fetch(rel, missing_ok=True)
    return path


# Other spellings of FreeSurfer files (the Penn dataset writes 'aparc_aseg.mgz' for 'aparc+aseg.mgz')
FS_ALTERNATIVES = {"aparc+aseg.mgz": ["aparc_aseg.mgz"]}


def mri_path(sub, fname):
    """Local path of a FreeSurfer volume (downloaded the first time in 'aws' mode). Tries the other
    spellings of the file name too. Returns a path that may not exist if the dataset lacks the file."""
    paths = [os.path.join(subject_dir(sub), "mri", f) for f in [fname] + FS_ALTERNATIVES.get(fname, [])]
    for p in paths:
        if os.path.exists(p):
            return p
    for p in paths:
        if os.path.exists(ensure_local(p)):
            return p
    return paths[0]


def has_mri(sub, fname):
    return os.path.exists(mri_path(sub, fname))


def load_mgz(sub, fname):
    path = mri_path(sub, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}\n  -> check DATA_SOURCE / FIELD in config.py and the subject name '{sub}'.")
    return nib.load(path)


def read_aparc_thickness(sub):
    """Average cortical thickness per cortical region from FreeSurfer's ?h.aparc.stats.
    Returns {region_id: thickness_mm}; empty dict if the stats files are missing."""
    out = {}
    for hemi, offset in (("lh", 1000), ("rh", 2000)):
        path = ensure_local(os.path.join(subject_dir(sub), "stats", f"{hemi}.aparc.stats"))
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if parts[0] in _DK:
                    out[offset + _DK.index(parts[0]) + 1] = float(parts[4])  # column 5 = ThickAvg
    return out


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def ras_to_vox(xyz_mm, img):
    """Convert millimetre coordinates (N x 3) to voxel indices (N x 3, float) of `img`."""
    if config.COORD_SPACE == "scanner":
        vox2ras = img.affine
    elif config.COORD_SPACE == "tkr":
        vox2ras = img.header.get_vox2ras_tkr()
    else:
        raise ValueError("COORD_SPACE must be 'scanner' or 'tkr'")
    xyz = np.asarray(xyz_mm, dtype=float).reshape(-1, 3)
    return nib.affines.apply_affine(np.linalg.inv(vox2ras), xyz)


def voxel_size(img):
    """Voxel size in mm of an MRI (FreeSurfer volumes are isotropic: 1 mm, or 0.8 mm for the patients of this dataset
    scanned with the 0.8 mm T1 protocol). Raises an error for clearly anisotropic voxels: the mirror geometry and the
    cubes need equal voxel sizes along the three axes."""
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    if zooms.max() / zooms.min() > 1.02:
        raise ValueError(f"the MRI has voxels of {zooms.round(3).tolist()} mm - not isotropic; use FreeSurfer's "
                         "conformed volumes (mri/norm.mgz, mri/aparc+aseg.mgz)")
    return float(zooms.mean())


def near_brain_mask(labels, voxel_mm, margin_mm=5.0):
    """The brain (labels > 0) grown by margin_mm millimetres (along the image axes), the same physical margin for
    every voxel size (5 mm = 5 voxels at 1 mm, 6 voxels at 0.8 mm)."""
    from scipy import ndimage
    steps = max(1, int(round(margin_mm / float(voxel_mm))))
    return ndimage.binary_dilation(np.asarray(labels) > 0, iterations=steps)


def assign_region(labels, ijk, radius_vox):
    """Atlas region of one contact. If the contact sits in white matter, take the nearest
    grey-matter region within `radius_vox`. Returns (region_id, distance_in_voxels);
    region_id is 0 when nothing is close enough or the point is outside the image."""
    centre = np.round(ijk).astype(int)
    if np.any(centre < 0) or np.any(centre >= labels.shape):
        return 0, np.nan
    lab = int(labels[tuple(centre)])
    if lab in REGIONS:
        return lab, 0.0
    r = int(np.ceil(radius_vox))
    lo = np.maximum(centre - r, 0)
    hi = np.minimum(centre + r + 1, labels.shape)
    cube = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    ii, jj, kk = np.meshgrid(*[np.arange(lo[d], hi[d]) for d in range(3)], indexing="ij")
    dist = np.sqrt((ii - ijk[0]) ** 2 + (jj - ijk[1]) ** 2 + (kk - ijk[2]) ** 2)
    ok = np.isin(cube, REGION_IDS) & (dist <= radius_vox)
    if not ok.any():
        return 0, np.nan
    best = np.argmin(np.where(ok, dist, np.inf))
    return int(cube.ravel()[best]), float(dist.ravel()[best])


# ---------------------------------------------------------------------------
# Left-right mirror (version 8): every place in the brain is compared with the same place on the other side
# ---------------------------------------------------------------------------
# left / right label pairs of aparc+aseg used to find the plane between the hemispheres
_MIRROR_PAIRS = ([(2, 41), (4, 43), (5, 44), (7, 46), (8, 47), (10, 49), (11, 50), (12, 51), (13, 52), (17, 53),
                  (18, 54), (26, 58), (28, 60)]
                 + [(1000 + k, 2000 + k) for k, name in enumerate(_DK, start=1) if name])


def label_centroids(labels):
    """Centre (voxel coordinates) and voxel count of every label value, without building a coordinate grid
    (16 million voxels - a grid would need ~400 MB). Returns (centres: n_labels x 3, counts: n_labels)."""
    lab = np.asarray(labels).astype(np.int64)
    n = int(lab.max()) + 1
    counts = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    centres = np.zeros((n, 3))
    for axis in range(3):
        moved = np.moveaxis(lab, axis, 0)
        for pos in range(moved.shape[0]):
            centres[:, axis] += pos * np.bincount(moved[pos].ravel(), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        centres /= counts[:, None]
    return centres, counts


def midsagittal_plane(labels, min_vox=50, voxel_mm=1.0):
    """The plane between the two hemispheres, from the centres of left / right structure pairs (thalamus, putamen,
    hippocampus, the cortical regions, ...): its normal is the average left-minus-right direction and it passes
    through the middle of the pairs. Works when the head is tilted. Voxel coordinates (isotropic voxels of voxel_mm).
    Returns dict normal (unit vector towards the LEFT hemisphere), offset b (plane: normal . x = b, in voxels),
    pairs (number used), tilt_deg (angle to the volume's first axis), spread_mm (how well the pairs agree, mm)."""
    centres, counts = label_centroids(labels)
    n_lab = len(counts)
    dirs, mids = [], []
    for left, right in _MIRROR_PAIRS:
        if left < n_lab and right < n_lab and counts[left] >= min_vox and counts[right] >= min_vox:
            d = centres[left] - centres[right]
            if np.linalg.norm(d) > 1.0:
                dirs.append(d / np.linalg.norm(d))
                mids.append((centres[left] + centres[right]) / 2.0)
    if len(dirs) < 3:   # cannot measure it: use the middle of the brain along the first axis
        brain = np.asarray(labels) > 0
        normal = np.array([1.0, 0.0, 0.0])
        b = float(np.argwhere(brain.any(axis=(1, 2)))[:, 0].mean()) if brain.any() else labels.shape[0] / 2.0
        return {"normal": normal, "offset": b, "pairs": len(dirs), "tilt_deg": 0.0, "spread_mm": np.nan}
    normal = np.mean(dirs, axis=0)
    normal /= np.linalg.norm(normal)
    proj = np.asarray(mids) @ normal
    b = float(np.median(proj))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[0])))))
    return {"normal": normal, "offset": b, "pairs": len(dirs), "tilt_deg": round(tilt, 2),
            "spread_mm": round(float(np.median(np.abs(proj - b))) * float(voxel_mm), 2)}


def reflect(points, plane):
    """Mirror points through the mid-sagittal plane. points: one point (3,) or N points (N x 3), voxel coordinates."""
    p = np.asarray(points, dtype=np.float64)
    n, b = np.asarray(plane["normal"], dtype=np.float64), float(plane["offset"])
    d = p @ n - b
    return p - 2.0 * np.multiply.outer(d, n) if p.ndim > 1 else p - 2.0 * d * n


def side_of_plane(points, plane):
    """Signed distance to the mid-sagittal plane in VOXELS (times the voxel size = mm): > 0 = left, < 0 = right."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return p @ plane["normal"] - plane["offset"]


def standardise(vol, mask):
    """Intensities to mean 0, standard deviation 1 inside the brain mask, applied to the WHOLE image.
    Version 9: nothing is set to 0 outside the brain any more. Before, the fluid between the gyri (outside FreeSurfer's
    labels) became 0 = 'average tissue' while the fluid in the ventricles stayed dark, so only places next to a
    ventricle had dark fluid in their cube - a shortcut the model used (it pointed to the caudate). Now fluid is dark
    everywhere, and the background of the skull-stripped norm.mgz is dark too."""
    vol = np.asarray(vol, dtype=np.float32)
    m, s = float(vol[mask].mean()), float(vol[mask].std()) + 1e-6
    return (vol - m) / s


def pair_patches(vol, centres, plane, size, spacing=1.0, batch=32):
    """For every centre (voxel coordinates, may be fractional): the cube of `size`^3 samples around it (channel 0)
    and the MIRROR cube from the same place on the other side of the brain (channel 1), reflected through the
    mid-sagittal plane so that both cubes show the same anatomy in the same orientation. Trilinear sampling, 0 outside
    the image. vol: one standardised image. spacing: distance between samples in VOXELS - pass
    config.PATCH_MM / voxel size, so that every cube covers the same millimetres whatever the MRI's voxel size
    (version 8.1: 32 samples x 1 mm = 32 mm on a 1 mm MRI and on a 0.8 mm MRI alike).
    Returns float16 array (N, 2, size, size, size)."""
    from scipy import ndimage
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
    offs = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * float(spacing)   # -15.5 ... 15.5 mm for 32
    grid = np.stack(np.meshgrid(offs, offs, offs, indexing="ij")).reshape(3, -1)
    out = np.zeros((len(centres), 2, size, size, size), dtype=np.float16)
    vol = np.asarray(vol, dtype=np.float32)
    background = float(vol.min())          # outside the image = background (version 9: not 'average tissue')
    n, b = np.asarray(plane["normal"], dtype=np.float64), float(plane["offset"])
    for s in range(0, len(centres), batch):
        c = centres[s:s + batch]
        pts = (c.T[:, :, None] + grid[:, None, :]).reshape(3, -1)          # 3 x (B * size^3)
        mir = pts - 2.0 * n[:, None] * (n @ pts - b)[None, :]               # the same points, mirrored
        for ch, coords in ((0, pts), (1, mir)):
            vals = ndimage.map_coordinates(vol, coords, order=1, mode="constant", cval=background)
            out[s:s + len(c), ch] = vals.reshape(len(c), size, size, size)
    return out


def clinical_record():
    """{subject: (seizure-onset side 'L' / 'R' / '', the side as written, the lobe as written)} from the dataset's
    clinical table (metadata_3T_7T.xlsx). 'Left only' and 'Left > Right' count as L. Empty dict if not available."""
    import aws_data
    df = aws_data.metadata_table()
    if df is None:
        return {}
    cols = {str(c).strip().lower().rstrip(":").strip(): c for c in df.columns}
    sub_col = next((c for k, c in cols.items() if "subject" in k or "participant" in k), None)
    lat_col = next((c for k, c in cols.items() if "onset lateral" in k), None)
    loc_col = next((c for k, c in cols.items() if "onset local" in k), None)
    if sub_col is None or lat_col is None:
        return {}
    out = {}
    for _, r in df.iterrows():
        sid = str(r[sub_col]).strip()
        if not sid or sid.lower() == "nan":
            continue
        sid = sid if sid.startswith("sub-") else "sub-" + sid
        lat = str(r[lat_col]).strip()
        low = lat.lower()
        side = "L" if low.startswith(("left only", "left >")) else "R" if low.startswith(("right only", "right >")) else ""
        out[sid] = (side, lat, str(r[loc_col]).strip() if loc_col is not None else "")
    return out


def mri_record():
    """{subject: (MRI result, lesion type, lesion side 'L' / 'R' / 'B' / '')} from the 3T MRI columns of
    metadata_3T_7T.xlsx (radiology report). Empty dict if not available."""
    import aws_data
    df = aws_data.metadata_table()
    if df is None:
        return {}
    cols = {str(c).strip().lower().rstrip(":").strip(): c for c in df.columns}
    sub_col = next((c for k, c in cols.items() if "subject" in k or "participant" in k), None)
    res_col, typ_col, lat_col = cols.get("mri result"), cols.get("mri lesion type"), cols.get("mri lesion laterality")
    if sub_col is None:
        return {}
    out = {}
    for _, r in df.iterrows():
        sid = str(r[sub_col]).strip()
        if not sid or sid.lower() == "nan":
            continue
        sid = sid if sid.startswith("sub-") else "sub-" + sid
        val = [str(r[c]).strip() if c is not None and str(r[c]).strip().lower() != "nan" else ""
               for c in (res_col, typ_col, lat_col)]
        out[sid] = (val[0], val[1], {"left": "L", "right": "R", "bilateral": "B"}.get(val[2].lower(), ""))
    return out


def is_mts(lesion_type):
    """Does the radiology report name mesial temporal (hippocampal) sclerosis?"""
    return "mesial temporal sclerosis" in str(lesion_type).lower()


# Version 9 side model: five left-right measures chosen IN ADVANCE after the published hippocampal-sclerosis models
# (AID-HS, Ann Neurol 2025: volume and shape asymmetry; Caldairou et al., Neurology 2021: T2 brightness) and the
# ENIGMA-Epilepsy finding of ipsilateral thalamus and temporal-cortex changes.
SIDE_FEATURES = ["asym_volume_Hippocampus", "asym_volume_Amygdala", "asym_volume_Thalamus", "asym_t2_Hippocampus",
                 "asym_thickness_temporal"]


def _asym(a, b):
    """(left - right) / mean; NaN when a value is missing or not positive."""
    return (a - b) / ((abs(a) + abs(b)) / 2.0) if (a > 0 and b > 0) else np.nan


def side_features(labels, t2=None, surf_dir=None, label_dir=None):
    """The SIDE_FEATURES of one brain. labels: aparc+aseg; t2: FreeSurfer T2.mgz (same grid) or None; surf_dir /
    label_dir: FreeSurfer surf/ and label/ folders (for ?h.thickness and ?h.aparc.annot) or None. NaN = not available
    (the side model then uses 'no difference' for it)."""
    lab = np.asarray(labels)
    cnt = np.bincount(lab.ravel().astype(np.int64), minlength=64).astype(np.float64)
    f = {"asym_volume_Hippocampus": _asym(cnt[17], cnt[53]), "asym_volume_Amygdala": _asym(cnt[18], cnt[54]),
         "asym_volume_Thalamus": _asym(cnt[10], cnt[49]), "asym_t2_Hippocampus": np.nan,
         "asym_thickness_temporal": np.nan}
    if t2 is not None:
        t2 = np.asarray(t2, dtype=np.float64)

        def relative(hip, wm):        # hippocampus T2 brightness / white-matter T2 brightness of the same side
            h, w = lab == hip, lab == wm
            return float(t2[h].mean() / t2[w].mean()) if h.any() and w.any() and t2[w].mean() > 0 else np.nan
        f["asym_t2_Hippocampus"] = _asym(relative(17, 2), relative(53, 41))
    if surf_dir and label_dir:
        thick = {}
        for hemi in ("lh", "rh"):
            tp, ap = os.path.join(surf_dir, f"{hemi}.thickness"), os.path.join(label_dir, f"{hemi}.aparc.annot")
            if not (os.path.exists(tp) and os.path.exists(ap)):
                break
            t = nib.freesurfer.read_morph_data(tp)
            vlab, _, names = nib.freesurfer.read_annot(ap)
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
            temporal = [i for i, n in enumerate(names) if n in LOBES["temporal"]]
            sel = np.isin(vlab, temporal) & (t > 0)
            if len(t) == len(vlab) and sel.any():
                thick[hemi] = float(t[sel].mean())
        if len(thick) == 2:
            f["asym_thickness_temporal"] = _asym(thick["lh"], thick["rh"])
    return f


def side_inputs(sub):
    """(labels, T2 or None, surf dir or None, label dir or None) of a dataset patient for side_features (the
    surface files are downloaded the first time in 'aws' mode; missing ones are simply not used)."""
    labels = np.asarray(load_mgz(sub, "aparc+aseg.mgz").dataobj).astype(np.int32)
    t2 = np.asarray(load_mgz(sub, "T2.mgz").dataobj) if has_mri(sub, "T2.mgz") else None
    base = subject_dir(sub)
    for rel in ("surf/lh.thickness", "surf/rh.thickness", "label/lh.aparc.annot", "label/rh.aparc.annot"):
        ensure_local(os.path.join(base, *rel.split("/")))
    return labels, t2, os.path.join(base, "surf"), os.path.join(base, "label")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def safe_auroc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def safe_auprc(y, p):
    y = np.asarray(y)
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan


def topk_hit_rate(df, prob_col, k, label_col="target", group_col="subject"):
    """Fraction of patients for whom at least one true positive region is among the
    k highest-scoring regions. Only patients with at least one positive count."""
    hits = []
    for _, g in df.groupby(group_col):
        if g[label_col].sum() == 0:
            continue
        top = g.sort_values(prob_col, ascending=False).head(k)
        hits.append(bool(top[label_col].max() == 1))
    return float(np.mean(hits)) if hits else np.nan


def summarize(df, prob_col, label_col="target"):
    d = df.dropna(subset=[prob_col])
    return {
        "n": len(d),
        "prevalence": round(float(d[label_col].mean()), 3),
        "AUROC": round(safe_auroc(d[label_col], d[prob_col]), 3),
        "AUPRC": round(safe_auprc(d[label_col], d[prob_col]), 3),
        "top1_hit": round(topk_hit_rate(d, prob_col, 1, label_col), 3),
        "top3_hit": round(topk_hit_rate(d, prob_col, 3, label_col), 3),
    }
