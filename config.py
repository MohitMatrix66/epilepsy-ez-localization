"""
config.py  -  all project settings live here. Normally this is the only file you edit.

Order of work (see README.md):
    tools/check_install.py -> tools/smoke_test.py                                           (installation)
    tools/get_benchmark.py -> tools/benchmark_onset.py                                       (choose the method)
    SOZ/01_scan_dataset.py -> SOZ/02_compute_soz.py -> tools/check_unlocalized.py --all
    -> SOZ/03_build_contacts.py -> tools/plot_seizure.py --audit                            (labels)
    01_label_contacts.py -> 02_region_features.py -> 03_train_baseline.py
    -> 04_make_patches.py -> 05_train_cnn.py (Google Colab GPU) -> tools/model_report.py (model)
    -> 08_predict_mri.py --unseen / --subject / --fs-dir                                 (final model on MRIs)
    06_make_heatmap.py, 07_lateralization.py                                             (extra)
    tools/check_dataset_extras.py         (what else the dataset holds: T2 / FLAIR / 7T scans, outcome tables)

The data is read from the public Penn 3T-7T dataset on AWS. Only the files the project needs are
downloaded, into CACHE_DIR, the first time a script needs them; later runs reuse them.

Any setting can also be overridden with an environment variable named EZ_<SETTING>
(for example EZ_CACHE_DIR). The smoke test uses this to point at fake patients.
"""
import os

EZ_VERSION = "9"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================ EDIT ME ============================
# Where the data comes from:
#   "aws"   = the public dataset on AWS (no account, free). Files are downloaded on demand.
#   "local" = a copy you already have on disk (set LOCAL_DATASET_DIR to the folder that contains "files").
DATA_SOURCE = "aws"
LOCAL_DATASET_DIR = ""

# Downloaded files are kept here (about 4 GB for this project, with the expert benchmark).
# C:\Users\<you>\ez_data on Windows.
CACHE_DIR = os.path.join(os.path.expanduser("~"), "ez_data")

# Results of the main pipeline (and outputs/logs/: what every script printed).
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# Results of the SOZ scripts (dataset inventory, benchmark choice, SOZ labels, contacts.csv).
SOZ_RESULTS_DIR = os.path.join(PROJECT_DIR, "SOZ", "results")

# The contact table the model is trained on (made by SOZ/03_build_contacts.py).
# Columns: subject, contact, x, y, z, soz, uncertain   (x, y, z = scanner RAS in mm of the research 3T MRI).
# With the bipolar montage a "contact" is a pair of neighbouring contacts, e.g. LA01-LA02, placed at
# the midpoint between them.
CONTACTS_CSV = os.path.join(SOZ_RESULTS_DIR, "contacts.csv")

# The expert benchmark (seizures marked by epileptologists) shipped with the project: benchmark/README.txt
BENCHMARK_DIR = os.path.join(PROJECT_DIR, "benchmark")

# Which FreeSurfer run to use: "3T" or "7T". The electrode coordinates in this dataset are in the
# research 3T MRI, so keep "3T" (7T needs an extra registration step first).
FIELD = "3T"

# Coordinate system of x, y, z in CONTACTS_CSV: "scanner" (what SOZ/03 writes) or "tkr".
COORD_SPACE = "scanner"

# What counts as a positive contact: "soz" (only option for this dataset; "soz_and_resected"
# needs a 'resected' column, which this dataset does not provide).
LABEL_MODE = "soz"

# ---------------- SOZ labels from the seizure recordings (SOZ/ folder) ----------------
# "auto"     = use clinician SOZ marks if the dataset's channel tables contain them, otherwise compute them
# "computed" = always compute them from the seizure recordings (onset detector, version 6.1)
# "clinical" = only clinician marks (stops if there are none)
SOZ_LABEL_SOURCE = "auto"
# How the start of a seizure is found:
#   "benchmark"    = the method that agreed best with the epileptologists in tools/benchmark_onset.py
#                    (SOZ/results/benchmark_choice.json) - the normal choice; step 2 stops if there is no choice yet.
#                    On the 40 expert seizures this is "ei+linelength" (version 6.1).
#   "ei+linelength" / "ei" / "linelength" / "multifeature" = that method with its default settings (for tests)
ONSET_METHOD = "benchmark"
# Line length gives no answer for a seizure when more than this share of the sites start together ("diffuse onset")
DIFFUSE_SHARE = 0.4
# From the onsets of every seizure to a patient's SOZ (rule "repeated"):
#   SOZ site       = an onset site in at least SOZ_MIN_ONSETS usable seizures AND in at least SOZ_MIN_SHARE of them
#                    (where the seizures usually start; several foci in one patient all count)
#   uncertain site = an onset site less often than that, or a site already active before the onset:
#                    used neither as SOZ nor as "not SOZ"
SOZ_MIN_ONSETS = 2
SOZ_MIN_SHARE = 0.25      # e.g. 20 usable seizures: onset site in at least 5 of them
SOZ_MIN_LED = 0.3         # a patient's SOZ is trusted only if the SOZ (or a site next to it, same electrode) is
                          # among the onset sites of at least this share of the usable seizures
MINORITY_SIDE_SHARE = 0.25  # step 3: SOZ sites on the side that holds less than this share of a patient's SOZ sites
                            # become uncertain (on the expert benchmark every patient's onset was on one side)
TRAIN_ONLY_CONSISTENT = True  # leave patients without a trustworthy SOZ out of training (contacts.csv)
EI_AS_SECOND_OPINION = True   # (only when the chosen method does not use the EI) sites the version 5 EI makes SOZ
                              # but the chosen method does not -> uncertain
# Version 5 EI settings (second opinion and comparison only)
EI_MONTAGE = "bipolar"    # montage of the EI comparison when the benchmark chose a non-EI method
EI_THRESHOLD = 0.3
SOZ_STRONG_EI = 0.7
SOZ_MEAN_EI = 0.3
EI_PRE_S = 30.0           # seconds of recording before seizure onset used as the normal period
EI_POST_S = 20.0          # seconds after onset searched
EI_LEAD_S = 10.0          # ... and seconds before it (version 5)
KEEP_EDF = True           # True keeps the downloaded seizure recordings (~2.2 GB in CACHE_DIR) so they never have
                          # to be downloaded again; False deletes each one after use
# Electrode-name fixes when a patient's recordings and coordinate files name an electrode differently:
# {"sub-PID02": {"RG": "G"}} means the recorded channel 'RG12' is 'G12' in the coordinate files.
# Found with SOZ/results/name_matching.csv in this dataset (same number of contacts on both sides):
#   PID02: the 64-contact grid is RG01-RG64 in the recordings, G1-G64 in the coordinates;
#          DH = DH (hippocampus); DHAA = DHA and DHT = DHP (the names fit, and their contacts start seizures
#          together). Note: DHA and DHP lie side by side about 3 cm BEHIND DA / DH (deep posterior temporal, next to
#          the ventricle) - they are not an anterior (amygdala / hippocampal head) probe. DHI has no coordinates
#          (it was not localised) and never starts a seizure (tools/check_unlocalized.py).
#   PID08: the amygdala / hippocampus depth electrodes are LA, LH, RA, RH in the recordings and
#          LDA, LDH, RDA, RDH (4 contacts each) in the coordinates
CHANNEL_ALIASES = {
    "sub-PID02": {"RG": "G", "DHAA": "DHA", "DHT": "DHP"},
    "sub-PID08": {"LA": "LDA", "LH": "LDH", "RA": "RDA", "RH": "RDH"},
}
# Patients left out of training, with the reason (printed by SOZ/03_build_contacts.py). Their MRI is still used
# by 07_lateralization.py and by 08_predict_mri.py --unseen (as a patient the model never saw). Every patient is
# judged by the same automatic checks (usable seizures, repeated onsets, SOZ leads, coordinates); add a line only
# after looking at the patient's results.
EXCLUDE_FROM_TRAINING = {
    "sub-PID20": "coordinates DOUBTFUL (tools/check_unlocalized.py), SOZ lobe not the clinical lobe, "
                 "not confirmed by line length on the common average (tools/v7_car_linelength.py)",
}
# =================================================================

S3_BUCKET = "edots-prod-aod-discover-publish50-use1"
S3_PREFIX = "662/files/"   # the dataset's "files" folder on AWS
S3_ENDPOINT = ""           # leave empty (only used by the tests)

FS_DIR_TEMPLATE = "{root}/derivatives/{sub}/freesurfer{field}/{sub}"
REGION_RADIUS_MM = 3.0     # a contact in white matter is assigned to the nearest grey-matter region within this radius
PATCH_SIZE = 32            # CNN input cube: PATCH_SIZE samples along each axis ...
PATCH_MM = 1.0             # ... PATCH_MM apart, so every cube is 32 mm wide whatever the MRI's voxel size.
                           # (This dataset has two 3T T1 protocols: 7 of the 10 training patients have 0.8 mm
                           # FreeSurfer volumes, 3 have 1 mm - before version 8.1 their cubes were 25.6 mm.)
USE_T2 = True              # add the T2 image as a second CNN channel / extra features, only when EVERY patient
                           # has one (in this dataset 5 of 30 patients have no FreeSurfer T2.mgz, 2 of them
                           # training patients - PID17, PID19 - so the model uses T1 only)
SEED = 42

_OVERRIDABLE = ["DATA_SOURCE", "LOCAL_DATASET_DIR", "CACHE_DIR", "OUT_DIR", "SOZ_RESULTS_DIR", "CONTACTS_CSV",
                "BENCHMARK_DIR", "FIELD", "COORD_SPACE", "LABEL_MODE", "SOZ_LABEL_SOURCE", "ONSET_METHOD",
                "DIFFUSE_SHARE", "SOZ_MIN_ONSETS", "SOZ_MIN_SHARE", "SOZ_MIN_LED", "MINORITY_SIDE_SHARE",
                "TRAIN_ONLY_CONSISTENT", "EI_AS_SECOND_OPINION",
                "EI_MONTAGE", "EI_THRESHOLD", "SOZ_STRONG_EI", "SOZ_MEAN_EI", "EI_PRE_S", "EI_POST_S", "EI_LEAD_S",
                "EXCLUDE_FROM_TRAINING", "KEEP_EDF", "S3_BUCKET", "S3_PREFIX", "S3_ENDPOINT",
                "REGION_RADIUS_MM", "PATCH_SIZE", "PATCH_MM", "USE_T2", "SEED", "DATA_ROOT"]


def _apply_env_overrides():
    g = globals()
    for name in _OVERRIDABLE:
        val = os.environ.get("EZ_" + name)
        if val is None:
            continue
        current = g.get(name)
        if isinstance(current, bool):
            g[name] = val.strip().lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            g[name] = int(val)
        elif isinstance(current, float):
            g[name] = float(val)
        elif isinstance(current, dict):   # e.g. EZ_EXCLUDE_FROM_TRAINING="sub-PID06,sub-PID07" ("" = none)
            g[name] = {s.strip(): f"listed in EZ_{name}" for s in val.split(",") if s.strip()}
        else:
            g[name] = val
    # CONTACTS_CSV follows SOZ_RESULTS_DIR unless it was set on its own
    if "EZ_SOZ_RESULTS_DIR" in os.environ and "EZ_CONTACTS_CSV" not in os.environ:
        g["CONTACTS_CSV"] = os.path.join(g["SOZ_RESULTS_DIR"], "contacts.csv")


DATA_ROOT = None
_apply_env_overrides()

# DATA_ROOT = the local folder that mirrors the dataset's "files" folder
if DATA_ROOT is None:
    if DATA_SOURCE == "aws":
        DATA_ROOT = os.path.join(CACHE_DIR, "files")
    elif DATA_SOURCE == "local":
        if not LOCAL_DATASET_DIR:
            raise SystemExit("config.py: DATA_SOURCE = 'local' needs LOCAL_DATASET_DIR (the folder that contains 'files')")
        DATA_ROOT = os.path.join(LOCAL_DATASET_DIR, "files")
    else:
        raise SystemExit("config.py: DATA_SOURCE must be 'aws' or 'local'")

os.makedirs(OUT_DIR, exist_ok=True)
