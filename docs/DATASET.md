# Dataset

No patient data is stored in this repository. The scripts download what they need from the public dataset, the first
time they need it.

## Penn 3T-7T Paired Epilepsy Imaging

| | |
|---|---|
| Dataset | Lucas A. et al., *Penn 3T-7T Paired Epilepsy Imaging*, Pennsieve Discover (2026), [doi:10.26275/aosz-hj9u](https://doi.org/10.26275/aosz-hj9u) |
| Paper | Imtiaz T., Lucas A., et al., "A multimodal epilepsy dataset of paired 3-Tesla and 7-Tesla MRI and intracranial EEG", *Scientific Data* 13:1320 (2026), [doi:10.1038/s41597-026-07540-5](https://doi.org/10.1038/s41597-026-07540-5) |
| Patients | 30 adults with drug-resistant focal epilepsy (18 female, 38.8 ± 11.7 years); 15 later had iEEG |
| Access | public copy on AWS: `s3://edots-prod-aod-discover-publish50-use1/662/files/` (anonymous, free, no account) |
| Terms | please cite the dataset and the paper, and follow the terms of use given on its Pennsieve Discover page |

### What it contains (checked with `tools/check_dataset_extras.py`)

- **3T T1:** two protocols, 0.8 mm (23 patients) and 1 mm (7 patients). Version 8.1 and later cut every cube in
  millimetres, so both give the same 32 mm cubes.
- **FreeSurfer 3T and 7T** for all 30 patients (`derivatives/sub-PIDxx/freesurfer3T/`). FreeSurfer `T2.mgz` for 25.
  No T2 at all for PID01, 16, 17, 19 and 26. No hippocampal subfields. No thickness files.
- **FLAIR:** raw 3T FLAIR for 19 patients; no FreeSurfer FLAIR.
- **7T:** FreeSurfer 7T for 30 patients, raw 7T T2w for 28.
- **iEEG** (15 patients): ictal and interictal recordings (EDF, with channels and events tables) and electrode
  coordinates in the research 3T MRI space.
- **Clinical metadata** (`metadata_3T_7T.xlsx`):
  - seizure-onset lateralization and localization (decided by the multidisciplinary epilepsy surgery conference);
  - seizure types and frequencies;
  - the 3T and 7T MRI reports: result, lesion type, side, location.

  There is no per-patient surgical outcome column.

### What this project uses

| Used for | Files | Download |
|---|---|---|
| finding the patients and file formats | small tables | ≈ 30 MB |
| seizure-onset labels (`SOZ/02_compute_soz.py`) | ictal iEEG recordings of the 15 iEEG patients | ≈ 2.2 GB |
| mapping contacts to the MRI (`SOZ/03_build_contacts.py`) | electrode coordinates, FreeSurfer 3T, reference MRI | ≈ 0.5 GB |
| the CNN, maps and side model | FreeSurfer 3T `norm.mgz`, `aparc+aseg.mgz`, `T2.mgz` | included above, plus a few MB per extra patient |
| the clinical reference | `metadata_3T_7T.xlsx` | < 1 MB |

The files are kept in `CACHE_DIR` (`config.py`; default `~/ez_data`), so they are downloaded only once. If you already
have a copy of the dataset, set `DATA_SOURCE = "local"` and `LOCAL_DATASET_DIR` (the folder that contains `files`)
in `config.py`.

### Patient groups in this project

| Group | Patients |
|---|---|
| training (iEEG labels pass every check) | PID02, 04, 08, 12, 17, 18, 19, 21, 22, 25 |
| iEEG, left out of training | PID03, 16, 27 (no repeated onset), PID07 (one usable seizure), PID20 (doubtful coordinates) |
| MRI only (no iEEG) | PID01, 05, 06, 09, 10, 11, 13, 14, 15, 23, 24, 26, 28, 29, 30 |

The 20 patients outside the training group form the unseen test (`08_predict_mri.py --unseen`).

## Expert benchmark: multi-expert seizure annotation (in `benchmark/`)

This benchmark is used only to choose the onset detector. It has 40 spontaneous seizures from 19 adult patients
(Penn), whose onset channels were marked by 3–5 board-certified epileptologists.

- **Source:** NEMAR nm000402 (doi:10.82901/nemar.nm000402), a copy of Pennsieve Discover dataset 530
  ([doi:10.26275/dely-fzwx](https://doi.org/10.26275/dely-fzwx)). Ojemann W.K.S. et al. (2026), medRxiv,
  [doi:10.64898/2026.01.15.26344025](https://doi.org/10.64898/2026.01.15.26344025).
- **Licence:** CC BY 4.0. The small tables are in `benchmark/`, unchanged
  ([benchmark/README.txt](../benchmark/README.txt)). `tools/get_benchmark.py` downloads only the 63 s around each onset
  (≈ 0.8 GB).
