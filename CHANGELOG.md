# Changelog

Each version was tested on the real data, and the next one fixed what the test showed. The reasons are kept here,
because they are part of the method.

## Version 9 (2026-10-22): the shortcut fixed, and a side model like the published ones

The version 8.1 model pointed to the **caudate** in 19 of 20 new patients. It had learned "ventricle fluid in the cube
= seizure onset", because most onset sites are in the hippocampus, next to a ventricle.

- **Fluid is dark everywhere** (`common.standardise`). Nothing is set to 0 outside FreeSurfer's labels any more, so
  dark fluid no longer marks only the places next to a ventricle.
- **Deep-structure examples** (`04_make_patches.py`). Every training patient gets 2 "not SOZ" cubes per deep structure
  and side (thalamus, caudate, putamen, pallidum, accumbens), at least 10 mm from any SOZ or uncertain contact.
- **Flip averaging** (`05`, `08`): every prediction is the average over the cube and its left-right flipped copy.
- **The map covers what the model was trained on** (`08_predict_mri.py`): the cortex, hippocampus and amygdala. The
  deep structures are outlined in grey as NOT ASSESSED.
- **Side model** (`07_lateralization.py`), after AID-HS and Caldairou et al. (*Neurology* 2021). It is a small
  logistic regression on 5 left-right measures fixed in advance: hippocampus, amygdala and thalamus volume,
  hippocampal T2 signal, and temporal-cortex thickness. It says "not sure" between p = 0.4 and 0.6. It is tested
  leave-one-patient-out on the 25 patients with a clinical side, and shown separately for hippocampal sclerosis.
- **Fairer tests**:
  - `08 --unseen` adds the fair side score (an AUROC, not affected by a lean to one side), the side model, and the
    sclerosis check;
  - `tools/model_report.py` adds the deep-structure check (section 5) and the shortcut check (section 6).
- **Scope in every summary:** drug-resistant focal epilepsy; not for generalized epilepsy; deep lesions not assessed;
  weakest when the MRI looks normal.
- **New:** `new_patient_results/run_new_patients.py` runs the final model on the unseen patients and compares it with
  the clinical record, without changing any project file.

These changes were decided after looking at the v8.1 results on the unseen patients. Say so when you report the
version 9 numbers.

## Version 8.1 (2026-10-21): cubes in millimetres

The dataset paper says the 3T T1 images were made with **two protocols: 0.8 mm and 1 mm voxels**. Up to version 8,
every cube was 32 voxels wide: 32 mm in 3 training patients but 25.6 mm in 7, so the same hippocampus looked 25% bigger
in some patients.

- **Cubes in millimetres** (`04_make_patches.py`, `08_predict_mri.py`): every cube is 32 samples 1 mm apart, which is
  32 mm on any MRI (`PATCH_MM` in `config.py`). The smoke test checks it: the same brain at 1 mm and at 0.8 mm gives the
  same cube within 1–2% (the old voxel cubes differed by about 65%). The 5 mm margin around the brain and the `--step`
  of `08_predict_mri.py` are now in mm on any MRI.
- `05_train_cnn.py` refuses older patches, and `08_predict_mri.py` refuses an older model.
- **New:** `tools/check_dataset_extras.py` lists what else the dataset holds: raw T2, FLAIR and 7T scans, voxel sizes,
  hippocampal subfields, brain-shift files, and the columns of every top-level table.
- The source of the clinical side is now named correctly: the decision of the multidisciplinary epilepsy surgery
  conference.

## Version 8: compare left with right, like a doctor

The first real training run showed that the CNN had learned **anatomy** ("the hippocampus is often SOZ"). It ranked SOZ
sites above chance (AUROC 0.73 within patients) but not clearly better than an anatomy-only guess. In patients with
electrodes on both sides, it even scored the healthy hippocampus higher than the epileptic one.

- **Mirror cubes:** every MRI cube comes with the cube from the same place on the other side of the brain, mirrored
  through the plane between the hemispheres. The plane is found from left/right structure pairs, so it works when the
  head is tilted.
- **Mirror controls:** for every SOZ site, the same place on the healthy side is a "not SOZ" example. It is used only
  when the clinical record says one side only and agrees with the SOZ side.
- **A fair test** (`tools/model_report.py`):
  - an anatomy-only baseline for every site;
  - regions scored with the MEAN of their sites (the maximum rewards regions with many electrodes);
  - a left-right test for every patient.
- **Steadier training:** 3 models per fold, averaged; early stopping on a 3-epoch average; `--resume` reuses only
  folds with exactly the same data and settings.
- **Final model:** `05_train_cnn.py --final`, then `08_predict_mri.py`, makes a whole-brain probability map
  (`.nii.gz`), a picture, the side and the regions. `--unseen` tests the final model on every patient it never saw.
- `tools/v7_car_linelength.py`: a line-length and common-average check of the labels (it confirms 10 of 11 patients).
  `sub-PID20` is left out of training (doubtful coordinates).

## Version 6.1: two detectors together, chosen by the expert benchmark

- **EI + line length together.** On 40 real seizures marked by epileptologists, no single detector was best: they
  succeed on *different* seizures. Used together (a site is an onset site when either marks it), they agree with the
  experts clearly better: test F1 0.40 against 0.30 for the EI alone. For the patient labels, the SOZ F1 is 0.71
  against 0.40.
- **Safety rules for both detectors:**
  - a site already active before the onset is "uncertain";
  - a detector is not used for a seizure with a state change;
  - the search ends 10 s after the marked onset.
- **SOZ = where the seizures usually start:** an onset site in at least 2 usable seizures and at least 25% of them.
- SOZ sites on the side that holds less than a quarter of a patient's SOZ sites become uncertain.

## Version 6: a new onset detector, and an expert benchmark to choose the method

- An 8-feature onset detector (`SOZ/onset_lib.py`, based on *Imprint*, Gascoigne et al. 2024): line length, energy and
  6 frequency bands, measured against each site's own normal period.
- **The method is chosen by an expert test, not by hand** (`tools/benchmark_onset.py`). It uses 40 seizures marked by
  3–5 epileptologists; the methods are tuned on half of the patients and compared on the other half.
- Unusable seizures are left out, and every report says which and why. Every patient is judged by the same checks.
- Logs: every script writes what it prints to `outputs/logs/<script>.txt`.
- New tools: `check_install.py`, `get_benchmark.py`, `benchmark_onset.py`, `plot_seizure.py --audit`.

## Version 5: first pipeline

- The Epileptogenicity Index (EI) on the seizure recordings → SOZ labels → region features, baselines and a first 3D
  CNN on T1 cubes.
