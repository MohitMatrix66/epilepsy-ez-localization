# Results (version 9)

Every number on this page comes from the files in [results/v9](../results/v9). Those are the outputs of
`tools/model_report.py`, `08_predict_mri.py --unseen` and `new_patient_results/run_new_patients.py` on the real
dataset. Read the [limitations](LIMITATIONS.md) together with these numbers.

> [!IMPORTANT]
> The version 9 design was changed after looking at the version 8.1 results on the same unseen patients (to remove a
> shortcut). The unseen-patient numbers are therefore honest but **not fully independent**. Please say so when you
> report them.

## 1. Summary

| Question | Result | Reference |
|---|---|---|
| **Side**, 20 unseen patients (15 with a one-sided clinical record), 8 mm grid | **12 / 15 = 80%** (95% CI 55–93%); fair score (AUROC) **0.84** | always "left": 67% |
| Side, the 10 MRI-only patients with a one-sided record, 6 mm grid | 10 / 10 (95% CI 72–100%); fair score 1.00 | always "left": 60% |
| Side model (07), leave-one-patient-out, 25 patients | 56% (AUROC 0.65); 7 / 10 when it is sure | majority: 64% |
| SOZ sites ranked within patients (10 training patients, leave-one-patient-out) | AUROC **0.73** (0.62–0.83) | anatomy alone: 0.64 (0.50–0.77) |
| CNN minus anatomy (sites) | +0.09 (−0.05 to +0.22), better in 6 of 10 patients | 0 = no gain |
| Sclerosis check: is the sick hippocampus scored higher? | 4 / 5 (8 mm grid), 3 / 5 (6 mm grid) | – |
| Lobe of the top region = clinical lobe | 8 / 14 | always "temporal": 86% |
| Top region in a deep structure (the v8.1 shortcut) | **0 / 20** | v8.1: 19 / 20 |
| Correlation of the score with fluid in the cube | −0.09 | above +0.5 = shortcut |

**What this means.**

- *The side is the useful output.* On new patients the CNN found the side of the seizures in 4 of 5 cases with a
  one-sided record, clearly above the always-left baseline, and its left-right margin ranks the patients well (AUROC
  0.84). The interval is wide (55–93%) because only 15 patients count.
- *The margins are small* (0.002 to 0.04 in mean probability). Two patients changed side between the 8 mm and 6 mm
  grids (PID11 and PID23), so a patient with a tiny margin should be read as "not sure".
- *The location inside the hemisphere is not patient-specific yet.* The top region is almost always medial temporal or
  medial occipital (parahippocampal, fusiform, lingual, amygdala, entorhinal), whatever the patient. In 7 of 20
  patients the top region is even on the opposite side to the side call. The "8 of 9 correct lobes" of the MRI-only
  run (below) only reflects that all 9 have temporal-lobe epilepsy.
- *The v8.1 shortcut is gone.* Deep structures are no longer the top region, and the score does not follow the fluid.
- *The side model (07) does not beat the baseline.* Its features are nearly collinear, there are no healthy controls to
  set what "normal asymmetry" is, and the thickness feature is missing in this dataset.

## 2. Training patients: leave-one-patient-out (`results/v9/model_report.txt`)

Each patient is predicted by models that never saw him or her.

| Patient | Sites | SOZ | CNN AUROC | Anatomy AUROC | SOZ in CNN top 3 |
|---|---|---|---|---|---|
| PID02 | 69 | 8 | 0.49 | 0.80 | yes |
| PID04 | 49 | 4 | 0.84 | 0.53 | yes |
| PID08 | 89 | 14 | 0.62 | 0.49 | yes |
| PID12 | 81 | 14 | 0.42 | 0.45 | no |
| PID17 | 36 | 11 | 0.90 | 0.66 | yes |
| PID18 | 76 | 2 | 0.70 | 0.24 | no |
| PID19 | 47 | 11 | 0.99 | 0.88 | yes |
| PID21 | 55 | 3 | 0.73 | 0.86 | no |
| PID22 | 86 | 13 | 0.77 | 0.56 | yes |
| PID25 | 91 | 6 | 0.81 | 0.90 | no |
| **all (mean over patients)** | **679** | **86** | **0.73** [0.62–0.83] | **0.64** [0.50–0.77] | **6 of 10** (random: 34%) |

**Regions with electrodes** (117 regions, 37 SOZ). The CNN, scored with the mean of its sites, reaches AUROC 0.74
(0.51–0.92), against 0.60 for the region prior and 0.48–0.49 for logistic regression and random forest on FreeSurfer
features. But the **number of electrodes in a region** reaches 0.77 without any MRI. Clinicians implant more
electrodes where they already suspect the onset, and this sampling bias limits what any label-based model can show.

**Left-right tests.**

- Each SOZ site against the same place on the other side: SOZ scored higher in 0.61 of cases (0.43–0.80; 0.5 = cannot
  tell).
- The epileptic region against the same region on the other side: higher in 6 of 7 comparisons.
- Patients with electrodes on both sides: the model prefers the *other* side for the non-SOZ sites (AUROC 0.24). This
  is one of the reasons the side is read from the whole hemisphere, not from single sites.

**Deep structures and shortcut** (version 9 checks). The deep "not SOZ" cubes have a mean probability of 0.42 (SOZ
sites 0.57, other sites 0.50; v8.1: deep 0.74). 70% of them are below the patient's median site. The correlation of
the score with the fluid in the cube is −0.09.

## 3. The 20 patients the model was never trained on (`results/v9/unseen_validation.txt`)

Final models, 8 mm grid. "Margin" is the mean probability of the left hemisphere minus the right one. Patients with an
inconclusive or bilateral clinical side are shown but not counted.

| Patient | Clinical side | CNN side | Correct | Margin L−R | Side model | Top region | Clinical lobe | 3T MRI report |
|---|---|---|---|---|---|---|---|---|
| PID01 | Left only | L | ✔ | +0.0263 | L | ctx-rh-lingual | temporal | Potential Epileptogenic Lesion |
| PID03 | Left only | L | ✔ | +0.0416 | not sure | ctx-lh-lingual | parietal | Nonspecific Abnormality |
| PID05 | Left only | L | ✔ | +0.0317 | L | ctx-lh-parahippocampal | temporal | Normal |
| PID06 | Right only | R | ✔ | −0.0249 | L | ctx-lh-parahippocampal | temporal | Potential Epileptogenic Lesion |
| PID07 | Left > Right | L | ✔ | +0.0359 | not sure | ctx-lh-lingual | frontal | Nonspecific Abnormality |
| PID09 | Inconclusive | R | – | −0.0102 | not sure | ctx-rh-parahippocampal | – | Potential Epileptogenic Lesion |
| PID10 | Bilateral independent | L | – | +0.0173 | not sure | ctx-lh-parahippocampal | temporal | Normal |
| PID11 | Inconclusive | R | – | −0.0074 | not sure | ctx-lh-parahippocampal | – | Normal |
| PID13 | Left only | L | ✔ | +0.0090 | L | ctx-rh-fusiform | temporal | Nonspecific Abnormality |
| PID14 | Right only | R | ✔ | −0.0018 | not sure | Right-Amygdala | – | Nonspecific Abnormality |
| PID15 | Inconclusive | L | – | +0.0067 | not sure | ctx-lh-parahippocampal | – | Not Indicated/Not Performed |
| PID16 | Left only | R | ✘ | −0.0069 | not sure | ctx-rh-lingual | temporal | Potential Epileptogenic Lesion |
| PID20 | Left only | L | ✔ | +0.0131 | not sure | ctx-lh-lingual | temporal | Potential Epileptogenic Lesion |
| PID23 | Left only | R | ✘ | −0.0133 | not sure | ctx-lh-entorhinal | temporal | Potential Epileptogenic Lesion |
| PID24 | Left only | L | ✔ | +0.0196 | not sure | ctx-rh-parahippocampal | temporal | Nonspecific Abnormality |
| PID26 | Right > Left | R | ✔ | −0.0181 | R | ctx-rh-lingual | temporal | Potential Epileptogenic Lesion |
| PID27 | Right only | L | ✘ | +0.0159 | L | Right-Amygdala | temporal | Nonspecific Abnormality |
| PID28 | Left only | L | ✔ | +0.0061 | not sure | Left-Amygdala | temporal | Potential Epileptogenic Lesion |
| PID29 | Inconclusive | L | – | +0.0063 | R | ctx-lh-lingual | – | Potential Epileptogenic Lesion |
| PID30 | Right only | R | ✔ | −0.0124 | not sure | ctx-rh-fusiform | – | Nonspecific Abnormality |

- **Side (CNN):** 12 of 15 = 80% (95% CI 55–93%). Balanced accuracy 0.80, fair score 0.84. The CNN said "left" for 12
  of 20 patients.
- **Side model:** 4 of 6 correct when it was sure (9 "not sure"), fair score 0.48.
- **Sclerosis check:** 4 of 5 patients with one-sided hippocampal sclerosis on the MRI report (PID01, 06, 23, 28, 29).
- **Lobe:** 8 of 14 (always "temporal": 86%).
- Two of the three side errors (PID16, PID27) are iEEG patients whose labels did not pass the checks (no repeated
  onset), possibly harder cases.

## 4. The 15 MRI-only patients, compared with the clinical record (`results/v9/comparison_mri_only.txt`)

These patients never had iEEG. The run used `new_patient_results/run_new_patients.py` with a 6 mm grid, and the maps
were saved per patient.

| Patient | Clinical side | CNN side | Correct | Side model | Clinical lobe | Top region | MRI lesion (side) |
|---|---|---|---|---|---|---|---|
| PID01 | Left only | L | ✔ | L | Temporal | ctx-rh-lingual | mesial temporal sclerosis (L) |
| PID05 | Left only | L | ✔ | L | Temporal | ctx-rh-parahippocampal | – |
| PID06 | Right only | R | ✔ | L | Temporal | ctx-rh-parahippocampal | mesial temporal sclerosis (R) |
| PID09 | Inconclusive | R | – | not sure | Non-localizable | ctx-rh-parahippocampal | malformation of cortical development (L) |
| PID10 | Bilateral independent | L | – | not sure | Temporal | ctx-rh-parahippocampal | – |
| PID11 | Inconclusive | L | – | not sure | Non-localizable | ctx-lh-parahippocampal | – |
| PID13 | Left only | L | ✔ | L | Temporal | ctx-rh-fusiform | – |
| PID14 | Right only | R | ✔ | not sure | Non-localizable | Right-Amygdala | – |
| PID15 | Inconclusive | L | – | not sure | Non-localizable | ctx-lh-parahippocampal | – |
| PID23 | Left only | L | ✔ | not sure | Temporal | ctx-lh-entorhinal | mesial temporal sclerosis (L) |
| PID24 | Left only | L | ✔ | not sure | Temporal | ctx-lh-fusiform | – |
| PID26 | Right > Left | R | ✔ | R | Temporal | ctx-lh-parahippocampal | ventriculomegaly and bilateral hippocampal atrophy (B) |
| PID28 | Left only | L | ✔ | not sure | Temporal | Left-Amygdala | mesial temporal sclerosis (L) |
| PID29 | Inconclusive | L | – | R | Multi-focal | ctx-lh-parahippocampal | mesial temporal sclerosis (R) |
| PID30 | Right only | R | ✔ | not sure | Multi-focal | ctx-rh-fusiform | – |

- **Side (CNN):** 10 of 10 = 100% (95% CI 72–100%), fair score 1.00 (always "left": 60%). With the 8 mm grid of
  section 3, PID23 was on the other side, which gives 9 of 10.
- **Side model:** 4 of 5 when sure (5 "not sure").
- **Side against a one-sided MRI lesion:** 4 of 6. The two misses (PID09, PID29) both have an inconclusive clinical
  side.
- **Sclerosis check:** 3 of 5.
- **Lobe:** 8 of 9, but all 9 are temporal, so always saying "temporal" also gives 9 of 9.
- **The top region is on the opposite side to the CNN's side call** in 5 of 15 patients (PID01, 05, 10, 13, 26).

## 5. Files

| File | Content |
|---|---|
| [model_report.txt](../results/v9/model_report.txt) | the full report of `tools/model_report.py` (sections 1–6 above) |
| [unseen_validation.txt](../results/v9/unseen_validation.txt) / [.csv](../results/v9/unseen_validation.csv) | the 20 unseen patients (section 3) |
| [comparison_mri_only.txt](../results/v9/comparison_mri_only.txt) | the 15 MRI-only patients (section 4) |
| [tables/](../results/v9/tables) | metrics and every out-of-fold prediction (sites, mirror controls, deep cubes, regions, baselines) |
| [labels/](../results/v9/labels) | the reports of the iEEG labelling (steps 2 and 3) |
| [logs/](../results/v9/logs) | what every step printed, including the Colab training log |
