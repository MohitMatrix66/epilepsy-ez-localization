# Method

This page describes version 9 of the model: what goes in, how the labels are made, how the CNN is trained and
tested, and why each design choice was made. The history of the changes is in [CHANGELOG.md](../CHANGELOG.md).

## 1. Data

**Penn 3T-7T Paired Epilepsy Imaging** (Imtiaz et al., *Scientific Data* 2026; Pennsieve Discover,
doi:10.26275/aosz-hj9u). It has 30 adults with drug-resistant focal epilepsy (18 female, 38.8 ± 11.7 years), with 3T and
7T MRI, FreeSurfer outputs for both, and clinical metadata. For the 15 patients who later had intracranial EEG (iEEG),
it also includes seizure recordings and electrode coordinates. [DATASET.md](DATASET.md) lists what the project uses.

| Group | Patients | Used for |
|---|---|---|
| iEEG patients whose seizure-onset labels pass every check | 10: PID02, 04, 08, 12, 17, 18, 19, 21, 22, 25 | training the CNN (leave-one-patient-out) |
| iEEG patients left out | 5: PID03, 16, 27 (no repeated onset), PID07 (a single usable seizure), PID20 (doubtful coordinates) | unseen test |
| MRI-only patients (no iEEG) | 15: PID01, 05, 06, 09, 10, 11, 13, 14, 15, 23, 24, 26, 28, 29, 30 | unseen test |

The **clinical reference** is the metadata table `metadata_3T_7T.xlsx`. It holds the seizure-onset lateralization and
localization decided by the multidisciplinary epilepsy surgery conference, and the 3T and 7T MRI reports. It is a strong
reference but not an independent ground truth: for iEEG patients it may already include the iEEG.

## 2. Seizure-onset labels from iEEG (`SOZ/`)

The dataset has no clinician marks of the seizure-onset zone (SOZ), so the labels are computed from the recordings.

- **Montage:** bipolar (neighbouring contacts on the same electrode); a "site" is the midpoint of the pair.
- **Detectors:**
  - the **Epileptogenicity Index** (EI; Bartolomei et al., 2008) sees the rise of fast activity, typical of
    low-voltage fast onsets;
  - **line length** (Esteller et al., 2001) sees rhythmic discharges, including slow ones.

  Both are measured against each site's own pre-ictal period.
- **The method is chosen on an expert benchmark**, not by hand. The benchmark has 40 spontaneous seizures (19 adult
  patients, Penn) whose onset channels were marked independently by 3–5 epileptologists (Ojemann et al. 2026;
  CC BY 4.0; [benchmark/](../benchmark)). Each method was tuned on half of the patients and compared on the other half:

  | Method | Test F1 |
  |---|---|
  | EI alone | 0.30 |
  | **EI + line length** (a site is an onset site when either detector marks it) | **0.40** |
  | one epileptologist against the others (the human level) | 0.66 |

  The detectors succeed on *different* seizures, which is why the combination wins.
- **Safety rules:**
  - a site already active before the onset is "uncertain", not an onset site;
  - a detector is not used for a seizure in which many sites are already active (a state change);
  - line length is not used when more than 40% of the sites start together;
  - the search ends 10 s after the marked onset.
- **SOZ rule:** a site is SOZ when it is an onset site in at least 2 usable seizures **and** in at least 25% of them.
  Less frequent onset sites are uncertain. In step 3, SOZ sites on a side that holds less than 25% of a patient's SOZ
  sites also become uncertain. Uncertain sites are never used as "not SOZ".
- **Patient checks:**
  - at least 2 usable seizures;
  - the SOZ leads in at least 30% of them;
  - the contacts that start seizures have coordinates;
  - the SOZ side and lobe are compared with the clinical record.

  The SOZ side agrees with the clinical side in 12 of 12 patients, and the lobe in 9 of 10.

The result is `contacts.csv`: 975 sites in 10 patients (88 SOZ, 236 uncertain). After removing the sites more than
5 mm outside the brain and the uncertain sites, the CNN trains on **679 sites, 86 of them SOZ**.

## 3. MRI input

- **Images:** FreeSurfer `norm.mgz` (an intensity-normalised, skull-stripped T1) and `aparc+aseg.mgz` (the
  Desikan-Killiany cortical regions plus subcortical structures). T2 is not used, because 2 training patients (PID17,
  PID19) have no T2.
- **Intensity:** z-scored with the mean and standard deviation inside the brain, applied to the whole image. Nothing
  is set to zero outside FreeSurfer's labels (version 9). The fluid between the gyri therefore stays dark, as in the
  ventricles; before version 9, only places next to a ventricle had dark fluid in their cube, and the model learned
  that shortcut.
- **Cubes in millimetres:** every input is 32 × 32 × 32 samples taken 1 mm apart (trilinear interpolation), so it is
  32 mm wide on any MRI. The dataset has two 3T T1 protocols (0.8 mm for 23 patients, 1 mm for 7). Before version 8.1
  the cubes were 32 *voxels* wide, so the same hippocampus looked 25% bigger in some patients.
- **Mirror cube:** the second input channel is the cube from the same place in the other hemisphere, reflected
  through the mid-sagittal plane. The plane is estimated from the centres of left/right structure pairs, so it works
  when the head is tilted. The model can therefore compare the two sides, as a radiologist compares the two
  hippocampi.

## 4. Training examples (`04_make_patches.py`)

| Kind | Label | Why |
|---|---|---|
| SOZ sites | 1 | the target |
| other sites (not uncertain) | 0 | normal tissue that was implanted |
| **mirror controls**: the same place as a SOZ site, on the other side | 0 | same anatomy, healthy, so only the disease can separate them. Used only when the clinical record is one-sided and agrees with the SOZ side. |
| **deep-structure cubes**: 2 per structure and side (thalamus, caudate, putamen, pallidum, accumbens), at least 10 mm from any SOZ or uncertain site | 0 | electrodes almost never reach them; without these examples the v8.1 model pointed to the caudate |

## 5. Network and training (`05_train_cnn.py`)

- **SmallCNN3D**:
  - four blocks of [3×3×3 convolution → batch norm → ReLU] × 2 → max-pool, with 16, 32, 64 and 64 channels
    (32³ → 2³);
  - then global average pooling, dropout 0.3 and one linear output (a logit);
  - **437,089 parameters**;
  - input: 2 channels (T1, T1-mirror).
- **Loss and optimiser:**
  - binary cross-entropy, weighted for the class balance;
  - AdamW (learning rate 1e-3, weight decay 1e-4), batch 32.
- **Augmentation:** random flips along the three axes (the cube and its mirror are flipped together), brightness and
  contrast jitter, and Gaussian noise.
- **Validation:**
  - **leave-one-patient-out** cross-validation: 10 folds, each patient tested by models that never saw him or her;
  - 3 seeds per fold, averaged;
  - early stopping on a 3-epoch average (patience 8, at least 3 epochs, at most 40).
- **Final models:** 3 seeds trained on all 10 patients for the median best epoch of the folds (7 epochs). These are
  the files in [models/v9](../models/v9).
- **Test-time flip averaging:** every prediction is the mean over the cube and its left-right flipped copy.
- Training runs on a free Google Colab T4 GPU in 15–40 minutes ([colab/](../colab)).

## 6. From the CNN to a brain map (`08_predict_mri.py`)

1. The CNN is applied on a regular grid (every 5 mm by default) over the **assessed** regions: the cortex,
   hippocampus and amygdala, where the training labels are. The deep grey structures are outlined as *not assessed*.
2. The coarse grid is interpolated (trilinear) back to the full MRI resolution, inside the assessed grey matter. The
   result is `<name>_ez_probability.nii.gz`.
3. **Zone:** the highest-scoring 2% of the assessed grey matter (`--zone-percent`), saved as `<name>_ez_zone.nii.gz`.
4. **Side (CNN):** the hemisphere with the higher mean probability over its assessed grey matter. The margin
   (left − right) is reported, because small margins are unreliable.
5. **Regions:** the mean probability of every FreeSurfer region with at least 3 grid points, sorted.

## 7. Side model (`07_lateralization.py`)

The side model is a second, simpler model, built after AID-HS and Caldairou et al. (*Neurology* 2021). It is a
logistic regression (C = 0.3) on **five left-right asymmetries, fixed before any test**:

- the volumes of the hippocampus, amygdala and thalamus;
- the hippocampal T2 signal relative to the white matter;
- the temporal cortical thickness.

Each asymmetry is (left − right) / mean. The model says **"not sure"** when p(left) is between 0.4 and 0.6. It is tested
leave-one-patient-out on the 25 patients with a one-sided clinical record. A new MRI gets the final side model
(`models/v9/side_model.json`).

The thickness feature is zero for everyone: the dataset has no FreeSurfer thickness files.

## 8. How the model is evaluated

| Test | Question | Reference |
|---|---|---|
| Site AUROC within patients (`tools/model_report.py` §1) | are SOZ sites ranked above the other sites of the same patient? | an anatomy-only prior (how often each region is SOZ in the other patients) |
| Region AUROC (§2) | the same for brain regions, scored with the **mean** of their sites | the anatomy prior, and the number of electrodes in the region (clinicians implant more where they suspect the SOZ) |
| Left-right test (§3) | is each SOZ site scored above its mirror? | 0.5 = cannot tell the side |
| Deep-structure check (§5) | are the deep "not SOZ" cubes scored low? | the v8.1 model scored them 0.74 |
| Shortcut check (§6) | does the score just follow the amount of fluid in the cube? | correlation above +0.5 = shortcut |
| Unseen side accuracy (`08 --unseen`) | the side, on the 20 patients never used for training | always guessing the most common side; 95% Wilson interval |
| Fair side score | AUROC of the left − right margin against the clinical side | not fooled by a lean to one side |
| Sclerosis check | when the MRI report names one-sided hippocampal sclerosis, is the sick hippocampus scored higher? | – |
| Lobe check | the lobe of the top region against the clinical lobe | always saying "temporal" |

Each 95% interval comes from re-sampling the patients (bootstrap) or from the Wilson formula for proportions. With
about 10 patients the intervals are wide. An interval that contains the reference means "we cannot tell yet".

## 9. Design rules that kept the evaluation honest

- The patients used for training never appear in their own test: within the training set, by leave-one-patient-out;
  outside it, through the 20 unseen patients.
- The side-model features were fixed in advance, from the literature, and not selected on the results.
- The "not sure" band and the scope ("drug-resistant focal epilepsy; deep lesions not assessed; weakest when the MRI
  looks normal") are printed in every summary.
- The v8.1 → v9 changes were made after the v8.1 results on the unseen patients had been seen. They fixed a clear
  bug (the caudate shortcut), but the unseen patients have now been looked at twice. A fully independent test needs new
  patients.
