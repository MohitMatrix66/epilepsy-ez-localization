# Limitations, and what would fix them

Version 9 predicts the **side** of the seizure onset reasonably well on new patients (80% on 15 unseen patients, fair
score 0.84). It does **not** yet find a patient-specific **location** inside the hemisphere. This page explains why,
separates what comes from the data from what comes from the design, and lists the next steps.

> [!CAUTION]
> This is a research prototype trained on 10 patients. It must not be used to make clinical decisions.

## What the model does today

| Output | Status |
|---|---|
| Side (left / right) | promising: 12/15 unseen patients, AUROC 0.84; but small margins, so a tiny margin means "not sure" |
| Ranking of the implanted sites within a patient | above chance (AUROC 0.73), not clearly above anatomy alone (0.64) |
| Location inside the hemisphere | **not patient-specific**: the top region is almost always medial temporal or occipital |
| Deep-structure shortcut of v8.1 | fixed (0/20 top regions in deep structures) |
| Hippocampal sclerosis side | 3–4 of 5, too few patients to tell |

## 1. Limits of the data (the main cause)

1. **Very few labelled patients.** There are 10 training patients with 86 seizure-onset sites: 37 medial temporal
   (43%), 20 lateral temporal, 17 in other cortex and 12 in white matter. The easiest rule for a CNN to learn is
   "medial-temporal-looking tissue = onset". That is anatomy, not disease, and it is why the map looks like an
   average temporal-lobe-epilepsy map. Published lesion-detection models use hundreds of patients plus healthy
   controls.
2. **Labels exist only where electrodes were implanted.** Clinicians implant where they already suspect the onset.
   The *number of electrodes* in a region predicts the SOZ regions (AUROC 0.77) better than any MRI feature. Most of
   the brain has no label at all.
3. **Half of the patients have a normal-looking MRI.** 15 of 30 MRI reports are "normal" or "nonspecific". This is
   the hardest group for any MRI model.
4. **T1 only.** Hippocampal sclerosis and dysplasia show best on T2 and FLAIR. In this dataset, 2 of the 10 training
   patients (PID17, PID19) have no T2. Raw 3T FLAIR exists for 19 of 30 patients (5 of the 10 training patients), and
   there is no FreeSurfer FLAIR.
5. **No healthy controls.** Without them the model cannot learn what a *normal* left-right difference looks like.
   This is the main reason the side model (07) did not beat the baseline.
6. **No resection masks or per-patient outcomes.** The paper reports Engel outcomes for 15 patients, but the public
   tables have no per-patient outcome column. So the labels cannot be checked against "removed, and seizure-free".
7. **Automatic labels.** The dataset has no clinician SOZ marks. The onset detector agrees with epileptologists with
   an F1 of 0.40 on the expert benchmark; one epileptologist against the others reaches 0.66.
8. **A small test set, looked at twice.** 15 patients with a one-sided clinical record give wide intervals (55–93%).
   The version 9 changes were made after seeing the version 8.1 results on them.

## 2. Limits of the design (fixable, but only with more data)

- **Local view.** The CNN sees a 32 mm cube (plus its mirror), never the whole brain.
- **The zone is a ranking, not a segmentation.** The "zone" is the highest-scoring 2% of the grey matter. A true
  lesion segmentation needs lesion outlines to train on, and this dataset has none.
- **The side comes from tiny differences.** The left and right mean probabilities differ by 0.002–0.04. Changing only
  the grid spacing flipped 2 of 20 patients.
- **Scope.** The model covers drug-resistant *focal* epilepsy. It is not meant for generalized epilepsy. Deep lesions
  (hypothalamic hamartoma, periventricular heterotopia) are not assessed, because there were no labels there.

## 3. Why more tuning on the same patients would not help

Every unseen patient has already been used to judge version 8.1 and version 9. More changes chosen on them would make
the numbers look better without making the model better. A real improvement has to come from new data, and it has to
be tested on patients that have not been looked at.

## 4. Roadmap

In order of expected impact:

1. **Train on a larger dataset with resection masks and outcomes.** For example, open multi-centre epilepsy surgery
   datasets such as IDEAS have hundreds of operated patients, resection outlines, FLAIR and healthy controls. Use them
   for training, and use this Penn dataset as a truly independent test.
2. **Add healthy controls**, so that left-right differences are measured against normal brains.
3. **Use T2 and FLAIR for every patient** as extra CNN channels. They need a T2/FLAIR → T1 registration when
   FreeSurfer does not provide them.
4. **Combine with a dedicated hippocampal-sclerosis tool** (such as AID-HS) for the side in mesial temporal sclerosis.
5. **Report "not sure" when the margin is tiny.** Fix this rule *before* the next test.
6. **Better labels**: clinician-marked onset zones, or resected tissue in seizure-free patients, instead of
   detector-based labels only.
7. **A new, untouched test set** for the next version.
