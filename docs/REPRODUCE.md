# Reproduce the results from scratch

This guide repeats the whole study: it chooses the onset detector on the expert benchmark, makes the seizure-onset
labels from the iEEG recordings, builds the MRI cubes, trains the CNN on Google Colab, and tests the final model on the
patients it never saw. The section numbers are the ones the scripts' messages refer to ("README section 5" = section 5
here).

Everything is read from the public dataset on AWS (no account, free). Only the files the project needs are
downloaded, into `~/ez_data` (Windows: `C:\Users\<you>\ez_data`; change `CACHE_DIR` in `config.py` to use another
folder). In total that is about 4 GB. Every script also writes what it prints to `outputs/logs/<script>.txt`.

Commands are written with `/`. They work the same in the Windows Anaconda Prompt.

## 1. Install (once)

Follow steps 1–4 of [GETTING_STARTED.md](GETTING_STARTED.md): clone the repository, create the environment, and run
`python tools/check_install.py --network`, which must end with `INSTALL OK`. It also checks the recordings already
downloaded in your data folder. If some are damaged, `python tools/check_install.py --fix` deletes them, and they are
downloaded again.

Coming from an older version? Keep your data folder (all downloads). The seizure-onset labels have not changed since
version 6.1, so you can copy an old `SOZ/results` folder into the new project and start at section 5.

## 2. Test on fake patients (about 10–30 min)

```bash
python tools/smoke_test.py
```

It must end with `SMOKE TEST PASSED`. It tests the onset detector on signals with a known answer, the whole expert
benchmark (a fake one, served from your own computer), and every step on fake patients that copy the real dataset's
quirks.

## 3. Choose the onset method: the expert benchmark (30–60 min the first time, mostly download)

```bash
python tools/get_benchmark.py --test     # must say CONNECTION OK
python tools/get_benchmark.py            # about 0.8 GB: only the 63 s around each onset; can be stopped and restarted
python tools/benchmark_onset.py
```

`benchmark_onset.py` runs every onset method on 40 seizures whose onset channels were marked by 3–5 epileptologists
(Penn, the same hospital as the dataset). It tunes the methods on half of the patients and compares them on the
other half, next to the agreement of one epileptologist with the others (the human level). It ends with
`CHOSEN: ...`. On the real benchmark this is **EI + line length** (test F1 0.40, against 0.30 for the EI alone). The
choice is saved in `SOZ/results/benchmark_choice.json` and used by step 2 of section 4.

## 4. Seizure-onset labels from the real data (folder `SOZ`)

| Step | Command | Downloads | Result in `SOZ/results` |
|---|---|---|---|
| 1 | `python SOZ/01_scan_dataset.py` | small tables, about 30 MB | usable patients, file formats |
| 2 | `python SOZ/02_compute_soz.py` | seizure recordings, about 2.2 GB, one at a time (kept for later runs) | onset of every seizure, SOZ / uncertain labels, `step2_report.txt` |
| check | `python tools/check_unlocalized.py --all` | – | do seizures start on recorded contacts that have no coordinates? |
| 3 | `python SOZ/03_build_contacts.py` | FreeSurfer files and reference MRI, about 0.5 GB | `contacts.csv` (the training table), coordinate and clinical checks, `step3_report.txt` |
| audit | `python tools/plot_seizure.py --audit` | – | one picture per patient in `outputs/seizure_plots/audit` |

Look at `step2_report.txt`, `step3_report.txt`, the `check_unlocalized` log and the audit pictures before you train.
The reports of the published run are in [results/v9/labels](../results/v9/labels).

**How the seizure-onset zone (SOZ) is found.** For every seizure, two detectors look for where it starts:

- the **EI** (Epileptogenicity Index) follows the ratio of fast to slow activity, so it sees low-voltage fast activity;
- **line length** follows how much the signal moves, so it sees rhythmic discharges, also slow ones.

Each detector gives onset sites. A site that was already active before the onset is not an onset site; it becomes
"uncertain". A detector is not used for a seizure in which many sites were already active (a state change, such as
slow waves fading before the seizure). Line length is also not used when more than 40% of the sites start together.
A seizure is usable when at least one detector finds onset sites. Then, per patient:

- **SOZ site:** an onset site in at least 2 usable seizures **and** in at least 25% of them. Several foci in one
  patient all count.
- **Uncertain site:** an onset site less often than that, or a site already active before the onset in some seizure.
  In step 3, SOZ sites on the side that holds less than a quarter of a patient's SOZ sites also become uncertain.
  Uncertain sites are used neither as SOZ nor as "not SOZ".
- **Trusted patient:** at least 2 usable seizures, and the SOZ (or a site next to it) is among the onset sites in at
  least 30% of them. Patients whose label is `single seizure`, `no repeated onset`, `SOZ rarely leads` or
  `no usable seizure` are left out of training. Their MRI is still used by `07_lateralization.py` and by the
  unseen-patient test.

This is still an automatic estimate: the dataset has no SOZ marks from clinicians. But the method is the one that
agreed best with the epileptologists on the benchmark, and step 3 compares the result with the clinical record.

Notes:

- Step 2 can be stopped and restarted; finished seizures are not done again. It saves every seizure in
  `SOZ/results/seizure_arrays`, so `python SOZ/02_compute_soz.py --aggregate-only` re-makes the labels in seconds,
  without downloading.
- Electrode-name fixes (when the recordings and the coordinate files name an electrode differently) are in
  `CHANNEL_ALIASES` in `config.py`. After a fix, run `python SOZ/02_compute_soz.py --subjects sub-PID02 --redo`.
- Patients left out on purpose are listed, with the reason, in `EXCLUDE_FROM_TRAINING` in `config.py` (version 9:
  sub-PID20, whose coordinates are doubtful).
- `python tools/plot_seizure.py --subject sub-PID04 --all` draws every seizure of one patient.

## 5. The model

| Step | Command | Where |
|---|---|---|
| 0 | `python SOZ/03_build_contacts.py` (again, only after changing `EXCLUDE_FROM_TRAINING`) | laptop |
| 1 | `python 01_label_contacts.py` | laptop |
| 2 | `python 02_region_features.py` | laptop |
| 3 | `python 03_train_baseline.py` | laptop |
| 4 | `python 04_make_patches.py`: cube + mirror cube, mirror controls, deep-structure cubes; 32 mm on any MRI | laptop |
| 4b | `python 07_lateralization.py`: the side model, needed by step 8 | laptop |
| 5 | `colab/train_cnn_colab.ipynb`: runs `05_train_cnn.py --lopo --resume --final --seeds 3` (about 15–40 min) | Google Colab GPU |
| 6 | download `outputs/colab_results.zip` from Google Drive and extract it into the project folder | laptop |
| 7 | `python tools/model_report.py`: the fair test of the model, per patient (`outputs/model_report.txt`) | laptop |
| 8 | `python 08_predict_mri.py --unseen`: the honest test on the patients the model never saw (30–45 min) | laptop |
| 9 | `python 08_predict_mri.py --subject sub-PID01`: one picture and map (section 6) | laptop |
| extra | `python new_patient_results/run_new_patients.py`: maps and a clinical comparison for the 15 MRI-only patients | laptop |
| extra | `python 06_make_heatmap.py --subject sub-PID19 --pred outputs/cnn_predictions_regions.csv --col p_cnn` | laptop |
| extra | `python tools/check_dataset_extras.py`: what else the dataset holds (T2, FLAIR, 7T scans, outcome tables) | laptop |

**Training on Google Colab (step 5).**

1. Upload the whole project folder to Google Drive, into *My Drive* (*New → Folder upload*). It must contain
   `outputs/patches.npz` from step 4. You don't need `synthetic_data` or `outputs_synthetic`. Delete an old copy in
   Drive first.
2. Open `colab/train_cnn_colab.ipynb` in Colab. Choose *Runtime → Change runtime type → T4 GPU*.
3. In cell 3, set `PROJECT` to the folder's path in Drive, for example
   `/content/drive/MyDrive/epilepsy-ez-localization`.
4. Run the cells one by one. If Colab disconnects, run them again: training continues where it stopped (`--resume`).
5. Download `outputs/colab_results.zip` from Drive and extract it into the project folder (step 6). It puts the model
   files into `outputs/`.

All results go to `outputs/`. In this dataset, 2 of the 10 training patients have no T2 image and there are no
FreeSurfer thickness tables, so the CNN uses the T1 image only. Not used for training: contacts more than 5 mm outside
the brain (the outer end of depth electrodes, in the skull) and uncertain sites.

Step 4b is a second, simpler question that uses every patient with a clinical answer (25 instead of 10): can the MRI
tell which hemisphere the seizures come from? The labels come from `metadata_3T_7T.xlsx`.

**Be careful with the test patients.** Choosing settings after looking at the results of the unseen patients makes
the model look better than it is. If you change something after a test, say so when you report the numbers (version 9
does this: it was designed after looking at the version 8.1 results).

## 6. The final model on an MRI

```bash
python 08_predict_mri.py --subject sub-PID01                    # a dataset patient
python 08_predict_mri.py --fs-dir /path/to/new_patient --name P31   # a NEW patient's FreeSurfer folder
python 08_predict_mri.py --unseen                               # every patient the model never saw
```

The results are in `outputs/predictions/<name>/`: `<name>_ez_probability.nii.gz` (open it on top of the patient's
T1, `norm.mgz` or `T1.mgz` of the same FreeSurfer folder, in ITK-SNAP, Freeview or 3D Slicer), `<name>_ez_zone.nii.gz`
(the highest 2% of the assessed grey matter), a picture, the regions and a summary. A new patient needs FreeSurfer
`recon-all` first, which makes `mri/norm.mgz` and `mri/aparc+aseg.mgz`.

Be honest in a presentation. Show the map of a patient the model **never** saw: any patient outside the 10 training
patients, or a training patient shown with the models of his own cross-validation fold. Show the `--unseen` table
(`outputs/predictions/unseen_validation.txt`) next to it, because that is the model's real accuracy on new patients.

## When something goes wrong

- A script stops with an error: the whole output and the error are in `outputs/logs/<script>.txt`. Please attach it to
  a GitHub issue.
- `get_benchmark.py --test` fails: the network (a university proxy, for example) may block the server. Try another
  network.
- Step 2 says the onset method has not been chosen: run section 3 first.
- `check_install.py` says files are not version 9: download the repository again into a new, empty folder.
- `05_train_cnn.py` says `patches.npz` is not version 9: run steps 1–4 of section 5 again, and upload the folder to
  Drive again.
- `08_predict_mri.py` says `cnn_model.json` is missing or old: copy the Colab results (step 6 of section 5), or the
  published model (`models/v9/*`) into `outputs/`.
