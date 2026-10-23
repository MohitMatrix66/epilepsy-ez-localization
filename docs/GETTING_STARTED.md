# Getting started

This guide goes from an empty computer to a probability map on an MRI. You don't need any experience with deep
learning. Every command is shown for **Windows** (Anaconda Prompt) and for **macOS / Linux** (terminal).

| You want to… | Read |
|---|---|
| install the project and check that it works | steps 1–4 below |
| use the trained model on a dataset patient or on your own MRI | steps 5–7 |
| train the model again from the raw data | [REPRODUCE.md](REPRODUCE.md) |

## 1. What you need

- A computer with Windows 10/11, macOS or Linux, at least 8 GB of RAM, and about 3 GB of free disk space (about 8 GB
  if you reproduce the training).
- **Miniconda** or Anaconda (free): <https://docs.conda.io/en/latest/miniconda.html>.
- Optionally **Git** (<https://git-scm.com/downloads>). Without Git, you can use *Code → Download ZIP* on the GitHub
  page.
- No GPU is needed to use the model. Training the CNN uses a free Google Colab GPU.

## 2. Download the project

With Git:

```bash
git clone https://github.com/MohitMatrix66/epilepsy-ez-localization.git
cd epilepsy-ez-localization
```

Without Git: on the GitHub page, click **Code → Download ZIP**, unzip it, then `cd` into the unzipped folder.

## 3. Create the Python environment

```bash
conda env create -f environment.yml
conda activate ez-localization
```

This installs Python 3.12 and the exact package versions the model was tested with. Activate the environment
(`conda activate ez-localization`) every time you open a new terminal.

<details>
<summary>Without conda (plain pip)</summary>

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Linux without an NVIDIA GPU, first install the smaller CPU-only PyTorch:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
</details>

## 4. Check the installation

```bash
python tools/check_install.py --network
```

It must end with **`INSTALL OK`**. It checks the Python packages, that every project file is complete
(`tools/FILES.sha256`), that the data folder can be written, and that the public dataset on AWS can be reached.

Then test the whole pipeline on synthetic patients. Nothing is downloaded, and it takes 10–30 minutes on a laptop CPU:

```bash
python tools/smoke_test.py
```

It must end with **`SMOKE TEST PASSED`**. The fake patients copy the quirks of the real dataset (two voxel sizes, a
tilted head, electrode names that differ between files, artefacts). Results on fake patients mean nothing
clinically; the test only shows that every step runs.

## 5. Put the trained model in place

The trained version 9 model is in `models/v9/`. The scripts read it from `outputs/`, so copy it there once:

```bash
# Windows (Anaconda Prompt)
mkdir outputs
copy models\v9\* outputs\

# macOS / Linux
mkdir -p outputs && cp models/v9/* outputs/
```

## 6. Run the model

**A patient of the dataset.** The script downloads only that patient's FreeSurfer files from AWS (into
`~/ez_data`; on Windows `C:\Users\<you>\ez_data`):

```bash
python 08_predict_mri.py --subject sub-PID06
```

**Your own MRI.** The model needs FreeSurfer's `recon-all` output, because it uses `mri/norm.mgz` (the T1 image) and
`mri/aparc+aseg.mgz` (the brain regions). With FreeSurfer installed (<https://surfer.nmr.mgh.harvard.edu>), this
takes a few hours per brain:

```bash
recon-all -subject P31 -i /path/to/T1.nii.gz -all       # writes $SUBJECTS_DIR/P31
python 08_predict_mri.py --fs-dir $SUBJECTS_DIR/P31 --name P31
```

`T2.mgz`, surfaces and labels are used when they exist (by the side model), but they are not required. Use the T1
image at its original resolution, as `recon-all` makes it: any isotropic voxel size works, because the cubes are cut
in millimetres.

**Every patient the model never saw, against the clinical record:**

```bash
python 08_predict_mri.py --unseen                         # 20 patients; about 30-45 min on a laptop
python new_patient_results/run_new_patients.py            # the 15 MRI-only patients, maps saved per patient
```

Useful options of `08_predict_mri.py`:

| Option | Meaning |
|---|---|
| `--step 5` | distance in mm between predictions (smaller = finer map, slower; default 5, and 6 with `--unseen`) |
| `--zone-percent 2` | size of the "zone": the highest-scoring share of the assessed grey matter |
| `--no-maps` | with `--unseen`: only the table, no maps (faster) |
| `--models final` | use the final models even for a training patient (then it is **not** a test) |

The 10 training patients (PID02, 04, 08, 12, 17, 18, 19, 21, 22, 25) are normally shown with the models of their own
cross-validation fold, which never saw them. Those fold models are not included in `models/` (30 more files); retrain
(see [REPRODUCE.md](REPRODUCE.md)) or use `--models final` and keep in mind that it is not a test.

## 7. Look at the result

Everything is saved in `outputs/predictions/<name>/`. The `.png` picture can be opened directly. For the 3D map:

**ITK-SNAP** (free, <http://www.itksnap.org>)

1. *File → Open Main Image*: the patient's `mri/norm.mgz`.
2. *File → Add Another Image*: `<name>_ez_probability.nii.gz`, shown as an overlay with the "Hot" colour map.
3. *Segmentation → Open Segmentation*: `<name>_ez_zone.nii.gz` (the zone, in red).

**Freeview** (comes with FreeSurfer)

```bash
freeview -v mri/norm.mgz <name>_ez_probability.nii.gz:colormap=heat:opacity=0.6 <name>_ez_zone.nii.gz:colormap=lut
```

**3D Slicer** (free, <https://www.slicer.org>): drag both files into the window and choose the probability map as the
foreground with a colour lookup table.

How to read it: brighter means the model thinks seizures are more likely to start there. The grey dashed outline
marks the deep structures the model does not assess. Always show the honest accuracy table
(`outputs/predictions/unseen_validation.txt`) next to any picture.

## Troubleshooting

| Message | What to do |
|---|---|
| `INSTALL OK` missing, packages listed as MISSING | activate the environment, then `python -m pip install <package>` |
| `files missing or not version 9` | the download is incomplete or files were edited. Download again into a new folder. If you changed code on purpose, see [CONTRIBUTING.md](../CONTRIBUTING.md). |
| `cnn_model.json not found` | step 5: copy `models/v9/*` into `outputs/` |
| `... cnn_fold1_s0.pt is missing` | you asked for a training patient; see the note about fold models in step 6 |
| `must contain norm.mgz and aparc+aseg.mgz` | `--fs-dir` must point to a FreeSurfer subject folder (the one that contains `mri/`) |
| a script stops with an error | the whole output, with the error, is in `outputs/logs/<script>.txt`. Please attach it to a GitHub issue. |
| the AWS check fails | a university or company proxy may block it. Try another network. |
