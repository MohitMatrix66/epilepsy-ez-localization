# Model card: epileptogenic-zone CNN, version 9

## Files (`models/v9/`)

| File | What it is |
|---|---|
| `cnn_final_s0.pt`, `cnn_final_s1.pt`, `cnn_final_s2.pt` | the three final CNNs (PyTorch `state_dict`, one per seed), trained on all 10 training patients; their predictions are averaged |
| `cnn_model.json` | model card read by the scripts: version, input channels, cube size in samples and mm, architecture settings, training settings, the cross-validation folds, the training patients |
| `side_model.json` | the side model of `07_lateralization.py`: features, standardisation, logistic-regression coefficients, "not sure" band |
| `lateralization_predictions.csv` | the side model's leave-one-patient-out answers for the 30 dataset patients; `08_predict_mri.py --subject` uses them, so a dataset patient gets an honest answer |

To use them, copy everything into `outputs/` (`cp models/v9/* outputs/`; on Windows `copy models\v9\* outputs\`). The
10 × 3 cross-validation fold models (`cnn_fold*_s*.pt`) are not included. They are only needed to show the 10
training patients with models that never saw them.

## Model

| | |
|---|---|
| Task | probability that a 32 mm cube of brain contains seizure-onset tissue |
| Input | 2 × 32 × 32 × 32: T1 (FreeSurfer `norm.mgz`, z-scored inside the brain) and the mirror cube from the other hemisphere; samples 1 mm apart |
| Architecture | `SmallCNN3D` in `05_train_cnn.py`: 4 × [conv3³-BN-ReLU × 2, max-pool], 16-32-64-64 channels, global average pool, dropout 0.3, linear; **437,089 parameters** |
| Output | one logit per cube; `sigmoid`, averaged over the cube and its left-right flip, and over the 3 seeds |
| Training | 10 patients, 679 electrode sites (86 SOZ) plus mirror controls and 200 deep-structure cubes; weighted BCE, AdamW, lr 1e-3, batch 32, augmentation; final models trained 7 epochs (median best epoch of the folds) |
| Validation | leave-one-patient-out, 3 seeds per fold; then 20 unseen patients |
| Framework | PyTorch 2.x (trained on a Google Colab T4 GPU; runs on a CPU) |

## Intended use

- Research on MRI-based localization in **drug-resistant focal epilepsy**, on adult T1 MRIs processed with FreeSurfer
  `recon-all`.
- Outputs: a probability map over the cortex, hippocampus and amygdala; the side; ranked regions.

**Not for:** clinical decisions, generalized epilepsy, deep lesions (hypothalamus, heterotopia), children (all
training patients are adults), or segmentations other than FreeSurfer `aparc+aseg`.

## Performance (details in [docs/RESULTS.md](../docs/RESULTS.md))

| Test | Result |
|---|---|
| Side, 15 unseen patients with a one-sided clinical record | 12 / 15 = 80% (95% CI 55–93%), AUROC 0.84 |
| Side, 10 MRI-only unseen patients | 10 / 10 at a 6 mm grid, 9 / 10 at 8 mm |
| SOZ sites within patients (leave-one-patient-out) | AUROC 0.73 (0.62–0.83); anatomy alone 0.64 |
| Location inside the hemisphere | not patient-specific yet (see [limitations](../docs/LIMITATIONS.md)) |

## Load the model in Python

Run this from the project folder (`05_train_cnn.py` imports `common` and `config` from there):

```python
import importlib.util, json, sys, torch

sys.path.insert(0, ".")                                                   # the project folder
spec = importlib.util.spec_from_file_location("cnn", "05_train_cnn.py")
cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn)

card = json.load(open("models/v9/cnn_model.json"))
nets = []
for name in card["final"]["models"]:
    net = cnn.SmallCNN3D(len(card["channels"]), base=card["base"], dropout=card["dropout"])
    net.load_state_dict(torch.load(f"models/v9/{name}", map_location="cpu"))
    nets.append(net.eval())
# x: float tensor (batch, 2, 32, 32, 32) made as in 08_predict_mri.py (common.standardise + common.pair_patches)
# with torch.no_grad():
#     p = torch.stack([cnn.flip_average(net, x) for net in nets]).mean(0)   # probability per cube
```

For whole brains, use `08_predict_mri.py`. It does the standardisation, the mirror plane, the grid and the map.

## Checksums (SHA-256)

<!-- checksums:start -->
| File | Size | SHA-256 |
|---|---|---|
| `cnn_final_s0.pt` | 1.77 MB | `7f37b201eaee0d4f85665c29017eeb17f1e925ec0bac8b3a512ddbc680744809` |
| `cnn_final_s1.pt` | 1.77 MB | `988da3d169fb5953282950866a62ea78099b5c670a59cad29fed8fdc78610747` |
| `cnn_final_s2.pt` | 1.77 MB | `385ac5b4603831b38daf57dd83ece8821d59caad8d5ad1dd4ae7af72db4381fd` |
| `cnn_model.json` | 2.6 kB | `54004d9ded8ca01020bc7d2e7d317beedda218b1414f32097f18a742bd3ab765` |
| `lateralization_predictions.csv` | 59.2 kB | `36a7069c68020aba9a20c2a0226f34f722261e6681d6806ada32fd1da3f1be18` |
| `side_model.json` | 1.0 kB | `6c3ee8d74dea03a4b4c4afa2b259824ec41abe7beb34e47d8ef28309893b4744` |
<!-- checksums:end -->

## Licence and citation

The weights are derived from the Penn 3T-7T Paired Epilepsy Imaging dataset (Lucas et al., Pennsieve Discover,
doi:10.26275/aosz-hj9u). They are shared for **non-commercial research**. Please cite the dataset and this repository
(see [CITATION.cff](../CITATION.cff)). The code is MIT-licensed ([LICENSE](../LICENSE)).
