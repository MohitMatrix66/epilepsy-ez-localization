# Results

The outputs of the published run, version 9, on the real Penn 3T-7T dataset. No MRI and no recordings are included,
only tables, reports and logs. Local paths are replaced by `<project>`, `<data>` and `<home>`. The summary and its
interpretation are in [docs/RESULTS.md](../docs/RESULTS.md).

```
results/v9/
├── model_report.txt              tools/model_report.py: leave-one-patient-out test of the 10 training patients
├── unseen_validation.txt / .csv  08_predict_mri.py --unseen --no-maps --step 8: the 20 unseen patients
├── comparison_mri_only.txt       new_patient_results/run_new_patients.py (6 mm grid): the 15 MRI-only patients
├── labels/                       step2_report.txt, step3_report.txt: the iEEG seizure-onset labels
├── tables/
│   ├── cnn_metrics.csv           CNN metrics (sites, regions, left-right pairs, deep-structure cubes)
│   ├── baseline_metrics.csv      anatomy prior, logistic regression, random forest
│   ├── lateralization_metrics.csv    side model and baselines (leave-one-patient-out, 25 patients)
│   ├── cnn_predictions_contacts.csv  out-of-fold CNN probability of every training site
│   ├── cnn_predictions_controls.csv  mirror controls (the healthy side of each SOZ site)
│   ├── cnn_predictions_deep.csv      deep-structure "not SOZ" cubes
│   ├── cnn_predictions_regions.csv   region scores (mean of their sites)
│   └── baseline_predictions.csv      region features and baseline probabilities
└── logs/                         what every step printed (05_train_cnn.txt = the Colab training log)
```

To make these files again, follow [docs/REPRODUCE.md](../docs/REPRODUCE.md). Your own runs are written to `outputs/`,
which Git ignores.
