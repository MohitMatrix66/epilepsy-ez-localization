NEW PATIENT RESULTS - the final model (version 9) on the patients it was never trained on
==========================================================================================

This folder is an add-on. It does not change any file, folder or setting of the project: it only reads the trained
models (outputs\cnn_model.json, outputs\cnn_final_s*.pt, outputs\side_model.json) and writes everything into this
folder. The MRI files come from the public dataset on AWS, like in every other step (already downloaded ones are
reused from your data folder).

RUN (Anaconda Prompt or a terminal, in the project folder):

    conda activate ez-localization      (or your own environment)
    python new_patient_results\run_new_patients.py                 the 15 patients without iEEG (MRI only)
    python new_patient_results\run_new_patients.py --all-unseen    all 20 patients the model never saw
    python new_patient_results\run_new_patients.py --subjects sub-PID06 sub-PID28      only these

Needs: the version 9 model in outputs\ - either the published one (copy models\v9\* into outputs\) or your own
Colab results (outputs\cnn_*.pt, cnn_model.json) with 07_lateralization.py already run.
Time: about 1-3 minutes per patient on a laptop (--step 8 is faster, --step 5 gives finer maps).

RESULTS (in this folder):

    comparison_mri_only.txt     the table and the accuracy: model against the clinical record
    comparison_mri_only.csv     the same, one row per patient, with every column and the file paths
                                (comparison_all_unseen.* with --all-unseen, comparison_selected.* with --subjects)
    sub-PIDxx\          one folder per patient:
        sub-PIDxx_ez_probability.nii.gz   probability map
        sub-PIDxx_ez_zone.nii.gz          the model's zone (highest 2% of the assessed grey matter)
        sub-PIDxx_ez_probability.png      ready-made picture (three views)
        sub-PIDxx_regions.csv, sub-PIDxx_summary.txt
    run_log_*.txt               everything the script printed

WHAT IS COMPARED (metadata_3T_7T.xlsx of the dataset):
    side   model (CNN, and the side model of 07) vs 'Seizure Onset Lateralization' (Left only / Left > Right = left)
    lobe   lobe of the model's top region vs 'Seizure Onset Localization'
    MRI    model side vs the side of a one-sided lesion in the MRI report; for hippocampal sclerosis: is the sick
           hippocampus scored above the healthy one?  The 7T report is listed for reference.

VIEW A MAP (segmentation) IN ITK-SNAP (free) - or 3D Slicer / Freeview:
    1. File > Open Main Image: the patient's norm.mgz (its full path is in the column t1_to_view of comparison_*.csv,
       e.g. C:\Users\<you>\ez_data\files\derivatives\sub-PID06\freesurfer3T\sub-PID06\mri\norm.mgz).
    2. File > Add Another Image: sub-PID06_ez_probability.nii.gz -> as an overlay; colour map 'Hot'.
    3. Segmentation > Open Segmentation: sub-PID06_ez_zone.nii.gz (the zone, in red).

Research prototype trained on 10 patients - not a clinical tool.
