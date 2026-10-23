EXPERT BENCHMARK - seizures whose onset was marked by epileptologists
=====================================================================

What this folder is
    The marks and small tables of the dataset "Multi-Expert Seizure Annotation (spontaneous and
    stimulation-induced seizures in SEEG)": stereotactic EEG seizures from the University of Pennsylvania (HUP,
    adults) and the Children's Hospital of Philadelphia (CHOP). Three to five board-certified epileptologists
    independently marked, for every seizure, the time and the channels of the unequivocal electrographic onset;
    the consensus is the majority of their marks.
    Version 6 of this project uses the 40 spontaneous adult (HUP) seizures of 19 patients to TEST which onset
    method agrees best with the epileptologists (tools/benchmark_onset.py). The recordings themselves are not in
    this folder; tools/get_benchmark.py downloads only the 63 seconds around each onset (about 0.8 GB).

Files
    seizures.csv            one row per spontaneous seizure used: recording, onset time, consensus onset channels
                            (ueo_consensus), each epileptologist's channels (ueo_each), low-voltage fast activity at
                            onset yes/no (lvfa), the file key on the public server
    excluded_seizures.csv   a seizure left out to be safe (annotated 'spontaneous' but its recording is named as a
                            stimulation-induced seizure)
    annotations.tsv/.json   the authors' full annotation table (all 78 seizures) and its column definitions
    participants.tsv        the authors' patient table
    sidecars/<patient>/     channels.tsv, events.tsv, ieeg.json of every used seizure and the electrode table
                            (electrodes.tsv; HUP: native-space mm coordinates)
    No file was changed; seizures.csv and the folder layout were made from the authors' files.

Source and licence
    NEMAR dataset nm000402 (doi:10.82901/nemar.nm000402), redistributing Pennsieve Discover dataset 530 version 3
    (doi:10.26275/dely-fzwx) by the Penn Center for Neuroengineering and Therapeutics.
    Licence: Creative Commons Attribution 4.0 (CC-BY-4.0).
    How to cite: the dataset (doi:10.26275/dely-fzwx) and the manuscript
        Ojemann W.K.S. et al. (2026), "Multi-expert consensus annotations of spontaneous and stimulation-induced
        seizures in stereotactic EEG", medRxiv, https://doi.org/10.64898/2026.01.15.26344025
    Recordings: https://github.com/nemarDatasets/nm000402 (public copy of the files on
    https://nemar.s3.us-east-2.amazonaws.com/nm000402/objects/).
