"""
SOZ/01_scan_dataset.py  -  STEP 1: which patients can be used, and what do their files look like?

Run from the project folder:     python SOZ\\01_scan_dataset.py
Downloads only small files (tables and coordinate lists, about 20-30 MB). Takes 1-3 minutes.

What it does:
  1. Reads the map of the whole dataset on AWS (all 30 patients).
  2. Decides which patients are usable (they need FreeSurfer 3T + seizure recordings + electrode
     coordinates) and which are left out, and why.
  3. Downloads the small tables of every seizure recording (channels.tsv, events.tsv, ieeg.json),
     the electrode coordinate files and the two metadata spreadsheets.
  4. Checks whether the channel tables already contain clinician SOZ marks.
  5. Writes a format report with real examples of every file type.

Results (in SOZ/results/):
  dataset_inventory.csv     one row per patient: what exists, usable or not, why
  seizure_recordings.csv    one row per seizure recording: files, size, seizure onset time found
  clinical_soz_marks.csv    contacts already marked as SOZ in the tables (only if any exist)
  format_report.txt         examples of each file + summary  <- send this to Claude
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

import aws_data  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402


def head(path, n=8):
    if not path or not os.path.exists(path):
        return "    (file not available)"
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [next(fh, "") for _ in range(n)]
    return "\n".join("    " + ln.rstrip("\n")[:160] for ln in lines if ln)


def main():
    out = config.SOZ_RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    rep = []

    def say(s=""):
        print(s)
        rep.append(s)

    inv, rec = aws_data.inventory()
    inv.to_csv(os.path.join(out, "dataset_inventory.csv"), index=False)
    usable = inv[inv["usable"]]["subject"].tolist()

    say("=" * 78)
    say("DATASET INVENTORY")
    say("=" * 78)
    show = inv[["subject", f"fs{config.FIELD}_aparc", f"fs{config.FIELD}_T2", "n_seizure_recordings",
                "has_electrode_coords", "recon_t1", "usable", "reason"]].copy()
    show["recon_t1"] = show["recon_t1"].map(lambda p: "yes" if p else "-")
    show.columns = ["subject", f"FreeSurfer{config.FIELD} labels", "T2", "seizures", "electrode coords",
                    "reference MRI", "usable", "reason"]
    say(show.to_string(index=False))
    no_t2 = inv.loc[inv["usable"] & ~inv[f"fs{config.FIELD}_T2"].astype(bool), "subject"].tolist()
    if no_t2:
        say(f"\nNo T2 image for: {', '.join(no_t2)}  -> the model will use the T1 image only "
            f"(T2 is used only when every patient has it).")
    say("")
    say(f"Usable patients: {len(usable)} of {len(inv)}  ->  {', '.join(usable) if usable else 'none'}")
    say("Left out: " + ("; ".join(f"{r.subject} ({r.reason})" for r in inv[~inv['usable']].itertuples()) or "none"))
    if not usable:
        with open(os.path.join(out, "format_report.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(rep))
        raise SystemExit("\nNo usable patients - see the reasons above.")

    seiz = rec[(rec["kind"] == "ictal") & rec["subject"].isin(usable) & (rec["edf"] != "")].copy()
    small = []
    for c in ("channels", "events", "json"):
        small += [r for r in seiz[c] if r]
    for c in ("recon_names", "recon_coords_mm", "recon_coords_vox"):
        small += [r for r in inv.loc[inv["usable"], c] if r]
    top_files, _ = aws_data.listing("")
    meta = [f for f, _ in top_files if f.lower().endswith((".xlsx", ".xls", ".csv", ".tsv", ".json"))]
    src_files, _ = aws_data.listing("sourcedata/")
    meta += [f for f, _ in src_files if f.lower().endswith((".json", ".tsv")) or "readme" in f.lower()]
    print(f"\nDownloading {len(small) + len(meta)} small files (tables, coordinates, metadata) ...")
    paths = aws_data.fetch_many(small + meta)

    # ---- read every seizure recording's tables
    onsets, sources, n_ch, n_bad, n_marked = [], [], [], [], []
    marks_rows, event_labels = [], Counter()
    channel_values = {}
    for r in seiz.itertuples():
        ch = soz_lib.read_tsv(paths[r.channels]) if r.channels and paths.get(r.channels) else None
        ev = soz_lib.read_tsv(paths[r.events]) if r.events and paths.get(r.events) else None
        onset, src = soz_lib.find_onset(ev)
        onsets.append(onset)
        sources.append(src)
        if ev is not None:
            text_cols = [c for c in ev.columns if c not in ("onset", "duration", "sample")]
            for _, e in ev.iterrows():
                event_labels[" | ".join(f"{c}={e[c]}" for c in text_cols)] += 1
        if ch is not None:
            n_ch.append(len(ch))
            n_bad.append(int((ch.get("status", pd.Series(dtype=str)).str.lower() == "bad").sum()))
            marks = soz_lib.clinical_soz_marks(ch)
            n_marked.append(len(marks))
            for name, where in marks.items():
                marks_rows.append({"subject": r.subject, "contact_norm": name, "recording": f"{r.task}_run-{r.run}",
                                   "where": where})
            for c in ch.columns:
                if c != "name":
                    channel_values.setdefault(c, Counter()).update(ch[c].astype(str))
        else:
            n_ch.append(0)
            n_bad.append(0)
            n_marked.append(0)
    seiz["onset_s"] = onsets
    seiz["onset_source"] = sources
    seiz["n_channels"] = n_ch
    seiz["n_bad_channels"] = n_bad
    seiz["n_clinical_soz_marks"] = n_marked
    seiz["edf_MB"] = (seiz["edf_bytes"] / 1e6).round(1)
    seiz.to_csv(os.path.join(out, "seizure_recordings.csv"), index=False)

    marks_path = os.path.join(out, "clinical_soz_marks.csv")
    if marks_rows:
        pd.DataFrame(marks_rows).drop_duplicates(["subject", "contact_norm"]).to_csv(marks_path, index=False)
    elif os.path.exists(marks_path):
        os.remove(marks_path)

    # ---- report
    say("")
    say("=" * 78)
    say("SEIZURE RECORDINGS (usable patients)")
    say("=" * 78)
    per = seiz.groupby("subject").agg(seizures=("task", "count"), data_MB=("edf_MB", "sum"),
                                      onset_found=("onset_s", lambda s: int(s.notna().sum())),
                                      clinical_marks=("n_clinical_soz_marks", "max")).reset_index()
    say(per.to_string(index=False))
    say(f"\nTotal: {len(seiz)} seizure recordings, {seiz['edf_bytes'].sum() / 1e9:.2f} GB to download in step 2 "
        + ("(one at a time; kept in CACHE_DIR afterwards because KEEP_EDF = True)." if config.KEEP_EDF else
           "(one at a time, each deleted after use because KEEP_EDF = False)."))
    say(f"Seizure onset time found in events.tsv for {seiz['onset_s'].notna().sum()} of {len(seiz)} recordings.")
    say("\nEvent labels seen in events.tsv (count):")
    for label, n in event_labels.most_common(15):
        say(f"    {n:4d} x  {label}")

    say("")
    say("=" * 78)
    say("CLINICIAN SOZ MARKS IN THE CHANNEL TABLES")
    say("=" * 78)
    if marks_rows:
        m = pd.DataFrame(marks_rows).drop_duplicates(["subject", "contact_norm"])
        say(f"FOUND for {m['subject'].nunique()} patients ({len(m)} contacts) -> saved clinical_soz_marks.csv")
        say(m.groupby("subject")["contact_norm"].apply(lambda s: ", ".join(s)).to_string())
    else:
        say("None found. The SOZ will be computed from the seizure recordings (Epileptogenicity Index).")

    say("")
    say("=" * 78)
    say("FILE FORMAT EXAMPLES")
    say("=" * 78)
    ex = seiz.iloc[0]
    say(f"channels.tsv  ({ex.channels.split('/')[-1]}):")
    say(head(paths.get(ex.channels), 8))
    say("\nchannels.tsv columns and their most common values (all seizure recordings):")
    for c, cnt in channel_values.items():
        say(f"    {c}: " + ", ".join(f"{v!r} x{n}" for v, n in cnt.most_common(6)))
    for r in seiz.drop_duplicates("subject").head(3).itertuples():
        say(f"\nevents.tsv  ({r.events.split('/')[-1] if r.events else 'missing'}):")
        say(head(paths.get(r.events), 6))
    say(f"\nieeg.json  ({ex.json.split('/')[-1] if ex.json else 'missing'}):")
    say(head(paths.get(ex.json), 30))
    first = inv[inv["usable"]].iloc[0]
    for c in ("recon_names", "recon_coords_mm", "recon_coords_vox"):
        say(f"\n{c}  ({first[c].split('/')[-1] if first[c] else 'missing'}):")
        say(head(paths.get(first[c]), 5))

    say("")
    say("=" * 78)
    say("METADATA FILES")
    say("=" * 78)
    for f in meta:
        p = paths.get(f)
        say(f"\n{f}:")
        if not p:
            say("    (could not download)")
        elif f.lower().endswith((".xlsx", ".xls")):
            try:
                for sheet, df in pd.read_excel(p, sheet_name=None).items():
                    say(f"  [sheet '{sheet}'] {df.shape[0]} rows x {df.shape[1]} columns")
                    say("    " + df.to_string(max_rows=40, max_colwidth=30).replace("\n", "\n    "))
            except Exception as e:  # noqa: BLE001
                say(f"    (could not read: {e})")
        else:
            say(head(p, 25))

    with open(os.path.join(out, "format_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep))
    print(f"\nSaved in {out}: dataset_inventory.csv, seizure_recordings.csv, format_report.txt"
          + (", clinical_soz_marks.csv" if marks_rows else ""))
    print("Next: python SOZ\\02_compute_soz.py")


if __name__ == "__main__":
    with runlog.Log("01_scan_dataset"):
        main()
