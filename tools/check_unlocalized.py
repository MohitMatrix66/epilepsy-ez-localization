"""
tools/check_unlocalized.py  -  do a patient's seizures start on recorded contacts that have NO coordinates?

Step 2 (SOZ/02_compute_soz.py) uses only contacts with electrode coordinates, because only those can be put
on the MRI. If the seizures really start on a recorded contact WITHOUT coordinates, the "SOZ" moves to the contacts
the seizure reaches next. This happened to sub-PID08 before the LA/LH name fix (its SOZ was put on the lateral
temporal contacts; with the hippocampus contacts it is the hippocampus) and to sub-PID02 before the DHAA/DHT name
fix (17 of its 18 seizures start there).
This check repeats the onset analysis of step 2 (the method chosen by the expert benchmark) WITH the recorded
contacts that have no coordinates and says whether the patient's SOZ labels can be trusted:
    OK        the contacts without coordinates do not start the seizures - the labels can be used
    DOUBTFUL  they do - leave the patient out of training (EXCLUDE_FROM_TRAINING in config.py)

Run from the project folder:
    python tools\\check_unlocalized.py --all                  every analysed patient, one line each (~1 min)
    python tools\\check_unlocalized.py --subject sub-PID02    one patient, every seizure shown
Uses the recordings kept in CACHE_DIR (no download when KEEP_EDF = True). Nothing is changed in the results of
steps 1-3; it adds unlocalized_check.csv (one row per patient) and check_unlocalized_<subject>.csv (start and onset
index of every site in every seizure) to SOZ/results.
"""
import argparse
import importlib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "SOZ"))
import pandas as pd  # noqa: E402

import aws_data  # noqa: E402
import config  # noqa: E402
import runlog  # noqa: E402
import soz_lib  # noqa: E402

step2 = importlib.import_module("02_compute_soz")


def electrode(norm):
    s = soz_lib.split_numbered(norm)
    return s[0] if s else None


def recording_files(r):
    edf = aws_data.fetch(r.edf)
    channels = soz_lib.read_tsv(aws_data.fetch(r.channels)) if r.channels else None
    return edf, channels


def unlocalized_contacts(recs, coord_names, aliases):
    """Recorded intracranial contacts (numbered, not marked bad) whose names have no coordinates, and the number
    of recordings that could be read. Reads only the recordings' headers, so it is fast."""
    import mne
    out, n_read = set(), 0
    for r in recs.itertuples():
        try:
            edf, channels = recording_files(r)
            names = mne.io.read_raw_edf(edf, preload=False, verbose="error").ch_names
            _, _, diag = soz_lib.select_channels(names, channels, coord_names, aliases)
            out |= {n for n in diag["unmatched"] if electrode(n)}   # numbered contacts only (not LOC / ROC)
            n_read += 1
        except Exception as e:  # noqa: BLE001
            print(f"    ({r.task} run-{r.run} could not be read: {type(e).__name__}: {e})")
    return out, n_read


def seizure_tables(recs, coord_names, coords, aliases, unloc, choice, verbose):
    """Per-seizure site tables of the chosen method (step 2), with the contacts without coordinates included."""
    frames, info = [], []
    m = choice["montage"]
    for i, r in enumerate(recs.itertuples(), start=1):
        name = f"{r.task}_run-{r.run}"
        try:
            edf, channels = recording_files(r)
            events = soz_lib.read_tsv(aws_data.fetch(r.events)) if r.events else None
            sidecar = soz_lib.read_json(aws_data.fetch(r.json)) if r.json else {}
            arr = soz_lib.analyse_recording(edf, channels, events, sidecar, coord_names | unloc, coords, aliases,
                                            config.EI_PRE_S, config.EI_POST_S)
            with tempfile.TemporaryDirectory() as tmp:   # same rounding as the saved results of step 2
                soz_lib.save_arrays(os.path.join(tmp, "a.npz"), arr)
                arr = soz_lib.load_arrays(os.path.join(tmp, "a.npz"))
            sites, seiz = step2.seizure_tables(arr, choice)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [{i}/{len(recs)}] {name}: skipped - {type(e).__name__}: {e}")
            continue
        finally:
            if not config.KEEP_EDF:
                aws_data.remove_downloaded(r.edf)
        sites = sites[sites["montage"] == m].copy()
        sites.insert(0, "seizure", name)
        sites["no_coords"] = [any(c in unloc for c in str(n).split("-")) for n in sites["contact_norm"]]
        frames.append(sites)
        s = seiz[seiz["montage"] == m].iloc[0]
        info.append({"seizure": name, "usable": bool(s["usable"]), "flags": s["flags"]})
        if verbose:
            on = sites[sites["onset_site"]].sort_values("start_s").head(6)
            shown = ", ".join(f"{c}{'*' if u else ''}({t:+.1f}s)" for c, u, t in
                              zip(on["contact"], on["no_coords"], on["start_s"])) or "none"
            print(f"  [{i}/{len(recs)}] {name}: " + (f"NOT USABLE ({s['flags']})" if s["flags"] else shown))
    if not frames:
        return None, None
    return pd.concat(frames, ignore_index=True), pd.DataFrame(info)


def judge(per, info, step2_soz, choice):
    """The SOZ rule of step 2 applied to every recorded contact, and the verdict."""
    usable = list(info.loc[info["usable"], "seizure"])
    n_use = len(usable)
    if n_use == 0:
        return {"seizures": 0, "led_unloc": 0, "soz": pd.DataFrame(columns=["contact", "no_coords"]),
                "result": "OK", "lost": [], "gained": []}
    lab, _ = step2.patient_labels(per, usable, choice)
    lab = lab.set_index("contact_norm")
    lab["no_coords"] = lab.index.map(per.groupby("contact_norm")["no_coords"].first()).fillna(False).astype(bool)
    soz = lab[lab["soz"] == 1].sort_values("n_onset", ascending=False)
    p = per[per["seizure"].isin(usable) & per["onset_site"]]
    first = p.loc[p.groupby("seizure")["start_s"].idxmin()] if len(p) else p
    led_unloc = int(first["no_coords"].sum())
    doubtful = bool(soz["no_coords"].any()) or led_unloc / n_use >= config.SOZ_MIN_LED
    with_coords = set(soz.index[~soz["no_coords"].astype(bool)])
    unloc_sites = set(per.loc[per["no_coords"].astype(bool), "contact_norm"])
    shown = dict(zip(per["contact_norm"], per["contact"]))
    lost = [shown.get(n, n) for n in sorted(set(step2_soz) - with_coords - unloc_sites)]
    gained = [shown.get(n, n) for n in sorted(with_coords - set(step2_soz))]
    return {"seizures": n_use, "led_unloc": led_unloc, "soz": soz, "result": "DOUBTFUL" if doubtful else "OK",
            "lost": lost, "gained": gained}


def names(soz):
    return ", ".join(f"{c}{'*' if u else ''}" for c, u in zip(soz["contact"], soz["no_coords"])) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None, help="one patient, e.g. sub-PID02 (every seizure is shown)")
    ap.add_argument("--all", action="store_true", help="every analysed patient, one line each")
    ap.add_argument("--hide", nargs="*", default=[],
                    help="(for testing) treat these electrodes as if they had no coordinates, e.g. --hide RA")
    args = ap.parse_args()
    if not args.all and not args.subject:
        raise SystemExit("Give --all or --subject, e.g.   python tools\\check_unlocalized.py --all")
    choice = step2.load_choice()
    out = config.SOZ_RESULTS_DIR
    inv = pd.read_csv(os.path.join(out, "dataset_inventory.csv"), keep_default_na=False)
    recs_all = pd.read_csv(os.path.join(out, "seizure_recordings.csv"), keep_default_na=False, dtype={"run": str})
    recs_all = recs_all[recs_all["kind"] == "ictal"]
    lab_path = os.path.join(out, "soz_labels.csv")
    labels = (pd.read_csv(lab_path) if os.path.exists(lab_path)
              else pd.DataFrame(columns=["subject", "contact", "contact_norm", "soz"]))
    excluded = getattr(config, "EXCLUDE_FROM_TRAINING", None) or {}
    summ_path = os.path.join(out, "soz_summary.csv")
    soz_summ = pd.read_csv(summ_path).set_index("subject") if os.path.exists(summ_path) else pd.DataFrame()

    def left_out(sub):
        """Why the patient is already not used for training ('' = it is used)."""
        if sub in excluded:
            return "already in EXCLUDE_FROM_TRAINING"
        if sub in soz_summ.index and not bool(soz_summ.loc[sub, "use_for_training"]):
            return f"already left out of training: label quality '{soz_summ.loc[sub, 'label_quality']}'"
        return ""

    if args.all:
        subjects = sorted(set(labels["subject"]) & set(recs_all["subject"]))
        print(f"Checking {len(subjects)} patients for recorded contacts without coordinates ...")
    else:
        subjects = [args.subject]
    hide = {h.upper() for h in args.hide}

    rows = []
    for sub in subjects:
        inv_rows = list(inv[inv["subject"] == sub].itertuples())
        recs = recs_all[recs_all["subject"] == sub]
        if not inv_rows or recs.empty:
            raise SystemExit(f"{sub}: no seizure recordings in {out} (run SOZ\\01 and SOZ\\02 first).")
        coord_names, coords = step2.load_coordinates(inv_rows[0])
        if hide:
            coord_names = {n for n in coord_names if electrode(n) not in hide}
            coords = {n: xyz for n, xyz in coords.items() if electrode(n) not in hide}
        aliases = step2.aliases_for(sub)
        unloc, n_read = unlocalized_contacts(recs, coord_names, aliases)
        row = {"subject": sub, "contacts_without_coords": " ".join(sorted(unloc)), "already_left_out": left_out(sub)}
        if n_read == 0:
            print(f"  {sub}: none of its recordings could be read - not checked")
            rows.append({**row, "result": "not checked"})
            continue
        if not unloc:
            print(f"  {sub}: every recorded contact has coordinates - nothing to check" if args.all else
                  f"{sub}: every recorded intracranial contact has coordinates - nothing to check.")
            rows.append({**row, "result": "nothing to check"})
            continue
        if not args.all:
            print(f"{sub}: {len(recs)} seizure recordings, {len(coord_names)} contacts with coordinates, "
                  f"{len(unloc)} recorded contacts without")
        per, sinfo = seizure_tables(recs, coord_names, coords, aliases, unloc, choice, verbose=not args.all)
        if per is None:
            print(f"  {sub}: no seizure could be analysed")
            rows.append({**row, "result": "no seizure analysed"})
            continue
        per.to_csv(os.path.join(out, f"check_unlocalized_{sub}.csv"), index=False)
        step2_soz = list(labels.loc[(labels["subject"] == sub) & (labels["soz"] == 1), "contact_norm"])
        j = judge(per, sinfo, step2_soz, choice)
        rows.append({**row, "seizures": j["seizures"], "strongest_site_without_coords": j["led_unloc"],
                     "soz_with_every_contact": names(j["soz"]), "soz_lost": " ".join(j["lost"]),
                     "soz_gained": " ".join(j["gained"]), "result": j["result"]})
        note = f"  ({left_out(sub)})" if left_out(sub) else ""
        if args.all:
            change = ("SOZ of step 2 unchanged" if not (j["lost"] or j["gained"]) else
                      f"SOZ of step 2 changes: lost {' '.join(j['lost']) or '-'}, gained {' '.join(j['gained']) or '-'}")
            print(f"  {sub}: {len(unloc)} contacts without coordinates ({' '.join(sorted(unloc)[:6])}"
                  f"{' ...' if len(unloc) > 6 else ''}); earliest onset site without coordinates in {j['led_unloc']} "
                  f"of {j['seizures']} usable seizures; {change} -> {j['result']}{note}")
        else:
            step2_names = labels.loc[(labels["subject"] == sub) & (labels["soz"] == 1), "contact"]
            print(f"\nContacts without coordinates (* above): {' '.join(sorted(unloc))}")
            print(f"SOZ from step 2 (contacts with coordinates only):  {', '.join(step2_names) or 'none'}")
            print(f"SOZ with every recorded contact (same rule):        {names(j['soz'])}")
            print(f"Usable seizures whose earliest onset site has no coordinates: {j['led_unloc']} of {j['seizures']}")
            if j["result"] == "DOUBTFUL":
                print(f"\nRESULT: DOUBTFUL - the seizures of {sub} start (partly) on contacts that cannot be put on "
                      "the MRI, so its SOZ labels would be wrong." + (note or
                      f"\nLeave it out of training: add \"{sub}\" to EXCLUDE_FROM_TRAINING in config.py, then run "
                      "SOZ\\03_build_contacts.py and the steps after it again."))
            else:
                print(f"\nRESULT: OK - the contacts without coordinates do not start the seizures of {sub}; its SOZ "
                      "labels can be used as they are.")

    summ = pd.DataFrame(rows)
    path = os.path.join(out, "unlocalized_check.csv")
    if os.path.exists(path) and not args.all:   # keep the rows of the other patients
        old = pd.read_csv(path)
        summ = pd.concat([old[~old["subject"].isin(summ["subject"])], summ], ignore_index=True)
    for c in ("seizures", "strongest_site_without_coords"):
        if c in summ.columns:
            summ[c] = summ[c].astype("Int64")
    summ.sort_values("subject").to_csv(path, index=False)
    if args.all:
        todo = [r["subject"] for r in rows if r["result"] == "DOUBTFUL" and not r["already_left_out"]]
        print("\nDOUBTFUL = its seizures start on contacts without coordinates, so its SOZ labels would be wrong.")
        if todo:
            print(f"Still used for training: {', '.join(todo)} -> add them to EXCLUDE_FROM_TRAINING in config.py, "
                  "then run SOZ\\03_build_contacts.py and the steps after it again.")
        else:
            print("No DOUBTFUL patient is used for training - nothing to change.")
        unchecked = [r["subject"] for r in rows if r["result"] in ("not checked", "no seizure analysed")]
        if unchecked:
            print(f"Could not be checked: {', '.join(unchecked)} (see the messages above).")
    print(f"Saved unlocalized_check.csv in {out}")


if __name__ == "__main__":
    with runlog.Log("check_unlocalized"):
        main()
