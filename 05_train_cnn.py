"""
05_train_cnn.py  -  PHASE 6b: the deep-learning model (a small 3D convolutional neural network).

Input : outputs/patches.npz (from 04_make_patches.py; version 8 = each cube together with its mirror cube from the
        other side of the brain, plus the mirror controls; version 8.1 = cubes of the same millimetres in every patient,
        whatever the voxel size of his MRI - an older patches.npz is refused: run 04_make_patches.py again)
Task  : for each MRI cube around a contact, predict the probability that the contact is in the seizure onset zone.
Test  : cross-validation grouped by PATIENT (a patient's contacts are never split between training and test).
        --lopo = leave-one-patient-out (train on all patients but one, test on that one, repeat for every patient).
        Inside each training fold, ~20% of the training patients are held out as a validation set for early stopping
        (on the 3-epoch moving average of their AUROC, after at least --min-epochs epochs: one lucky epoch on a few
        validation patients should not choose the model).
        --seeds N trains N models per fold with different random starts and averages them (less luck).
Mirror controls (version 8) are training examples ("the same place on the healthy side = not SOZ") only where
04_make_patches.py allowed it (train_ok); in the TEST patient every control is predicted for the left-right test:
is the SOZ site scored above its mirror?
--final also trains the FINAL models on ALL patients (the number of epochs = the median best epoch of the
cross-validation), for 08_predict_mri.py (new MRI -> probability map).
Version 9: the deep-structure cubes of 04_make_patches.py ('not SOZ') are training examples too (and are predicted in
the test patient: cnn_predictions_deep.csv - they should now score LOW), and every prediction is the average over the
cube and its left-right flipped copy (removes a lean to one side that comes only from the cube's orientation).

Outputs:
  outputs/cnn_predictions_contacts.csv   out-of-fold probability for every contact site
  outputs/cnn_predictions_controls.csv   the same for every mirror control, next to its SOZ site (left-right test)
  outputs/cnn_predictions_regions.csv    site probabilities pooled per region (MEAN - the maximum would favour
                                         regions with many electrodes)
  outputs/cnn_metrics.csv
  outputs/cnn_fold{k}_s{seed}.pt         trained weights of each fold and seed; cnn_final_s{seed}.pt with --final
  outputs/cnn_model.json                 which weights exist, how the cubes were made (read by 08_predict_mri.py)
  outputs/cnn_folds/fold{k}.npz          test predictions of each fold (used by --resume)

Run:   python 05_train_cnn.py --lopo --resume --final        (recommended; on Google Colab with a GPU, see colab/)
--resume skips folds that already finished with exactly the same data and settings (useful when Colab disconnects).
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

import common
import config
import runlog

MODEL_VERSION = "9"


# --------------------------------------------------------------------------- model
class SmallCNN3D(nn.Module):
    """Four blocks of (conv -> batchnorm -> ReLU) x2 -> max-pool, then a tiny classifier.
    32^3 input -> 16^3 -> 8^3 -> 4^3 -> 2^3 -> one number (logit)."""

    def __init__(self, in_channels, base=16, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                nn.MaxPool3d(2))

        self.features = nn.Sequential(block(in_channels, base), block(base, base * 2),
                                      block(base * 2, base * 4), block(base * 4, base * 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(base * 4, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


# --------------------------------------------------------------------------- helpers
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augment(x):
    """Random flips, brightness/contrast jitter and noise - makes the model robust and multiplies the effective number
    of training examples. A cube and its mirror are flipped TOGETHER (same anatomy stays in the same place)."""
    b = x.shape[0]
    for dim in (2, 3, 4):
        flip = torch.rand(b, device=x.device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1, 1), x.flip(dim), x)
    x = x * (1 + 0.1 * torch.randn(b, 1, 1, 1, 1, device=x.device)) \
        + 0.1 * torch.randn(b, 1, 1, 1, 1, device=x.device)
    return x + 0.05 * torch.randn_like(x)


def flip_average(model, xb):
    """Probability averaged over the cubes and their left-right flipped copies (cube and mirror flipped together, as
    in training): the answer cannot depend on the cube's orientation (version 9)."""
    return (torch.sigmoid(model(xb)) + torch.sigmoid(model(xb.flip(2)))) / 2.0


@torch.no_grad()
def predict(model, X, device, batch=64):
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch].astype(np.float32)).to(device)
        out.append(flip_average(model, xb).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def split_validation(train_subjects, y_by_subject, rng, frac=0.2):
    """Hold out ~20% of training patients for early stopping; try to include a positive."""
    n_val = max(1, int(round(frac * len(train_subjects))))
    val = set()
    for _ in range(50):
        val = set(rng.choice(train_subjects, n_val, replace=False))
        if any(y_by_subject[s] > 0 for s in val):
            return val
    return val


def make_loss(y, device):
    pos = max(int(y.sum()), 1)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y) - pos) / pos], device=device))


def run_epoch(model, X, y, opt, loss_fn, args, device, rng):
    model.train()
    order = rng.permutation(len(X))
    losses = []
    for s in range(0, len(order), args.batch):
        idx = order[s:s + args.batch]
        xb = augment(torch.from_numpy(X[idx].astype(np.float32)).to(device))
        yb = torch.from_numpy(y[idx].astype(np.float32)).to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


def train_fold(Xtr, ytr, Xva, yva, args, device, seed):
    """Train with early stopping on the validation patients. Returns (model, best epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SmallCNN3D(Xtr.shape[1]).to(device)
    loss_fn = make_loss(ytr, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best_score, best_state, best_epoch, bad = [], -np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, Xtr, ytr, opt, loss_fn, args, device, rng)
        pva = predict(model, Xva, device)
        auc = common.safe_auroc(yva, pva)
        score = auc if np.isfinite(auc) else -float(
            nn.functional.binary_cross_entropy(torch.tensor(pva), torch.tensor(yva, dtype=torch.float32)))
        history.append(score)
        smooth = float(np.mean(history[-3:]))
        if epoch >= args.min_epochs:
            if smooth > best_score:
                best_score, best_epoch, bad = smooth, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        print(f"    epoch {epoch:3d}  train loss {loss:.4f}  val AUROC {auc:.3f}  (3-epoch mean {smooth:.3f})")
        if bad >= args.patience:
            print(f"    early stop - best epoch {best_epoch}")
            break
    if best_state is None:     # fewer epochs than --min-epochs: keep the last one
        best_state, best_epoch = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
    model.load_state_dict(best_state)
    return model, best_epoch


def train_fixed(X, y, epochs, args, device, seed):
    """Train on every example for a fixed number of epochs (the final model - no validation patients left)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SmallCNN3D(X.shape[1]).to(device)
    loss_fn = make_loss(y, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        loss = run_epoch(model, X, y, opt, loss_fn, args, device, rng)
        print(f"    epoch {epoch:3d}  train loss {loss:.4f}")
    return model


def pooled_side_accuracy(p_site, p_ctrl):
    ok = np.isfinite(p_site) & np.isfinite(p_ctrl)
    return float(np.mean(p_site[ok] > p_ctrl[ok])) if ok.any() else np.nan


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-epochs", type=int, default=3, dest="min_epochs")
    ap.add_argument("--seeds", type=int, default=3, help="models per fold (averaged)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lopo", action="store_true", help="leave-one-patient-out (one fold per patient)")
    ap.add_argument("--resume", action="store_true", help="skip folds that already finished")
    ap.add_argument("--final", action="store_true", help="also train the final models on all patients")
    args = ap.parse_args()

    data = np.load(os.path.join(config.OUT_DIR, "patches.npz"))
    X, y, subjects = data["X"], data["y"].astype(np.int64), data["subject"].astype(str)
    n = len(y)
    kind = data["kind"].astype(str) if "kind" in data.files else np.full(n, "site")
    train_ok = data["train_ok"].astype(bool) if "train_ok" in data.files else np.ones(n, bool)
    pair = data["pair"].astype(np.int64) if "pair" in data.files else np.full(n, -1)
    channels = [str(c) for c in data["channels"]] if "channels" in data.files else ["T1"]
    patch_version = str(data["version"]) if "version" in data.files else "6.1"
    if patch_version != "9" or "patch_mm" not in data.files:
        raise SystemExit(f"outputs/patches.npz is from version {patch_version}, not 9 (version 9: fluid kept dark, "
                         "deep-structure cubes). Run 04_make_patches.py again and upload the new patches.npz.")
    patch_mm = float(data["patch_mm"])
    is_site, is_ctrl, is_deep = kind == "site", kind == "control", kind == "deep"
    usable = is_site | ((is_ctrl | is_deep) & train_ok)                # rows that may be used for training
    device = pick_device()
    uniq = np.unique(subjects)
    print(f"Device: {device}   patches: {X.shape} ({X.shape[-1]} samples x {patch_mm:g} mm)   patients: {len(uniq)}   "
          f"sites: {int(is_site.sum())} (positives: {int(y[is_site].sum())})   mirror controls: {int(is_ctrl.sum())} "
          f"({int((is_ctrl & train_ok).sum())} for training)   deep-structure cubes: {int(is_deep.sum())}   "
          f"channels: {channels}")
    if device.type == "cpu":
        print("NOTE: no GPU found - this will be slow. Run it on Google Colab with a GPU (see the colab folder).")

    n_splits = len(uniq) if args.lopo else min(args.folds, len(uniq))
    sig = hashlib.md5()
    for part in (",".join(subjects.tolist()), y.tobytes(), ",".join(kind.tolist()), train_ok.tobytes(), X.tobytes(),
                 json.dumps([n_splits, args.epochs, args.batch, args.lr, args.patience, args.min_epochs, args.seeds,
                             MODEL_VERSION, patch_mm])):
        sig.update(part if isinstance(part, bytes) else part.encode())
    signature = sig.hexdigest()[:16]
    fold_dir = os.path.join(config.OUT_DIR, "cnn_folds")
    os.makedirs(fold_dir, exist_ok=True)

    y_by_subject = {s: y[(subjects == s) & is_site].sum() for s in uniq}
    oof = np.full(n, np.nan)
    folds_card, best_epochs = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=subjects), start=1):
        fold_file = os.path.join(fold_dir, f"fold{fold}.npz")
        test_subs = sorted(np.unique(subjects[te_idx]).tolist())
        models = [f"cnn_fold{fold}_s{s}.pt" for s in range(args.seeds)]
        if args.resume and os.path.exists(fold_file):
            saved = np.load(fold_file)
            if (str(saved["signature"]) == signature and np.array_equal(saved["te_idx"], te_idx)
                    and all(os.path.exists(os.path.join(config.OUT_DIR, m)) for m in models)):
                oof[te_idx] = saved["pred"]
                best_epochs += [int(e) for e in saved["best_epochs"]]
                folds_card.append({"fold": fold, "test_subjects": test_subs, "models": models})
                print(f"Fold {fold}/{n_splits}: already done (test patients: {', '.join(test_subs)})")
                continue
        rng = np.random.default_rng(config.SEED + fold)
        tr_subjects = np.unique(subjects[tr_idx])
        val_subjects = split_validation(tr_subjects, y_by_subject, rng)
        is_val = np.isin(subjects[tr_idx], list(val_subjects))
        fit_idx = tr_idx[~is_val & usable[tr_idx]]
        val_idx = tr_idx[is_val & usable[tr_idx]]
        print(f"\nFold {fold}/{n_splits}: train {len(fit_idx)} | val {len(val_idx)} | test {len(te_idx)} cubes "
              f"(test patients: {', '.join(test_subs)}; validation: {', '.join(sorted(val_subjects))})")
        preds, fold_best = [], []
        for s in range(args.seeds):
            print(f"  model {s + 1} of {args.seeds}")
            model, best = train_fold(X[fit_idx], y[fit_idx], X[val_idx], y[val_idx], args, device,
                                     seed=config.SEED + 100 * s + fold)
            preds.append(predict(model, X[te_idx], device))
            fold_best.append(best)
            torch.save(model.state_dict(), os.path.join(config.OUT_DIR, models[s]))
        oof[te_idx] = np.mean(preds, axis=0)
        best_epochs += fold_best
        np.savez(fold_file, te_idx=te_idx, pred=oof[te_idx], best_epochs=np.array(fold_best), signature=signature)
        folds_card.append({"fold": fold, "test_subjects": test_subs, "models": models})

    # ------------------------------------------------------------------ tables and metrics
    sites = pd.DataFrame({"subject": subjects, "contact": data["contact"], "region_id": data["region_id"],
                          "target": y, "p_cnn": oof})[is_site].reset_index(drop=True)
    ctrl_idx = np.flatnonzero(is_ctrl)
    deep = pd.DataFrame({"subject": subjects, "contact": data["contact"], "region_id": data["region_id"],
                         "p_cnn": oof})[is_deep].reset_index(drop=True)
    deep["region_name"] = deep["region_id"].map(common.REGIONS)
    controls = pd.DataFrame({"subject": subjects[ctrl_idx], "contact": data["contact"][ctrl_idx],
                             "region_id_mirror": data["region_id"][ctrl_idx], "train_ok": train_ok[ctrl_idx],
                             "p_control": oof[ctrl_idx],
                             "p_site": [oof[pair[i]] if pair[i] >= 0 else np.nan for i in ctrl_idx]})
    regions = (sites[sites["region_id"] > 0]
               .groupby(["subject", "region_id"])
               .agg(target=("target", "max"), p_cnn=("p_cnn", "mean"))
               .reset_index())
    regions["region_name"] = regions["region_id"].map(common.REGIONS)
    table_path = os.path.join(config.OUT_DIR, "region_table.csv")
    if os.path.exists(table_path):   # region labels as in step 2 (a region with an 'uncertain' site has none)
        rt = pd.read_csv(table_path)
        labelled = rt.loc[rt["sampled"].astype(bool), ["subject", "region_id"]]
        regions = regions.merge(labelled, on=["subject", "region_id"], how="inner")

    rows = [{"level": "contact", "model": "cnn", **common.summarize(sites, "p_cnn")},
            {"level": "region (mean of its sites)", "model": "cnn", **common.summarize(regions, "p_cnn")}]
    base_path = os.path.join(config.OUT_DIR, "baseline_predictions.csv")
    if os.path.exists(base_path):  # compare with Phase 5 on exactly the same regions
        base = pd.read_csv(base_path)
        both = regions.merge(base[["subject", "region_id", "p_region_prior", "p_random_forest"]],
                             on=["subject", "region_id"], how="inner")
        for col in ["p_region_prior", "p_random_forest", "p_cnn"]:
            rows.append({"level": "region (shared)", "model": col[2:], **common.summarize(both, col)})
    metrics = pd.DataFrame(rows)
    side = pooled_side_accuracy(controls["p_site"].to_numpy(float), controls["p_control"].to_numpy(float))
    print("\nOut-of-fold results:")
    print(metrics.to_string(index=False))
    if len(controls):
        print(f"\nLeft-right test: the SOZ site scored above the same place on the other side in {side:.0%} of "
              f"{int(controls[['p_site', 'p_control']].notna().all(axis=1).sum())} pairs (50% = cannot tell the side)."
              "  Per patient: python tools\\model_report.py")
        metrics = pd.concat([metrics, pd.DataFrame([{"level": "left-right pairs", "model": "cnn",
                                                     "n": len(controls), "side_accuracy": round(side, 3)}])],
                            ignore_index=True)
    if len(deep):
        soz_p, non_p = sites.loc[sites["target"] == 1, "p_cnn"].mean(), sites.loc[sites["target"] == 0, "p_cnn"].mean()
        print(f"\nDeep structures (version 9 check): mean probability of the deep-structure cubes {deep['p_cnn'].mean():.2f} "
              f"vs SOZ sites {soz_p:.2f} and other sites {non_p:.2f} (v8.1: deep 0.74, SOZ 0.57, other 0.43).")
        metrics = pd.concat([metrics, pd.DataFrame([{"level": "deep-structure cubes (not SOZ)", "model": "cnn",
                                                     "n": len(deep), "mean_p": round(float(deep["p_cnn"].mean()), 3),
                                                     "mean_p_soz_sites": round(float(soz_p), 3),
                                                     "mean_p_other_sites": round(float(non_p), 3)}])],
                            ignore_index=True)

    sites.to_csv(os.path.join(config.OUT_DIR, "cnn_predictions_contacts.csv"), index=False)
    controls.to_csv(os.path.join(config.OUT_DIR, "cnn_predictions_controls.csv"), index=False)
    deep.to_csv(os.path.join(config.OUT_DIR, "cnn_predictions_deep.csv"), index=False)
    regions.to_csv(os.path.join(config.OUT_DIR, "cnn_predictions_regions.csv"), index=False)
    metrics.to_csv(os.path.join(config.OUT_DIR, "cnn_metrics.csv"), index=False)

    # ------------------------------------------------------------------ final models + model card
    card_path = os.path.join(config.OUT_DIR, "cnn_model.json")
    final, old = None, {}
    if os.path.exists(card_path):
        try:
            with open(card_path) as fh:
                old = json.load(fh)
        except (ValueError, OSError):
            old = {}
    old_final = old.get("final") if old.get("signature") == signature else None
    if old_final and not all(os.path.exists(os.path.join(config.OUT_DIR, m)) for m in old_final.get("models", [])):
        old_final = None
    if args.final and args.resume and old_final:
        final = old_final                     # the same data and settings: the final models are already there
        print(f"\nFinal models already trained with the same data and settings: {', '.join(final['models'])}")
    elif args.final:
        epochs = int(max(args.min_epochs, np.median(best_epochs))) if best_epochs else args.epochs
        final_models = [f"cnn_final_s{s}.pt" for s in range(args.seeds)]
        print(f"\nFinal models: all {len(uniq)} patients, {int(usable.sum())} cubes, {epochs} epochs (the median best "
              f"epoch of the cross-validation), {args.seeds} model(s)")
        for s in range(args.seeds):
            print(f"  final model {s + 1} of {args.seeds}")
            model = train_fixed(X[usable], y[usable], epochs, args, device, seed=config.SEED + 7 + 100 * s)
            torch.save(model.state_dict(), os.path.join(config.OUT_DIR, final_models[s]))
        final = {"subjects": uniq.tolist(), "epochs": epochs, "models": final_models}
    else:                                     # keep final models made by an earlier run with the same data
        final = old_final
    card = {"version": MODEL_VERSION, "patch_version": patch_version, "channels": channels,
            "patch_size": int(X.shape[-1]), "patch_mm": patch_mm,
            "base": 16, "dropout": 0.3, "mirror": any("mirror" in c for c in channels),
            "signature": signature, "settings": {k: getattr(args, k) for k in
                                                 ("epochs", "batch", "lr", "patience", "min_epochs", "seeds")},
            "lopo": bool(args.lopo), "folds": folds_card, "final": final,
            "training_subjects": uniq.tolist(), "best_epochs": best_epochs,
            "left_right_test": None if not np.isfinite(side) else round(side, 3)}
    with open(card_path, "w") as fh:
        json.dump(card, fh, indent=1)
    print(f"\nSaved CNN predictions, metrics, weights and cnn_model.json to {config.OUT_DIR}/")


if __name__ == "__main__":
    with runlog.Log("05_train_cnn"):
        main()
