"""Stage 5 — development analyses on the patient-disjoint held-out set.

Reproduces the per-class breakdown, the complementarity test, the component
ablation and the fusion-weight sensitivity curve.

    python src/05_analysis.py --root ./brats2026

Heads are retrained on the patient-disjoint training portion only, under three
seeds, so the held-out patients are never seen. The resulting probability
matrices are cached; re-runs skip training entirely.

Outputs (under <root>/analysis/):
    per_class_f1.csv          per-class F1 for both heads and the ensemble
    table_ablation.csv        six configurations, mean +/- std over seeds
    complementarity.txt       agreement, McNemar, oracle accuracy
    weight_sweep.csv          macro-F1 vs fusion weight, mean and std
    cm_*.csv                  row-normalised confusion matrices
"""
import argparse, gc, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, recall_score)
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(__file__))
from common import (MLP, patient_split, softmax_probs, CLASSES,
                    CALIBRATION_MULTIPLIERS, ENSEMBLE_WEIGHT)

SEEDS = [7, 42, 123]
DIMS  = {"hoptimus": 1536, "virchow2": 2560}
LEAKAGE_THRESHOLD = 0.80


def metrics(y, p):
    return dict(F1=f1_score(y, p, average="macro"),
                Accuracy=accuracy_score(y, p),
                MCC=matthews_corrcoef(y, p),
                Sensitivity=recall_score(y, p, average="macro"))


def calibrated(P):
    Q = P * CALIBRATION_MULTIPLIERS
    return Q / Q.sum(1, keepdims=True)


def configurations(Ph, Pv):
    """The six ablation configurations.

    'raw' / 'cal' refer to the H-optimus-1 branch only; calibration is never
    applied to Virchow2 or to the fused probabilities.
    """
    Qh, w = calibrated(Ph), ENSEMBLE_WEIGHT
    return {
        "H-optimus-1, raw":        Ph.argmax(1),
        "H-optimus-1, cal":        Qh.argmax(1),
        "Virchow2":                Pv.argmax(1),
        "Ens. 0.50/0.50, cal":     (0.5 * Qh + 0.5 * Pv).argmax(1),
        "Ens. 0.55/0.45, raw":     (w * Ph + (1 - w) * Pv).argmax(1),
        "Ens. 0.55/0.45, cal":     (w * Qh + (1 - w) * Pv).argmax(1),
    }


def build_cache(root, cache, device):
    """Train both heads under every seed on the patient-disjoint split."""
    from importlib import import_module
    tr_mod = import_module("02_train_heads")

    store, y_ref = {}, None
    for backbone in ("hoptimus", "virchow2"):
        X, y, fn = tr_mod.load_embeddings(root, backbone)
        tr, va = patient_split(
            fn, y, f"{root}/data/train/BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv")
        for seed in SEEDS:
            print(f"training {backbone} seed={seed}")
            tmp = f"{root}/models/_tmp_{backbone}_{seed}.pt"
            tr_mod.train(X[tr], y[tr], DIMS[backbone], tmp, device=device)
            m = MLP(DIMS[backbone]).to(device)
            m.load_state_dict(torch.load(tmp, map_location=device))
            store[f"{backbone}_{seed}"] = softmax_probs(m, X[va], device)
            os.remove(tmp); del m; gc.collect(); torch.cuda.empty_cache()
        if y_ref is None:
            y_ref, fn_ref = y[va], fn[va]
        else:
            # align the two backbones by filename
            idx = {f: i for i, f in enumerate(fn[va])}
            order = [idx[f] for f in fn_ref]
            for s in SEEDS:
                store[f"{backbone}_{s}"] = store[f"{backbone}_{s}"][order]
        del X, y; gc.collect()

    np.savez_compressed(cache, y=y_ref, **store)
    print(f"cached -> {cache}")
    return store, y_ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./brats2026")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = f"{a.root}/analysis"; os.makedirs(out, exist_ok=True)
    cache = f"{a.root}/models/analysis_cache.npz"

    if os.path.exists(cache):
        z = np.load(cache)
        store = {k: z[k] for k in z.files if k != "y"}; y = z["y"]
        print(f"loaded cache ({len(y):,} held-out patches)")
    else:
        store, y = build_cache(a.root, cache, device)

    Ph0, Pv0 = store[f"hoptimus_{SEEDS[0]}"], store[f"virchow2_{SEEDS[0]}"]

    probe = f1_score(y, Ph0.argmax(1), average="macro")
    print(f"\n[guard] held-out macro-F1 of a freshly trained head: {probe:.4f}")
    if probe > LEAKAGE_THRESHOLD:
        raise SystemExit(f"ABORT: {probe:.3f} is impossibly high for a patient-disjoint "
                         "split (expected ~0.45). Leakage suspected.")

    # ---- ablation over seeds -------------------------------------------------
    rows = []
    for s in SEEDS:
        for name, p in configurations(store[f"hoptimus_{s}"], store[f"virchow2_{s}"]).items():
            r = metrics(y, p); r["Configuration"] = name; r["Seed"] = s; rows.append(r)
    df = pd.DataFrame(rows)
    M = ["F1", "Accuracy", "MCC", "Sensitivity"]
    agg = df.groupby("Configuration", sort=False)[M].agg(["mean", "std"]).round(4)
    agg.to_csv(f"{out}/table_ablation.csv")
    pretty = pd.DataFrame({m: agg[(m, "mean")].map("{:.3f}".format) + " ± "
                              + agg[(m, "std")].map("{:.3f}".format) for m in M})
    print("\n=== ablation (mean +/- std over seeds) ===")
    print(pretty.to_string())

    # ---- per-class and confusion matrices (seed 0) ---------------------------
    cfg = configurations(Ph0, Pv0)
    keep = {"H-optimus-1": cfg["H-optimus-1, cal"], "Virchow2": cfg["Virchow2"],
            "Ensemble": cfg["Ens. 0.55/0.45, raw"]}
    pd.DataFrame({k: f1_score(y, p, average=None, labels=range(10)).round(3)
                  for k, p in keep.items()}, index=CLASSES).to_csv(f"{out}/per_class_f1.csv")
    for k, p in keep.items():
        cm = confusion_matrix(y, p, labels=range(10)).astype(float)
        cm = cm / cm.sum(1, keepdims=True) * 100
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
            f"{out}/cm_{k.split()[0].lower()}.csv", float_format="%.6f")

    # ---- complementarity ----------------------------------------------------
    ph, pv = cfg["H-optimus-1, cal"], cfg["Virchow2"]
    ch, cv = (ph == y), (pv == y)
    nh, nv = int((ch & ~cv).sum()), int((~ch & cv).sum())
    stat = (abs(nh - nv) - 1) ** 2 / max(nh + nv, 1)
    txt = (f"patient-disjoint held-out, n={len(y):,}\n"
           f"agreement            {(ph == pv).mean()*100:.1f}%\n"
           f"both correct         {(ch & cv).mean()*100:.1f}%\n"
           f"both wrong           {(~ch & ~cv).mean()*100:.1f}%\n"
           f"only H-optimus-1     {nh/len(y)*100:.1f}%\n"
           f"only Virchow2        {nv/len(y)*100:.1f}%\n"
           f"exactly one          {(nh + nv)/len(y)*100:.1f}%\n"
           f"oracle accuracy      {(ch | cv).mean():.4f}"
           f"  (vs {ch.mean():.4f} / {cv.mean():.4f})\n"
           f"McNemar chi2={stat:.1f}, p={1 - chi2.cdf(stat, 1):.3e}\n")
    open(f"{out}/complementarity.txt", "w").write(txt)
    print("\n=== complementarity ===\n" + txt)

    # ---- fusion-weight sweep (raw, the submitted fusion) ---------------------
    ws = np.arange(0, 1.01, 0.05)
    sweep = np.array([[f1_score(y, (w * store[f"hoptimus_{s}"]
                                    + (1 - w) * store[f"virchow2_{s}"]).argmax(1),
                                average="macro") for w in ws] for s in SEEDS])
    mean, std = sweep.mean(0), sweep.std(0, ddof=1)
    pd.DataFrame({"weight": ws, "macro_F1_mean": mean, "macro_F1_std": std}
                 ).to_csv(f"{out}/weight_sweep.csv", index=False)
    i55, ib = int(np.argmin(abs(ws - 0.55))), int(np.argmax(mean))
    flat = ws[mean >= mean[ib] - std[ib]]
    print("=== weight sweep (raw fusion) ===")
    print(f"  w=0.55       {mean[i55]:.4f} +/- {std[i55]:.4f}")
    print(f"  best w={ws[ib]:.2f}  {mean[ib]:.4f} +/- {std[ib]:.4f}")
    print(f"  within 1 std : {flat.min():.2f} - {flat.max():.2f}")
    print(f"\nwritten to {out}")
