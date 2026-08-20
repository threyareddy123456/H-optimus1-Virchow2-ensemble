"""Stage 3 — per-class probability calibration for the standalone H-optimus-1 head.

Arg-max over the raw softmax of an imbalanced classifier under-predicts rare
classes. This stage searches, one class at a time, for a fixed multiplier that
maximises macro-F1 on the patient-disjoint held-out set.

The search must use a head that has NOT seen the held-out patients, otherwise
the multipliers are fitted on data the model memorised. Train that head first:

    python src/02_train_heads.py --backbone hoptimus --patient-split
    python src/03_calibrate.py

The multipliers this reproduces are [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0,
3.0, 1.5], i.e. WM x3.0, DM x2.0, PL and NOTA x1.5, everything else unchanged.
They are applied to the standalone H-optimus-1 configuration (stage 04,
--single hoptimus --calibrate).
"""
import argparse, os, sys
import numpy as np, torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from common import MLP, patient_split, softmax_probs, CLASSES

GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
LEAKAGE_THRESHOLD = 0.80


def greedy_search(probs, y):
    """One pass over classes, keeping the best multiplier found for each."""
    mult = np.ones(10)
    best = f1_score(y, probs.argmax(1), average="macro")
    print(f"baseline (arg-max) macro-F1 {best:.4f}\n")

    for c in range(10):
        chosen = 1.0
        for m in GRID:
            trial = mult.copy(); trial[c] = m
            f1 = f1_score(y, (probs * trial).argmax(1), average="macro")
            if f1 > best:
                best, chosen = f1, m
        mult[c] = chosen
        print(f"  {CLASSES[c]:<5} multiplier {chosen:<4} macro-F1 {best:.4f}")

    return mult, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./brats2026")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module
    load = import_module("02_train_heads").load_embeddings

    X, y, fn = load(a.root, "hoptimus")
    _, val = patient_split(
        fn, y, f"{a.root}/data/train/BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv")
    Xv, yv = X[val], y[val]
    print(f"held-out patients: {Xv.shape[0]:,} patches\n")

    ckpt = f"{a.root}/models/head_hoptimus_patientsplit.pt"
    if not os.path.exists(ckpt):
        raise SystemExit("run: python src/02_train_heads.py --backbone hoptimus --patient-split")

    model = MLP(1536).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    probs = softmax_probs(model, Xv, device)

    # Guard: a head that never saw these patients cannot score near-perfectly.
    probe = f1_score(yv, probs.argmax(1), average="macro")
    if probe > LEAKAGE_THRESHOLD:
        raise SystemExit(
            f"ABORT: held-out macro-F1 {probe:.3f} is impossibly high for a "
            f"patient-disjoint split (expected ~0.45). Leakage suspected.")

    mult, best = greedy_search(probs, yv)
    print(f"\nmultipliers: {dict(zip(CLASSES, mult))}")
    np.save(f"{a.root}/models/calibration_multipliers.npy", mult)
