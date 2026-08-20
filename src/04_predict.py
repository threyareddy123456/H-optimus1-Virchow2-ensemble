"""Stage 4 — decision-level fusion and prediction.

The ensemble combines the softmax outputs of the two heads by a fixed weighted
average and takes the arg-max:

    p = 0.55 * p_hoptimus + 0.45 * p_virchow2
    y = argmax(p)

This is the configuration behind the reported validation results.

Modes
-----
  (default)             0.55/0.45 fusion of both heads' softmax outputs
  --calibrate-fusion    applies the per-class multipliers to the H-optimus-1
                        branch, renormalises, then fuses. This is what the
                        submitted inference container does; it is NOT the
                        configuration that produced the reported validation
                        numbers, and the two differ on roughly 0.3% of patches.
  --single hoptimus     H-optimus-1 alone
  --single virchow2     Virchow2 alone
  --calibrate           with --single hoptimus, applies the multipliers

Output CSV has columns SubjectID and Prediction. SubjectID carries no file
extension.

    python src/04_predict.py                                   # ensemble
    python src/04_predict.py --single hoptimus --calibrate     # standalone, calibrated
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (MLP, softmax_probs, CLASSES,
                    CALIBRATION_MULTIPLIERS, ENSEMBLE_WEIGHT)

DIMS = {"hoptimus": 1536, "virchow2": 2560}


def load_val(root, backbone):
    suffix = "" if backbone == "hoptimus" else "_virchow2"
    d = f"{root}/embeddings/val{suffix}/"
    files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    E, N = [], []
    for f in files:
        z = np.load(os.path.join(d, f))
        E.append(z["embeddings"]); N.append(z["filenames"])
    return np.concatenate(E, 0), np.concatenate(N, 0)


def head(root, backbone, device):
    m = MLP(DIMS[backbone]).to(device)
    m.load_state_dict(torch.load(f"{root}/models/head_{backbone}_full.pt",
                                 map_location=device))
    return m


def calibrate(P, renormalise=True):
    Q = P * CALIBRATION_MULTIPLIERS
    return Q / Q.sum(1, keepdims=True) if renormalise else Q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./brats2026")
    ap.add_argument("--out", default=None)
    ap.add_argument("--single", choices=list(DIMS), default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="with --single hoptimus: apply per-class multipliers")
    ap.add_argument("--calibrate-fusion", action="store_true",
                    help="calibrate the H-optimus-1 branch before fusing (container behaviour)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{a.root}/results", exist_ok=True)

    if a.single:
        X, fn = load_val(a.root, a.single)
        P = softmax_probs(head(a.root, a.single, device), X, device)
        if a.calibrate:
            if a.single != "hoptimus":
                raise SystemExit("--calibrate applies to the H-optimus-1 head only")
            P = P * CALIBRATION_MULTIPLIERS      # arg-max is scale-invariant here
        preds = P.argmax(1)
        tag = f"{a.single}{'_calibrated' if a.calibrate else ''}"
    else:
        Xh, fh = load_val(a.root, "hoptimus")
        Xv, fv = load_val(a.root, "virchow2")
        if not np.array_equal(fh, fv):
            raise SystemExit("embedding filename order differs between backbones")

        Ph = softmax_probs(head(a.root, "hoptimus", device), Xh, device)
        Pv = softmax_probs(head(a.root, "virchow2", device), Xv, device)

        if a.calibrate_fusion:
            Ph = calibrate(Ph)
        w = ENSEMBLE_WEIGHT
        preds = (w * Ph + (1 - w) * Pv).argmax(1)
        fn = fh
        tag = f"ensemble_{int(w*100)}_{int((1-w)*100)}" + \
              ("_calibrated" if a.calibrate_fusion else "")

    out = a.out or f"{a.root}/results/predictions_{tag}.csv"
    # SubjectID must not carry a file extension
    pd.DataFrame({"SubjectID": [str(f) for f in fn],
                  "Prediction": preds}).to_csv(out, index=False)

    print(f"wrote {out}  ({len(preds):,} rows)\n")
    for c in range(10):
        n = int((preds == c).sum())
        print(f"  {CLASSES[c]:<5} {n:>7,}  ({n/len(preds)*100:4.1f}%)")
