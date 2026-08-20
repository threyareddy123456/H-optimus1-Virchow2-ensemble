"""Stage 2 — train one MLP head per backbone on the cached embeddings.

Both encoders stay frozen; only these heads are trained. Class imbalance is
handled on two fronts: inverse-frequency class weights inside a focal loss
(gamma = 2), and a weighted random sampler so scarce classes appear more
often in a minibatch.

    python src/02_train_heads.py --backbone hoptimus
    python src/02_train_heads.py --backbone virchow2

By default this trains the production heads on all labelled data, holding out
3% at random purely as an early-stopping signal. Pass --patient-split to train
on the patient-disjoint training portion instead; that variant is what stage 03
calibrates against and what the development analyses use.
"""
import argparse, glob, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from common import (MLP, FocalLoss, class_weights, patient_split,
                    TRAIN_SEED, CLASSES)

DIMS = {"hoptimus": 1536, "virchow2": 2560}


def load_embeddings(root, backbone):
    suffix = "" if backbone == "hoptimus" else "_virchow2"
    d = f"{root}/embeddings/train{suffix}/"
    E, N = [], []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            E.append(z["embeddings"]); N.append(z["filenames"])
    X, fn = np.concatenate(E, 0), np.concatenate(N, 0)

    lab = np.load(f"{root}/embeddings/train_labels_full.npz")
    fn_to_label = dict(zip(lab["filenames"], lab["labels"]))
    y = np.array([fn_to_label[f] for f in fn])
    assert len(X) == len(y)
    return X, y, fn


def train(X, y, dim, out_path, epochs=40, patience=8, device="cuda"):
    torch.manual_seed(TRAIN_SEED); np.random.seed(TRAIN_SEED)

    w  = class_weights(y)
    cw = torch.FloatTensor(w).to(device)
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))

    n_tr = int(0.97 * len(ds))
    tr, ho = random_split(ds, [n_tr, len(ds) - n_tr],
                          generator=torch.Generator().manual_seed(TRAIN_SEED))

    sw = torch.DoubleTensor([w[y[i]] for i in tr.indices])
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    tl = DataLoader(tr, batch_size=1024, sampler=sampler, num_workers=2, pin_memory=True)
    hl = DataLoader(ho, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    model = MLP(dim).to(device)
    crit  = FocalLoss(weight=cw, gamma=2.0)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.03)
    sch   = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best, waited = 0.0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        sch.step()

        model.eval(); P, T = [], []
        with torch.no_grad():
            for xb, yb in hl:
                P.extend(model(xb.to(device)).argmax(1).cpu().numpy()); T.extend(yb.numpy())
        f1 = f1_score(T, P, average="macro")

        if f1 > best:
            best, waited = f1, 0
            torch.save(model.state_dict(), out_path)
            print(f"epoch {ep+1:3d}  macro-F1 {f1:.4f}  * saved")
        else:
            waited += 1
            if (ep + 1) % 5 == 0:
                print(f"epoch {ep+1:3d}  macro-F1 {f1:.4f}")
            if waited >= patience:
                print(f"early stop at epoch {ep+1}"); break

    print(f"\nbest held-out macro-F1 {best:.4f} -> {out_path}")
    print("note: this hold-out is patch-level, not patient-disjoint, so it is optimistic")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./brats2026")
    ap.add_argument("--backbone", required=True, choices=list(DIMS))
    ap.add_argument("--patient-split", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, fn = load_embeddings(a.root, a.backbone)
    print(f"{a.backbone}: {X.shape}")

    os.makedirs(f"{a.root}/models", exist_ok=True)
    if a.patient_split:
        tr, _ = patient_split(
            fn, y, f"{a.root}/data/train/BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv")
        X, y = X[tr], y[tr]
        out = f"{a.root}/models/head_{a.backbone}_patientsplit.pt"
        print(f"patient-disjoint training portion: {X.shape}")
    else:
        out = f"{a.root}/models/head_{a.backbone}_full.pt"

    train(X, y, DIMS[a.backbone], out, epochs=a.epochs, device=device)
