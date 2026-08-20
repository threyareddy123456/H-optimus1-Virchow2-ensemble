"""Shared model definition, loss, and the patient-level split."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

CLASSES = ["CT", "DM", "IC", "LI", "MP", "NC", "PL", "PN", "WM", "NOTA"]

# Selected by greedy search on the patient-disjoint held-out set (stage 03).
# Order matches CLASSES.
CALIBRATION_MULTIPLIERS = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 3.0, 1.5])

ENSEMBLE_WEIGHT = 0.55          # weight on H-optimus-1; Virchow2 gets 1 - w
SPLIT_SEED      = 42            # patient-level split
TRAIN_SEED      = 7             # head initialisation / sampling


class MLP(nn.Module):
    """Classifier head: input -> 256 -> 128 -> 10."""
    def __init__(self, input_dim, num_classes=10, dropout=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.network(x)


class FocalLoss(nn.Module):
    """Class-weighted focal loss, gamma = 2."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__(); self.weight, self.gamma = weight, gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def class_weights(y, n_classes=10):
    """Inverse frequency: w_c = N / (C * n_c)."""
    from collections import Counter
    cnt, N = Counter(y.tolist()), len(y)
    return [N / (n_classes * max(cnt[i], 1)) for i in range(n_classes)]


def softmax_probs(model, X, device, batch=4096):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.FloatTensor(X[i:i + batch]).to(device)
            out.append(F.softmax(model(b), 1).cpu().numpy())
    return np.concatenate(out, 0)


def patient_split(filenames, labels, mapping_csv, val_frac=0.25, seed=SPLIT_SEED):
    """Patient-disjoint split.

    For each class in turn, `val_frac` of the patients contributing that class
    are drawn and added to the held-out set. Because one patient contributes
    patches of several classes, the per-class draws accumulate into a single
    held-out set and a patient selected on behalf of any class is held out
    entirely. No patch of a held-out patient is ever seen in training.
    """
    mp = pd.read_csv(mapping_csv)
    fn_to_pat = dict(zip(mp["Name"], mp["Patient"]))
    patients = np.array([fn_to_pat.get(f, -1) for f in filenames])

    df = pd.DataFrame({"patient": patients, "label": labels})
    df = df[df.patient != -1]

    rng = np.random.RandomState(seed)
    val = set()
    for c in range(10):
        p = sorted(df[df.label == c]["patient"].unique())
        rng.shuffle(p)
        val.update(p[:max(1, int(val_frac * len(p)))])

    val_mask   = np.array([p in val for p in patients])
    train_mask = (~val_mask) & (patients != -1)
    return train_mask, val_mask
