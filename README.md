# BraTS-Path 2026 — Ensemble of Pathology Foundation Models

Training and inference code for our BraTS-Path 2026 submission (Task 5,
ten-class histologic classification of glioblastoma H&E patches).

Two pathology foundation models are used as **frozen** feature extractors —
H-optimus-1 and Virchow2. Only a lightweight MLP head per backbone is trained.
Because the encoders are never updated, every patch is embedded once and cached,
so head training, calibration and fusion all operate on stored feature vectors.

## Method

| Stage | What happens |
|---|---|
| 1. Features | H-optimus-1 → 1536-d pooled; Virchow2 → 2560-d `[CLS ; mean(patch tokens)]` |
| 2. Heads | MLP `d → 256 → 128 → 10`, BatchNorm + ReLU + dropout 0.5 |
| 3. Calibration | per-class multipliers for the standalone H-optimus-1 configuration |
| 4. Fusion | `p = 0.55 · p_hoptimus + 0.45 · p_virchow2`, then arg-max |

Class imbalance is handled with inverse-frequency class weights inside a focal
loss (γ = 2) and a weighted random sampler.

**Hyperparameters:** AdamW, lr 5e-4, weight decay 0.03, cosine annealing
(T_max = 40, η_min = 1e-6), batch 1024, 40 epochs, early stopping patience 8.

**Seeds:** patient-level split 42; head initialisation and sampling 7.

## Evaluation split

The development split is **patient-disjoint**. For each class in turn, 25% of
the patients contributing that class are drawn and added to the held-out set.
Since one patient contributes patches of several classes, these per-class draws
accumulate into a single held-out set, and a patient selected on behalf of any
class is held out entirely — no patch of a held-out patient is ever seen during
training.

This matters: a random patch-level split places visually near-identical patches
from the same slide on both sides and inflates every metric. `03_calibrate.py`
carries a guard that aborts if held-out macro-F1 exceeds 0.80, which is not
attainable on a genuinely disjoint split here.

## Fusion variants

`04_predict.py` fuses the **raw** softmax outputs of both heads by default.
This is the configuration behind our reported validation results.

Per-class calibration is applied to the **standalone** H-optimus-1
configuration (`--single hoptimus --calibrate`), not inside the ensemble.

`--calibrate-fusion` calibrates the H-optimus-1 branch before fusing. This is
what the submitted inference container does. The two fusion variants differ on
roughly 0.3% of validation patches. Both are provided so either can be
reproduced.

## Usage

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # H-optimus-1 and Virchow2 are gated on the Hub

# 1. features (data/{train,val}/*.tar, WebDataset shards)
python src/01_extract_features.py --backbone hoptimus --split train
python src/01_extract_features.py --backbone virchow2 --split train
python src/01_extract_features.py --backbone hoptimus --split val
python src/01_extract_features.py --backbone virchow2 --split val
python src/01_extract_features.py --labels

# 2. production heads (all labelled data)
python src/02_train_heads.py --backbone hoptimus
python src/02_train_heads.py --backbone virchow2

# 3. calibration — needs a head that has not seen the held-out patients
python src/02_train_heads.py --backbone hoptimus --patient-split
python src/03_calibrate.py

# 4. predictions
python src/04_predict.py                                # ensemble, 0.55/0.45
python src/04_predict.py --single hoptimus --calibrate   # standalone, calibrated
```

Expected layout under `--root`:

```
brats2026/
├── data/train/          shard-0000{00..39}.tar + patch-patient-slide mapping CSV
├── data/val/            val-shard-000000.tar
├── embeddings/          written by stage 1
├── models/              written by stages 2-3
└── results/             written by stage 4
```

## Output format

`SubjectID,Prediction` — `SubjectID` carries **no** file extension;
`Prediction` is an integer in `[0, 9]` following the challenge class order:
`CT, DM, IC, LI, MP, NC, PL, PN, WM, NOTA`.

## Notes

Credentials are read from the environment (`HF_TOKEN`, `SYNAPSE_AUTH_TOKEN`)
and are never stored in this repository.

Hardware: a single NVIDIA A100. Feature extraction dominates the cost
(~3 h per backbone over 40 shards); head training takes minutes.
