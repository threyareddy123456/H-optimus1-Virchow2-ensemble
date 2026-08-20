# Trained classifier heads

The two MLP heads behind the reported results. Both encoders are frozen, so
these are the only trained parameters in the pipeline.

| File | Backbone | Input dim | Original name |
|---|---|---|---|
| `head_hoptimus_full.pt` | H-optimus-1 | 1536 | `best_mlp_final_smallnet.pt` |
| `head_virchow2_full.pt` | Virchow2 | 2560 | `best_mlp_virchow2_full.pt` |

Architecture: `d → 256 → 128 → 10`, BatchNorm + ReLU + dropout 0.5
(`src/common.py:MLP`). Trained on all labelled data — see `src/02_train_heads.py`.

To run inference without retraining, copy them where stage 4 expects them:

```bash
mkdir -p brats2026/models
cp checkpoints/*.pt brats2026/models/
python src/04_predict.py
```

This still needs cached validation embeddings from stage 1, since the frozen
encoders are not included here — they are downloaded from the Hugging Face Hub.
