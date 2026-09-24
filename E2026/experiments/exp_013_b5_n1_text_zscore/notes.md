# B5-N1 text-only train-stat feature-wise Z-score

## Objective

Test whether feature-wise z-scoring only the 768-dimensional text input independently improves the original B0-WCE. Seed42 only; no pooling, fusion, gate, reconstruction, transformer, or augmentation changes.

## Protocol

N0 reuses the historical seed42 B0-WCE best-selection checkpoint and was re-evaluated on clean plus the frozen 54-scenario validation benchmark. It reproduced the saved clean metrics and benchmark robust score exactly. N1 fully retrained the complete original B0 architecture from seed42 initialization with train-derived balanced CE and SmoothL1 (lambda=1), AdamW (LR 0.001, weight decay 0.0001), batch 128, max 80 epochs, patience 12, clean-only training, and no normalization for audio/vision. N0/N1 initial states were verified tensor-identical; train sampler used the same seed and order protocol.

The text scaler was fit only on valid timesteps in train (83,672 positions), per each of 768 features, with epsilon 1e-6. Valid/test/attachment3 data do not contribute to its statistics. Dataset normalization occurs before synthetic block masking; `apply_blocks` then overwrites artificial missing spans to exact zero. Padding remains controlled only by the audited padding mask and is excluded from fit and masked mean pooling. The aligned pickle is monolithic, so Python deserializes its container; code selects only train/valid keys and never indexes, constructs a Dataset for, evaluates, or uses the test key. Attachment3 is not accessed.

## Result

N0 robust score is 0.740248. N1 best-robust score is 0.734474 at epoch 1, for Δ(N1−N0) = −0.005774. Best-clean and best-robust checkpoints both select epoch 1. Training continues through the fixed patience stop (13 epochs; 69.6 seconds). N1 improves Neutral recall/F1 but reduces Accuracy and the aggregate clean/missing selection scores; every reported modality/rho/location/double-modality subgroup selection score is below N0. This is a negative seed42 screening result, so stop the normalization route; do not run other modalities, epsilon grids, or P2+z-score.

Exact train scaler stats, normalization sanity checks, full per-scenario metrics, projections, and checkpoints are recorded in `outputs/metrics/b5_n1_text_zscore_*` and the two best checkpoints under `outputs/checkpoints/` on the GPU server.
