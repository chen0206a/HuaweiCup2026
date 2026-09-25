# MISA regression branch sanity check

Status: `MISA_REGRESSION_SANITY = PASS`.

`BASELINE_TABLE_LOCKED = YES` (the existing three-seed baseline results are retained unchanged).

## Target path and scale

Training reads `Aligned50Dataset.reg_labels` from the same `regression_labels` field used by TFN/MulT. Dataset casts to float32; normalization is `none`; valid prediction metrics use the original regression scale. No inverse transform is configured or required.

Train target: n=3395, min=-3, max=3, mean=0.165832, std=1.11513.
Valid target: n=728, min=-3, max=3, mean=0.160943, std=1.0436.
Both raw-to-Dataset target comparisons were exact after the training pipeline's float32 cast.

## Regression head and loss

MISA output path is shared/private factors → six-vector Transformer encoder → 768→384 fusion → parallel heads. Regression head is `Linear(384,1)` with no activation and `.squeeze(-1)`, producing `[B]`; target is `[B]`. Evaluation calls the same model forward and regression head. Validation model is in `eval()`; dropout is disabled.

The optimized objective is `WeightedCE + SmoothL1 + 0.3 × DifferenceLoss + CMD + ReconstructionLoss`. `lambda_reg=1.0`; no target normalization or clipping is present. Gradient diagnostics use three fixed, ordered train batches per seed and do not update weights.

| Seed | CE | SmoothL1 | 0.3×Diff | CMD | Reconstruction | Fusion ‖g_reg‖ | Shared ‖g_reg‖ | Shared ‖g_diff‖ | Shared ‖g_CMD‖ | Shared ‖g_rec‖ | Reg-head ‖g‖ | CE/reg fusion cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6247 | 0.2345 | 0.0061 | 0.0132 | 0.0152 | 0.1717 | 0.09713 | 0.007879 | 0.01146 | 0.007164 | 0.7956 | 0.2114 |
| 43 | 0.6124 | 0.2656 | 0.0061 | 0.0144 | 0.0148 | 0.2344 | 0.09624 | 0.008476 | 0.01226 | 0.005828 | 1.066 | 0.3022 |
| 44 | 0.5680 | 0.2157 | 0.0037 | 0.0102 | 0.0154 | 0.1495 | 0.05162 | 0.005015 | 0.01075 | 0.006282 | 0.8308 | 0.2074 |

## Validation prediction distribution

| Seed | Pred mean±SD | Pred min / q05 / q10 / q25 / median / q75 / q90 / q95 / max | True mean±SD | True min / q05 / q10 / q25 / median / q75 / q90 / q95 / max | Bias | MAE | RMSE | Pearson | Spearman | Pred/target SD | Slope | Intercept |
|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.389 ± 1.689 | -3.570 / -3.230 / -2.241 / -0.563 / 0.536 / 1.701 / 2.510 / 2.767 / 3.171 | 0.161 ± 1.044 | -3.000 / -1.667 / -1.333 / -0.333 / 0.000 / 0.667 / 1.383 / 2.000 / 3.000 | 0.228 | 1.029 | 1.322 | 0.636 | 0.637 | 1.618 | 1.029 | 0.223 |
| 43 | 0.525 ± 1.303 | -2.974 / -2.253 / -1.642 / -0.066 / 0.769 / 1.311 / 1.983 / 2.433 / 3.087 | 0.161 ± 1.044 | -3.000 / -1.667 / -1.333 / -0.333 / 0.000 / 0.667 / 1.383 / 2.000 / 3.000 | 0.364 | 0.891 | 1.103 | 0.626 | 0.629 | 1.248 | 0.781 | 0.399 |
| 44 | 0.245 ± 1.382 | -3.210 / -2.469 / -1.997 / -0.596 / 0.474 / 1.070 / 1.960 / 2.268 / 3.189 | 0.161 ± 1.044 | -3.000 / -1.667 / -1.333 / -0.333 / 0.000 / 0.667 / 1.383 / 2.000 / 3.000 | 0.085 | 0.833 | 1.104 | 0.619 | 0.622 | 1.324 | 0.819 | 0.114 |

## Evaluation reproduction

Saved validation metrics were recomputed from the MISA checkpoints with the same ordered valid loader and shared evaluator. A fixed RNG seed sampled 20 valid rows per MISA seed; NumPy MAE and SciPy Pearson were independently compared with the project's metric functions.

| Seed | MAE absolute delta | Pearson absolute delta |
|---:|---:|---:|
| 42 | 0 | 0 |
| 43 | 0 | 1.11e-16 |
| 44 | 0 | 0 |

## Paired error inspection

Per-seed top-20 cases where MISA absolute error exceeds corresponding-seed MulT/P2, common top-20 hard cases, and intensity-bin metrics are in `misa_regression_error_cases.csv`. Full model prediction distributions are in `misa_regression_prediction_stats.csv`.

The three validation prediction means are above the target mean (bias +0.085 to +0.364). Prediction SD is 1.25–1.62 times target SD; seed 42 has the widest tails, including 60/728 predictions outside [-3,3]. Seeds 43 and 44 have 3 and 7 such predictions, respectively. Errors remain present across all four target-intensity bins, so the high MAE is not limited to only the strongest targets.

There is no evidence of a target-scale, head-shape, target-alignment, or metric implementation error. The current high absolute error is retained as an outcome of this aligned-50 MISA adaptation and training setup.

No checkpoint or training result was modified. Attachment2 test was not included in a Dataset; Attachment3/4 were not loaded. The monolithic Attachment2 pickle container is deserialized, and only train/valid splits are passed to Dataset.
