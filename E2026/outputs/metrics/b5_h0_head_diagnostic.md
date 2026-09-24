# B5-H0 Classification–Regression Head Consistency Diagnostic

Read-only inference on the historical B0-WCE seed42 checkpoint; no training, checkpoint writes, or attachment3. Only the validation split was indexed or evaluated. The aligned pickle is a combined multi-split file and is deserialized as a single pickle object to isolate `valid`; its test split was not indexed, passed to a loader, or evaluated.
Benchmark seed `20260923`, SHA-256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`; 728 validation samples × clean + 54 scenarios.

Default regression-to-class map uses sign (tau=0). Oracle is offline only, not deployable and not a model result.

## Head agreement and error quadrants

- Clean: 82.69% agreement; cls-only 9.34%; reg-only 5.08%; both correct 55.22%; both wrong 30.36%; oracle union 69.64%.
- Mean missing: 81.62% agreement; cls-only 9.53%; reg-only 5.53%; both correct 54.68%; both wrong 30.26%; oracle union 69.74%.
- Clean cls errors corrected by regression: 37/258 = 14.34%; regression errors corrected by cls: 68/289 = 23.53%.
- Mean-missing cls errors corrected by regression: 2175/14070 = 15.46%; regression errors corrected by cls: 3747/15642 = 23.95%.

## True-class breakdown

Values below are conditional rates within each true class (tau=0).

| Condition | Class | N | Cls acc/recall | Reg acc/recall | Agreement | cls-only | reg-only | both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | Negative | 206 | 73.30% | 75.24% | 88.83% | 2.91% | 4.85% | 21.84% |
| Clean | Neutral | 184 | 33.15% | 0.00% | 65.22% | 33.15% | 0.00% | 66.85% |
| Clean | Positive | 338 | 76.33% | 84.02% | 88.46% | 0.30% | 7.99% | 15.68% |
| Mean missing | Negative | 11124 | 72.56% | 74.90% | 87.18% | 3.19% | 5.53% | 21.91% |
| Mean missing | Neutral | 9936 | 33.70% | 0.00% | 64.55% | 33.70% | 0.00% | 66.30% |
| Mean missing | Positive | 18252 | 75.73% | 84.03% | 87.52% | 0.24% | 8.55% | 15.72% |

## Threshold sweep

Descriptive only; no threshold is selected or used as a predictor.

| τ | Clean reg accuracy | Clean agreement | Clean neutral output | Missing reg accuracy | Missing agreement | Missing neutral output |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 60.30% | 86.95% | 9.89% | 60.12% | 85.48% | 9.89% |
| 0.2 | 61.13% | 85.85% | 17.86% | 60.88% | 85.86% | 18.23% |
| 0.3 | 60.85% | 83.65% | 26.10% | 59.88% | 83.32% | 26.84% |
| 0.4 | 56.87% | 77.47% | 36.13% | 57.02% | 77.60% | 36.56% |
| 0.5 | 56.18% | 69.78% | 45.19% | 55.53% | 70.37% | 45.31% |
| 0.75 | 50.27% | 55.22% | 60.03% | 49.87% | 54.92% | 61.04% |
| 1 | 44.92% | 40.25% | 75.00% | 44.25% | 40.61% | 75.36% |

## Scenario groups and paired changes

Scenario rates are unweighted means over the scenarios in each group. Rho rows contain the single-modality scenarios; double-modality rows are separate.

| Group | Name | Scenarios | Agreement | cls-only | reg-only | both wrong |
|---|---|---:|---:|---:|---:|---:|
| by_modality | text | 15 | 80.73% | 9.60% | 5.98% | 30.50% |
| by_modality | audio | 15 | 82.73% | 9.29% | 5.04% | 30.37% |
| by_modality | vision | 15 | 81.85% | 9.56% | 5.40% | 30.04% |
| by_rho | 0.1 | 9 | 82.65% | 9.19% | 5.11% | 30.43% |
| by_rho | 0.2 | 9 | 82.17% | 9.37% | 5.25% | 30.33% |
| by_rho | 0.3 | 9 | 81.55% | 9.63% | 5.51% | 30.10% |
| by_rho | 0.4 | 9 | 81.33% | 9.63% | 5.59% | 30.31% |
| by_rho | 0.5 | 9 | 81.15% | 9.60% | 5.91% | 30.34% |
| by_location | early | 18 | 81.56% | 9.43% | 5.42% | 30.65% |
| by_location | middle | 18 | 81.04% | 9.94% | 5.96% | 29.63% |
| by_location | late | 18 | 82.26% | 9.22% | 5.22% | 30.49% |
| double_modality | text+audio | 3 | 80.04% | 10.07% | 6.18% | 29.76% |
| double_modality | text+vision | 3 | 80.68% | 9.84% | 6.04% | 30.04% |
| double_modality | audio+vision | 3 | 81.91% | 9.39% | 5.27% | 30.31% |

### Paired clean→missing changes

Flip and transition rates are paired by validation ID; rates are then averaged over the scenarios in each group.

| Group | Name | Scenarios | Cls flip | Reg sign flip | Agreement Δ | Agree→disagree | Disagree→agree |
|---|---|---:|---:|---:|---:|---:|---:|
| by_modality | text | 15 | 6.84% | 4.72% | -1.96% | 4.64% | 2.68% |
| by_modality | audio | 15 | 0.19% | 0.11% | +0.04% | 0.12% | 0.16% |
| by_modality | vision | 15 | 1.79% | 1.39% | -0.84% | 1.49% | 0.65% |
| by_rho | 0.1 | 9 | 0.90% | 0.76% | -0.05% | 0.64% | 0.60% |
| by_rho | 0.2 | 9 | 1.98% | 1.36% | -0.52% | 1.50% | 0.98% |
| by_rho | 0.3 | 9 | 2.84% | 1.98% | -1.14% | 2.30% | 1.16% |
| by_rho | 0.4 | 9 | 4.00% | 2.82% | -1.36% | 2.79% | 1.43% |
| by_rho | 0.5 | 9 | 4.98% | 3.43% | -1.54% | 3.19% | 1.65% |
| by_location | early | 18 | 3.43% | 2.57% | -1.14% | 2.43% | 1.30% |
| by_location | middle | 18 | 3.41% | 2.39% | -1.65% | 2.77% | 1.12% |
| by_location | late | 18 | 3.08% | 2.05% | -0.43% | 1.97% | 1.54% |
| double_modality | text+audio | 3 | 6.68% | 4.62% | -2.66% | 5.36% | 2.70% |
| double_modality | text+vision | 3 | 6.91% | 5.08% | -2.01% | 4.95% | 2.93% |
| double_modality | audio+vision | 3 | 1.83% | 1.28% | -0.78% | 1.47% | 0.69% |

## Regression distributions by true class

Predicted continuous regression values; negative/positive wrong-sign rates use strict sign tests. Neutral abs(prediction) is not a class prediction.

| Condition | True class | Mean | Std | Median | P05 | P25 | P75 | P95 | Sign / abs diagnostic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | Negative | -0.458 | 0.700 | -0.390 | -1.680 | -0.923 | -0.014 | 0.689 | reg>0 24.76% |
| clean | Neutral | 0.151 | 0.593 | 0.192 | -0.931 | -0.168 | 0.540 | 0.964 | |reg| mean 0.479; median 0.413; P75 0.708 |
| clean | Positive | 0.628 | 0.662 | 0.646 | -0.537 | 0.249 | 1.098 | 1.609 | reg<0 15.98% |
| mean_missing | Negative | -0.451 | 0.696 | -0.391 | -1.663 | -0.941 | 0.001 | 0.694 | reg>0 25.10% |
| mean_missing | Neutral | 0.151 | 0.588 | 0.187 | -0.935 | -0.159 | 0.532 | 0.990 | |reg| mean 0.476; median 0.396; P75 0.712 |
| mean_missing | Positive | 0.620 | 0.659 | 0.645 | -0.537 | 0.234 | 1.096 | 1.611 | reg<0 15.97% |

## Confidence and regression evidence

| Condition | Subset | N | Mean max prob | Mean top1 margin | Mean |reg_pred| |
|---|---|---:|---:|---:|---:|
| clean | cls_correct | 470 | 0.712 | 0.517 | 0.764 |
| clean | cls_wrong | 258 | 0.611 | 0.355 | 0.502 |
| clean | heads_agree | 602 | 0.711 | 0.515 | 0.780 |
| clean | heads_disagree | 126 | 0.510 | 0.196 | 0.152 |
| clean | neutral | 184 | 0.602 | 0.338 | 0.479 |
| mean_missing | cls_correct | 25242 | 0.710 | 0.513 | 0.759 |
| mean_missing | cls_wrong | 14070 | 0.609 | 0.351 | 0.496 |
| mean_missing | heads_agree | 32087 | 0.711 | 0.515 | 0.779 |
| mean_missing | heads_disagree | 7225 | 0.507 | 0.189 | 0.159 |
| mean_missing | neutral | 9936 | 0.601 | 0.336 | 0.476 |

Confidence quintiles (clean and mean-missing) and |reg_pred| quintiles are stored in the JSON. Across both conditions, disagreement concentrates at low classification confidence and small |reg_pred|; Neutral predictions have broad regression magnitudes rather than a narrow band around zero.

## Oracle diagnostic

Oracle union accuracy (tau=0): clean 69.64% (gain 5.08% over cls); mean missing 69.74% (gain 5.53%). This is an offline upper bound only, not deployable and not a model result.

## Mechanism judgment

Classification and regression show **partial complementarity with substantial shared errors**. Regression alone corrects some classification errors (5.08% clean / 5.53% mean-missing under the fixed sign map), and the oracle union adds about five percentage points. However, both heads are wrong on about 30% of examples, and Neutral recall of sign-mapped regression is necessarily zero; the neutral regression predictions are broadly spread and overlap the Negative/Positive distributions. This is not strong complementarity, but not pure redundancy either. The diagnostic supports, at most, a future lightweight consistency hypothesis focused on continuous evidence and Neutral handling; no such method is implemented here.

## Per-sample exports

- `b5_h0_predictions.jsonl`: one JSON object per validation sample and condition.
- `b5_h0_predictions.csv`: flattened CSV; logits/probabilities stored as JSON arrays.
