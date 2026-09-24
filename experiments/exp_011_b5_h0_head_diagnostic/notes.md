# B5-H0 Classification–Regression Head Consistency Diagnostic

## Why

Determine whether the original B0-WCE classification and regression heads provide complementary predictions or primarily share errors. This is a read-only diagnostic; no model or checkpoint was trained or modified.

## Protocol

Used historical B0-WCE seed42 best-selection-score checkpoint (`lambda_reg=1.0`) on 728 validation samples under clean and the unchanged 54-scenario benchmark, seed 20260923, SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`. The sign mapping is the fixed tau=0 descriptive diagnostic; the seven tau thresholds are descriptive only. No threshold was selected. No attachment2 test samples were indexed/evaluated and attachment3 was not accessed. The aligned pickle is a combined file and is deserialized as one object to isolate its validation split.

## Findings

Under the fixed sign mapping, head agreement was 82.69% clean and 81.62% mean-missing. Clean quadrants: both correct 55.22%, cls-only correct 9.34%, reg-only correct 5.08%, both wrong 30.36%. Mean-missing: 54.68%, 9.53%, 5.53%, and 30.26%, respectively. Oracle union accuracy is 69.64% clean and 69.74% mean-missing, a 5.08/5.53 percentage-point upper-bound gain over classification alone; oracle is not deployable and not a model result.

Neutral is the main gap: classification recall is 33.15% clean / 33.70% mean-missing, while sign-mapped regression recall is 0 by construction. Neutral regression predictions are not confined to a narrow interval: clean |prediction| mean 0.479, median 0.413, P75 0.708; mean-missing mean 0.476, median 0.396, P75 0.712, with broad overlap against Negative and Positive predictions. For tau=.5, descriptive regression Neutral recall reaches 61.96% clean, but overall regression-derived accuracy is 56.18% and agreement falls to 69.78%; no tau is selected.

Confidence and |regression| help explain disagreement: clean disagreement examples have mean max-softmax 0.510, top1 margin 0.196, and |reg_pred| 0.152, versus 0.711, 0.515, and 0.780 when heads agree. Mean-missing pattern is similar. Agreement declines modestly as single-modality rho rises (.826 at rho=.1 to .812 at rho=.5); paired cls flip rises .90% to 4.98%. This change is driven mainly by text masking (6.84% cls flip, 4.72% regression sign flip); audio masking has little effect. The both-wrong rate remains near 30% across conditions.

## Mechanism judgment

Classify this as **partial complementarity with substantial shared errors**. Regression sometimes corrects classification, and the oracle gain is nonzero, but both heads are wrong on about 30% and the sign mapping cannot recover Neutral. This suggests a possible future lightweight consistency/Neutral-axis hypothesis, but does not establish that it will help. No H1 or other follow-up was started.

Full scenario groups, threshold sweep, class-specific regression distributions, confidence bins, paired flips, and prediction-level details are in `metrics.json` and `E2026/outputs/metrics/b5_h0_head_diagnostic.md`. Per-sample JSONL/CSV exports are present in `E2026/outputs/metrics/` as generated diagnostic artifacts.
