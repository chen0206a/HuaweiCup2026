# B4' reliability-aware dynamic fusion, seed 42 screening

## Why run it

The three-seed B3.2 check did not establish a stable reconstruction gain. This independent branch tests whether a small router can adjust the frozen B0-WCE modality embeddings using only observable feature quality statistics.

## Procedure and verification

The seed-42 B0-WCE best-selection checkpoint initializes all prediction parameters; only the 25,283-parameter router trains. The B2 block-mask generator and fixed validation benchmark (54 missing scenarios plus clean, benchmark seed 20260923, SHA-256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`) are unchanged. Train/valid only; no test or attachment3. Weighted CE uses train-split class counts, with the original SmoothL1 regression term.

At initialization, B4' matches B0 logits and regression predictions exactly on all 728 validation samples. All 20 frozen B0 tensors equal the source checkpoint before and after training. Validation caches frozen B0 embeddings and seven observable statistics once; the initial and trained cached evaluator matches normal full forward exactly across all 55 rows and compared metric fields. CPU/CUDA forward and backward, zero-statistic edge cases, and checkpoint round trip pass.

## Result

The best robust checkpoint occurs at epoch 2; best clean at epoch 1. B4' robust score is 0.743974 versus B0-WCE 0.740248 (+0.003726). Clean selection score is 0.745131 versus 0.741331, and mean-missing selection score is 0.742817 versus 0.739165. Macro-F1 rises by 0.010821 clean and 0.010312 mean missing. MAE is effectively unchanged; clean MAE is 0.000090 worse. All modality, ratio, position, and double-stress group selection scores improve relative to B0, although the text+audio and text+vision double-stress gains are below 0.001.

The 15 validation `vision_all_zero` samples retain identical clean Accuracy (0.5333) and Macro-F1 (0.5556), while MAE worsens from 0.5445 to 0.5564 and Pearson falls from 0.3371 to 0.3237. Their mean-missing four metrics also worsen versus B0. This small subset is a material caveat, though it does not show a classification collapse. The gate is mostly a modality recalibration: its mean values change little between clean and missing, and the damaged text gate increases slightly as rho grows. A genuine damage-sensitive routing mechanism is not yet established.

## Paper status / next action

This is seed-42 screening evidence only. The positive robust score without broad group collapse makes a B4.1 three-seed stability check reasonable, but no B4.1 work begins here. Do not claim a general or statistically established gain. Full per-scenario metrics and gate distributions are in `outputs/metrics/b4_reliability_metrics.json` and `b4_reliability_report.md`.
