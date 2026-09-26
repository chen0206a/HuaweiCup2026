# MMIM: three-seed experiment notes

Purpose: compare this published mechanism's aligned-50 adaptation under common clean-selected two-task protocol.

Regime: standard clean training.

| Seed | Clean Acc | Clean F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6332 | 0.6103 | 0.6006 | 0.6323 | 0.6273 | 0.6059 | 0.6037 | 0.6285 |
| 43 | 0.6415 | 0.6116 | 0.6201 | 0.6369 | 0.6348 | 0.6056 | 0.6239 | 0.6324 |
| 44 | 0.6332 | 0.6112 | 0.6100 | 0.6432 | 0.6284 | 0.6063 | 0.6128 | 0.6385 |

Paper use: aligned-50 adapted baseline comparison only; not an original-paper reproduction.

Next action: report observed values without further model-specific tuning.
