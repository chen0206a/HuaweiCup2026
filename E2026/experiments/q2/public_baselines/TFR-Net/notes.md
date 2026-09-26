# TFR-Net: three-seed experiment notes

Purpose: compare this published mechanism's aligned-50 adaptation under common clean-selected two-task protocol.

Regime: missing-aware training.

| Seed | Clean Acc | Clean F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6415 | 0.6304 | 0.6021 | 0.6328 | 0.6397 | 0.6232 | 0.6095 | 0.6280 |
| 43 | 0.6291 | 0.6145 | 0.6255 | 0.6105 | 0.6222 | 0.6045 | 0.6385 | 0.6056 |
| 44 | 0.6415 | 0.6240 | 0.6159 | 0.6368 | 0.6389 | 0.6185 | 0.6215 | 0.6322 |

Paper use: aligned-50 adapted baseline comparison only; not an original-paper reproduction.

Next action: report observed values without further model-specific tuning.
