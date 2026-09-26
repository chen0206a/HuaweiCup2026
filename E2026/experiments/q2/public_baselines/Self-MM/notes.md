# Self-MM: three-seed experiment notes

Purpose: compare this published mechanism's aligned-50 adaptation under common clean-selected two-task protocol.

Regime: standard clean training.

| Seed | Clean Acc | Clean F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6415 | 0.6196 | 0.6150 | 0.6383 | 0.6317 | 0.6096 | 0.6178 | 0.6329 |
| 43 | 0.6332 | 0.6029 | 0.6025 | 0.6419 | 0.6304 | 0.6012 | 0.6053 | 0.6374 |
| 44 | 0.6319 | 0.6094 | 0.6122 | 0.6466 | 0.6285 | 0.6052 | 0.6163 | 0.6409 |

Paper use: aligned-50 adapted baseline comparison only; not an original-paper reproduction.

Next action: report observed values without further model-specific tuning.
