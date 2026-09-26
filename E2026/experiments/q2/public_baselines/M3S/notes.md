# M3S: three-seed experiment notes

Purpose: compare this published mechanism's aligned-50 adaptation under common clean-selected two-task protocol.

Regime: missing-aware training.

| Seed | Clean Acc | Clean F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6071 | 0.5970 | 0.6519 | 0.5803 | 0.5670 | 0.5571 | 0.6542 | 0.5609 |
| 43 | 0.5962 | 0.5692 | 0.6860 | 0.5958 | 0.5757 | 0.5529 | 0.6636 | 0.5757 |
| 44 | 0.6071 | 0.5663 | 0.6739 | 0.5617 | 0.5773 | 0.5378 | 0.6682 | 0.5376 |

Paper use: aligned-50 adapted baseline comparison only; not an original-paper reproduction.

Next action: report observed values without further model-specific tuning.
