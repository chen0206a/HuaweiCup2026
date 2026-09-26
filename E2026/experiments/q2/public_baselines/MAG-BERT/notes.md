# MAG-BERT: three-seed experiment notes

Purpose: compare this published mechanism's aligned-50 adaptation under common clean-selected two-task protocol.

Regime: standard clean training.

| Seed | Clean Acc | Clean F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6277 | 0.6090 | 0.6067 | 0.6651 | 0.6105 | 0.5978 | 0.6105 | 0.6483 |
| 43 | 0.6346 | 0.6223 | 0.5886 | 0.6713 | 0.6141 | 0.5859 | 0.6211 | 0.6470 |
| 44 | 0.6470 | 0.6204 | 0.6084 | 0.6653 | 0.6232 | 0.6013 | 0.6133 | 0.6470 |

Paper use: aligned-50 adapted baseline comparison only; not an original-paper reproduction.

Next action: report observed values without further model-specific tuning.
