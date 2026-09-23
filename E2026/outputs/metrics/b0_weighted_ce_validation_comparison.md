# B0 vs B0-weighted-CE validation comparison

Validation only; the attachment 2 test split was not used.

| Metric | B0 | B0-weighted-CE | Weighted − B0 |
|---|---:|---:|---:|
| Accuracy | 0.6223 | 0.6319 | +0.0096 |
| Macro-F1 | 0.5394 | 0.6062 | +0.0668 |
| MAE | 0.6051 | 0.6276 | +0.0225 |
| Pearson | 0.6356 | 0.6360 | +0.0003 |

## Per-class recall and F1

| Class | B0 recall / F1 | Weighted recall / F1 | Recall Δ | F1 Δ |
|---|---:|---:|---:|---:|
| Negative | 0.7136 / 0.6667 | 0.6650 / 0.6555 | -0.0485 | -0.0112 |
| Neutral | 0.1522 / 0.2414 | 0.4348 / 0.4545 | +0.2826 | +0.2132 |
| Positive | 0.8225 / 0.7101 | 0.7189 / 0.7085 | -0.1036 | -0.0016 |
