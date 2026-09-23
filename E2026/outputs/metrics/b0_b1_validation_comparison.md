# B0 / B1 validation comparison

Every row uses that run's best project validation selection-score checkpoint. Attachment 2 test was not used.

| Model | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Best score epoch | Best loss epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-CE | 0.6387 | 0.6056 | 0.6213 | 0.6209 | 0.7378 | 5 | 3 |
| B0-WeightedCE | 0.6456 | 0.6040 | 0.6158 | 0.6368 | 0.7413 | 3 | 2 |
| B1-CE | 0.6250 | 0.5861 | 0.5928 | 0.6569 | 0.7352 | 7 | 3 |
| B1-WeightedCE | 0.6236 | 0.6038 | 0.6145 | 0.6500 | 0.7375 | 5 | 4 |

## Per-class recall / F1

| Model | Negative | Neutral | Positive |
|---|---:|---:|---:|
| B0-CE | 0.6796 / 0.6699 | 0.3804 / 0.4348 | 0.7544 / 0.7123 |
| B0-WeightedCE | 0.7330 / 0.6756 | 0.3315 / 0.4136 | 0.7633 / 0.7227 |
| B1-CE | 0.7282 / 0.6711 | 0.3261 / 0.3822 | 0.7249 / 0.7050 |
| B1-WeightedCE | 0.6408 / 0.6667 | 0.4620 / 0.4497 | 0.7012 / 0.6950 |

## Best-loss vs best-score

| Model | Different epochs? | Best loss epoch | Best score epoch |
|---|---:|---:|---:|
| B0-CE | Yes | 3 | 5 |
| B0-WeightedCE | Yes | 2 | 3 |
| B1-CE | Yes | 3 | 7 |
| B1-WeightedCE | Yes | 4 | 5 |

## Vision-all-zero validation subsets

| Model | N | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|---:|
| B0-CE | 15 | 0.6667 | 0.6603 | 0.6225 | 0.1527 |
| B0-WeightedCE | 15 | 0.5333 | 0.5556 | 0.5445 | 0.3371 |
| B1-CE | 15 | 0.5333 | 0.5460 | 0.5659 | 0.4385 |
| B1-WeightedCE | 15 | 0.5333 | 0.5256 | 0.5591 | 0.7182 |
