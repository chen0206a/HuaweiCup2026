# Q2 B5-F1 low-rank pairwise fusion screening

Seed 42, original B0-WCE checkpoint; B0 prediction parameters frozen in eval mode.
Clean train only; fixed B2 clean + 54 missing validation scenarios. No test or attachment3.

| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson | Score | Robust |
|---|---|---:|---:|---:|---:|---:|---:|
| F0 | clean | 0.6456 | 0.6040 | 0.6158 | 0.6368 | 0.7413 | 0.7402 |
| F0 | mean_missing | 0.6421 | 0.6016 | 0.6183 | 0.6320 | 0.7392 | 0.7402 |
| F1 | clean | 0.6332 | 0.6191 | 0.6138 | 0.6327 | 0.7416 | 0.7389 |
| F1 | mean_missing | 0.6244 | 0.6091 | 0.6161 | 0.6271 | 0.7361 | 0.7389 |

F1 minus F0 robust score: -0.001398.

## by_modality

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| F0 | text | 0.6352 | 0.5955 | 0.6213 | 0.6247 | 0.7349 |
| F0 | audio | 0.6460 | 0.6044 | 0.6161 | 0.6367 | 0.7415 |
| F0 | vision | 0.6456 | 0.6048 | 0.6163 | 0.6364 | 0.7415 |
| F1 | text | 0.6143 | 0.5982 | 0.6204 | 0.6185 | 0.7296 |
| F1 | audio | 0.6333 | 0.6188 | 0.6137 | 0.6327 | 0.7415 |
| F1 | vision | 0.6291 | 0.6141 | 0.6135 | 0.6321 | 0.7393 |

## by_ratio

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| F0 | 0.1 | 0.6445 | 0.6024 | 0.6165 | 0.6359 | 0.7405 |
| F0 | 0.2 | 0.6442 | 0.6026 | 0.6173 | 0.6349 | 0.7404 |
| F0 | 0.3 | 0.6439 | 0.6037 | 0.6180 | 0.6330 | 0.7403 |
| F0 | 0.4 | 0.6410 | 0.6011 | 0.6178 | 0.6318 | 0.7387 |
| F0 | 0.5 | 0.6375 | 0.5981 | 0.6199 | 0.6274 | 0.7365 |
| F1 | 0.1 | 0.6296 | 0.6141 | 0.6144 | 0.6317 | 0.7393 |
| F1 | 0.2 | 0.6282 | 0.6126 | 0.6150 | 0.6307 | 0.7384 |
| F1 | 0.3 | 0.6258 | 0.6105 | 0.6156 | 0.6283 | 0.7369 |
| F1 | 0.4 | 0.6239 | 0.6089 | 0.6157 | 0.6266 | 0.7359 |
| F1 | 0.5 | 0.6204 | 0.6057 | 0.6185 | 0.6214 | 0.7334 |

## by_location

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| F0 | early | 0.6393 | 0.5984 | 0.6205 | 0.6293 | 0.7372 |
| F0 | middle | 0.6441 | 0.6056 | 0.6145 | 0.6333 | 0.7410 |
| F0 | late | 0.6429 | 0.6007 | 0.6198 | 0.6336 | 0.7393 |
| F1 | early | 0.6224 | 0.6068 | 0.6190 | 0.6249 | 0.7346 |
| F1 | middle | 0.6236 | 0.6096 | 0.6124 | 0.6277 | 0.7363 |
| F1 | late | 0.6273 | 0.6111 | 0.6170 | 0.6286 | 0.7375 |

## double_stress

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| F0 | text+audio | 0.6406 | 0.6022 | 0.6217 | 0.6260 | 0.7380 |
| F0 | text+vision | 0.6392 | 0.6000 | 0.6224 | 0.6253 | 0.7370 |
| F0 | audio+vision | 0.6442 | 0.6029 | 0.6165 | 0.6363 | 0.7406 |
| F1 | text+audio | 0.6154 | 0.5988 | 0.6194 | 0.6203 | 0.7303 |
| F1 | text+vision | 0.6126 | 0.5969 | 0.6197 | 0.6193 | 0.7290 |
| F1 | audio+vision | 0.6282 | 0.6134 | 0.6139 | 0.6318 | 0.7388 |

## vision_all_zero subset

| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|
| F0 | clean | 0.5333 | 0.5556 | 0.5445 | 0.3371 |
| F0 | mean_missing | 0.5407 | 0.5563 | 0.5574 | 0.3297 |
| F1 | clean | 0.5333 | 0.5460 | 0.5484 | 0.2960 |
| F1 | mean_missing | 0.5160 | 0.5238 | 0.5569 | 0.2934 |

## Interaction contribution

| Condition | R mean | R sample std | R P05 | R P50 | R P95 | TA mean norm | TV mean norm | AV mean norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.15606 | 0.08916 | 0.05025 | 0.13586 | 0.33914 | 0.60407 | 0.57367 | 0.36885 |
| mean_missing | 0.15681 | 0.08940 | 0.05025 | 0.13674 | 0.33766 | 0.60613 | 0.57406 | 0.36815 |

## Verification and timing

Identity: True; frozen B0 tensors: True (20); cached/full rows: True; checkpoint round trip: True.
Parameters: 175748 total, 12288 trainable.
Best clean epoch 5; best robust epoch 5; epochs run 17.
Training 14.36s; validation cache build 3.63s; cached validation 0.75s/epoch; full validation verification 4.39s.
