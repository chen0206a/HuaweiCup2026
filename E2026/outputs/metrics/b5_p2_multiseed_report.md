# B5-P2 attention residual pooling: three-seed replication

Paired B0-WCE/P2 validation comparison; P2 checkpoints selected by robust score. Frozen benchmark clean + 54 scenarios, seed 20260923. No test/attachment3.

## Paired seed results

| Training seed | B0 robust | P2 robust | Paired Δ robust | B0 epoch | P2 best-clean epoch | P2 best-robust epoch |

|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.740248 | 0.745847 | +0.005600 | 3 | 20 | 15 |
| 43 | 0.741620 | 0.745411 | +0.003791 | 3 | 12 | 12 |
| 44 | 0.743162 | 0.743073 | -0.000090 | 4 | 1 | 1 |

Mean paired Δrobust = +0.003100 ± 0.002907 (sample SD); positive in 2/3 seeds.

## Three-seed mean ± sample SD

| Model | Split | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Robust score |

|---|---|---:|---:|---:|---:|---:|---:|
| B0-WCE | clean | 0.6401 ± 0.0048 | 0.6139 ± 0.0091 | 0.6079 ± 0.0103 | 0.6409 ± 0.0043 | 0.7433 ± 0.0018 | 0.7417 ± 0.0015 |
| B0-WCE | mean_missing | 0.6349 ± 0.0062 | 0.6092 ± 0.0075 | 0.6111 ± 0.0098 | 0.6359 ± 0.0038 | 0.7401 ± 0.0011 | 0.7417 ± 0.0015 |
| B5-P2 | clean | 0.6419 ± 0.0057 | 0.6241 ± 0.0026 | 0.6059 ± 0.0066 | 0.6472 ± 0.0025 | 0.7472 ± 0.0019 | 0.7448 ± 0.0015 |
| B5-P2 | mean_missing | 0.6340 ± 0.0049 | 0.6163 ± 0.0012 | 0.6091 ± 0.0060 | 0.6414 ± 0.0025 | 0.7424 ± 0.0011 | 0.7448 ± 0.0015 |

## Per-seed four metrics

Values are selected B0 checkpoint and paired P2 best-robust checkpoint. Full values and groups are in JSON.

| Seed | Model | Condition | Acc | Macro-F1 | MAE | Pearson | Score |

|---:|---|---|---:|---:|---:|---:|---:|
| 42 | B0-WCE | clean | 0.6456 | 0.6040 | 0.6158 | 0.6368 | 0.7413 |
| 42 | B0-WCE | mean_missing | 0.6421 | 0.6016 | 0.6183 | 0.6320 | 0.7392 |
| 42 | P2 | clean | 0.6484 | 0.6234 | 0.6083 | 0.6462 | 0.7484 |
| 42 | P2 | mean_missing | 0.6397 | 0.6152 | 0.6105 | 0.6403 | 0.7433 |
| 43 | B0-WCE | clean | 0.6374 | 0.6160 | 0.5962 | 0.6406 | 0.7436 |
| 43 | B0-WCE | mean_missing | 0.6312 | 0.6095 | 0.6000 | 0.6360 | 0.7397 |
| 43 | P2 | clean | 0.6401 | 0.6270 | 0.5984 | 0.6501 | 0.7481 |
| 43 | P2 | mean_missing | 0.6316 | 0.6176 | 0.6025 | 0.6443 | 0.7427 |
| 44 | B0-WCE | clean | 0.6374 | 0.6218 | 0.6118 | 0.6454 | 0.7450 |
| 44 | B0-WCE | mean_missing | 0.6314 | 0.6167 | 0.6152 | 0.6397 | 0.7413 |
| 44 | P2 | clean | 0.6374 | 0.6218 | 0.6110 | 0.6454 | 0.7450 |
| 44 | P2 | mean_missing | 0.6308 | 0.6163 | 0.6143 | 0.6397 | 0.7411 |

## P2 training diagnostics

| Seed | γ text | γ audio | γ vision | Clean epoch | Robust epoch | Seconds | Frozen B0 exact |

|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | +0.3645 | -0.2614 | -0.1222 | 20 | 15 | 138.1 | PASS |
| 43 | +0.2820 | -0.2115 | +0.0479 | 12 | 12 | 125.7 | PASS |
| 44 | -0.0102 | +0.0065 | +0.0126 | 1 | 1 | 68.5 | PASS |

### Attention entropy and concentration

Normalized entropy, mean max temporal weight, and fraction with max weight ≥0.9 are reported for P2 best-robust; seed42 values are reused from exp_008.

| Seed | Condition | Modality | Normalized entropy | Mean max weight | Fraction max≥.9 |

|---:|---|---|---:|---:|---:|
| 42 | clean | text | 0.5930 | 0.4351 | 0.0275 |
| 42 | clean | audio | 0.2389 | 0.4949 | 0.0000 |
| 42 | clean | vision | 0.8654 | 0.1695 | 0.0027 |
| 42 | mean_missing | text | 0.6469 | 0.3786 | 0.0221 |
| 42 | mean_missing | audio | 0.4011 | 0.3608 | 0.0000 |
| 42 | mean_missing | vision | 0.8624 | 0.1695 | 0.0031 |
| 43 | clean | text | 0.7637 | 0.2945 | 0.0014 |
| 43 | clean | audio | 0.6288 | 0.3795 | 0.0247 |
| 43 | clean | vision | 0.8992 | 0.1412 | 0.0000 |
| 43 | mean_missing | text | 0.7901 | 0.2725 | 0.0015 |
| 43 | mean_missing | audio | 0.6037 | 0.3913 | 0.0295 |
| 43 | mean_missing | vision | 0.9050 | 0.1276 | 0.0000 |
| 44 | clean | text | 0.9882 | 0.0790 | 0.0000 |
| 44 | clean | audio | 0.7975 | 0.2333 | 0.0000 |
| 44 | clean | vision | 0.9313 | 0.1223 | 0.0000 |
| 44 | mean_missing | text | 0.9893 | 0.0783 | 0.0000 |
| 44 | mean_missing | audio | 0.8153 | 0.1883 | 0.0000 |
| 44 | mean_missing | vision | 0.9327 | 0.1129 | 0.0000 |

## Subgroup selection score (mean ± sample SD across seeds)

| Group | B0-WCE | P2 | Δ(P2−B0) |

|---|---:|---:|---:|
| by modality: text | 0.7350 ± 0.0009 | 0.7358 ± 0.0008 | +0.0008 |
| by modality: audio | 0.7433 ± 0.0017 | 0.7470 ± 0.0021 | +0.0037 |
| by modality: vision | 0.7427 ± 0.0012 | 0.7458 ± 0.0020 | +0.0031 |
| by ratio: 0.1 | 0.7423 ± 0.0018 | 0.7457 ± 0.0020 | +0.0034 |
| by ratio: 0.2 | 0.7415 ± 0.0011 | 0.7440 ± 0.0015 | +0.0025 |
| by ratio: 0.3 | 0.7410 ± 0.0011 | 0.7434 ± 0.0010 | +0.0023 |
| by ratio: 0.4 | 0.7395 ± 0.0010 | 0.7419 ± 0.0016 | +0.0024 |
| by ratio: 0.5 | 0.7372 ± 0.0008 | 0.7392 ± 0.0011 | +0.0020 |
| by location: early | 0.7390 ± 0.0019 | 0.7418 ± 0.0008 | +0.0028 |
| by location: middle | 0.7413 ± 0.0003 | 0.7430 ± 0.0018 | +0.0017 |
| by location: late | 0.7399 ± 0.0014 | 0.7424 ± 0.0010 | +0.0025 |
| double stress: text+audio | 0.7370 ± 0.0014 | 0.7374 ± 0.0014 | +0.0004 |
| double stress: text+vision | 0.7371 ± 0.0018 | 0.7371 ± 0.0016 | +0.0001 |
| double stress: audio+vision | 0.7423 ± 0.0017 | 0.7458 ± 0.0022 | +0.0035 |

## Vision-all-zero subset

The 15-sample subset is diagnostic only; it does not change training or selection.

| Seed | Model | Split | Acc | Macro-F1 | MAE | Pearson |

|---:|---|---|---:|---:|---:|---:|
| 42 | B0 | clean | 0.5333 | 0.5556 | 0.5445 | 0.3371 |
| 42 | B0 | mean_missing | 0.5407 | 0.5563 | 0.5574 | 0.3297 |
| 42 | P2 | clean | 0.4667 | 0.4973 | 0.5888 | 0.2438 |
| 42 | P2 | mean_missing | 0.4531 | 0.4776 | 0.5925 | 0.2583 |
| 43 | B0 | clean | 0.5333 | 0.5556 | 0.5580 | 0.3699 |
| 43 | B0 | mean_missing | 0.5210 | 0.5343 | 0.5663 | 0.3589 |
| 43 | P2 | clean | 0.4667 | 0.4973 | 0.5721 | 0.4009 |
| 43 | P2 | mean_missing | 0.4716 | 0.4903 | 0.5759 | 0.3972 |
| 44 | B0 | clean | 0.4667 | 0.4444 | 0.5688 | 0.2905 |
| 44 | B0 | mean_missing | 0.4580 | 0.4364 | 0.5749 | 0.2855 |
| 44 | P2 | clean | 0.4667 | 0.4444 | 0.5695 | 0.2919 |
| 44 | P2 | mean_missing | 0.4580 | 0.4364 | 0.5753 | 0.2869 |

## Subgroup details

The per-seed B0/P2 subgroup scores and four metrics are preserved in JSON under `per_seed` and `seed43_44_details`. Each P2 seed also includes attention diagnostics.

## Stability interpretation

Seed-42 screening showed +0.005600. The seed43/44 paired deltas determine whether that gain replicates. The category (all positive / 2-of-3 / only seed42 positive) is filled from the observed paired deltas below.
B: two seeds are positive and one is negative; report seed sensitivity and compare the paired mean and negative-seed magnitude without overstating stability.
