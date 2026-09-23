# Q2 B2.1 training-seed stability

Training seeds: 42, 43, 44. Sample standard deviation (ddof=1).
Frozen 54-scenario benchmark; attachment2 test and attachment3 unused.

| Model | Section | Accuracy | Macro-F1 | MAE | Pearson | Selection score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | clean | 0.6401 ± 0.0048 | 0.6139 ± 0.0091 | 0.6079 ± 0.0103 | 0.6409 ± 0.0043 | 0.7433 ± 0.0018 |
| B0-WCE | mean_missing | 0.6349 ± 0.0062 | 0.6092 ± 0.0075 | 0.6111 ± 0.0098 | 0.6359 ± 0.0038 | 0.7401 ± 0.0011 |
| B0-WCE | robust score | — | — | — | — | 0.7417 ± 0.0015 |
| B1-WCE | clean | 0.6200 ± 0.0042 | 0.6033 ± 0.0021 | 0.6043 ± 0.0089 | 0.6578 ± 0.0096 | 0.7379 ± 0.0021 |
| B1-WCE | mean_missing | 0.6102 ± 0.0017 | 0.5956 ± 0.0031 | 0.6068 ± 0.0126 | 0.6491 ± 0.0105 | 0.7323 ± 0.0023 |
| B1-WCE | robust score | — | — | — | — | 0.7351 ± 0.0020 |
| B2-B0-BlockMask | clean | 0.6305 ± 0.0073 | 0.6122 ± 0.0070 | 0.6120 ± 0.0139 | 0.6378 ± 0.0036 | 0.7399 ± 0.0035 |
| B2-B0-BlockMask | mean_missing | 0.6264 ± 0.0033 | 0.6084 ± 0.0024 | 0.6150 ± 0.0132 | 0.6328 ± 0.0034 | 0.7372 ± 0.0015 |
| B2-B0-BlockMask | robust score | — | — | — | — | 0.7385 ± 0.0025 |
| B2-B1-BlockMask | clean | 0.6241 ± 0.0057 | 0.5982 ± 0.0131 | 0.6087 ± 0.0082 | 0.6529 ± 0.0028 | 0.7368 ± 0.0019 |
| B2-B1-BlockMask | mean_missing | 0.6191 ± 0.0049 | 0.5922 ± 0.0137 | 0.6142 ± 0.0090 | 0.6452 ± 0.0014 | 0.7329 ± 0.0024 |
| B2-B1-BlockMask | robust score | — | — | — | — | 0.7348 ± 0.0022 |

## Paired score differences

| Contrast | Score | Per-seed deltas | Same direction? |
|---|---|---|---|
| B2-B0 minus B0-WCE | clean | 42: -0.0001, 43: -0.0010, 44: -0.0091 | True |
| B2-B0 minus B0-WCE | mean_missing | 42: -0.0018, 43: -0.0011, 44: -0.0058 | True |
| B2-B0 minus B0-WCE | robust_score | 42: -0.0009, 43: -0.0011, 44: -0.0074 | True |
| B2-B1 minus B1-WCE | clean | 42: +0.0004, 43: +0.0021, 44: -0.0056 | False |
| B2-B1 minus B1-WCE | mean_missing | 42: +0.0043, 43: +0.0018, 44: -0.0046 | False |
| B2-B1 minus B1-WCE | robust_score | 42: +0.0023, 43: +0.0020, 44: -0.0051 | False |
| B0 main line minus B1 main line | clean | 42: +0.0034, 43: +0.0045, 44: +0.0013 | True |
| B0 main line minus B1 main line | mean_missing | 42: +0.0031, 43: +0.0043, 44: +0.0055 | True |
| B0 main line minus B1 main line | robust_score | 42: +0.0032, 43: +0.0044, 44: +0.0034 | True |
| B0-WCE minus B1-WCE | clean | 42: +0.0038, 43: +0.0076, 44: +0.0048 | True |
| B0-WCE minus B1-WCE | mean_missing | 42: +0.0091, 43: +0.0073, 44: +0.0068 | True |
| B0-WCE minus B1-WCE | robust_score | 42: +0.0065, 43: +0.0075, 44: +0.0058 | True |

Individual model/seed scores and best epochs: `b21_multiseed_runs.csv`.
