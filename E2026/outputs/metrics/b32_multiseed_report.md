# B3.2 frozen reconstruction multi-seed stability

Seeds 42/43/44; benchmark seed 20260923 and SHA256 unchanged. Validation only; no test or attachment3 data.
All displayed uncertainty is mean ± sample standard deviation (ddof=1). Alpha is fixed globally, never selected per seed.

## Clean, mean-missing, robust

| Alpha | Clean Acc | Clean Macro-F1 | Clean MAE | Clean Pearson | Clean score | Missing Acc | Missing Macro-F1 | Missing MAE | Missing Pearson | Missing score | Robust score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.6401 ± 0.0048 | 0.6139 ± 0.0091 | 0.6079 ± 0.0103 | 0.6409 ± 0.0043 | 0.7433 ± 0.0018 | 0.6349 ± 0.0062 | 0.6092 ± 0.0075 | 0.6111 ± 0.0098 | 0.6359 ± 0.0038 | 0.7401 ± 0.0011 | 0.7417 ± 0.0015 |
| 0.25 | 0.6346 ± 0.0071 | 0.6107 ± 0.0039 | 0.6068 ± 0.0079 | 0.6408 ± 0.0035 | 0.7412 ± 0.0009 | 0.6317 ± 0.0064 | 0.6091 ± 0.0041 | 0.6097 ± 0.0074 | 0.6359 ± 0.0030 | 0.7393 ± 0.0011 | 0.7402 ± 0.0010 |
| 0.5 | 0.6332 ± 0.0073 | 0.6129 ± 0.0042 | 0.6060 ± 0.0064 | 0.6405 ± 0.0029 | 0.7413 ± 0.0023 | 0.6291 ± 0.0053 | 0.6093 ± 0.0014 | 0.6088 ± 0.0058 | 0.6354 ± 0.0023 | 0.7387 ± 0.0009 | 0.7400 ± 0.0016 |
| 0.75 | 0.6332 ± 0.0071 | 0.6147 ± 0.0029 | 0.6054 ± 0.0056 | 0.6401 ± 0.0021 | 0.7418 ± 0.0021 | 0.6276 ± 0.0048 | 0.6096 ± 0.0006 | 0.6085 ± 0.0047 | 0.6347 ± 0.0017 | 0.7383 ± 0.0010 | 0.7400 ± 0.0016 |
| 1.0 | 0.6346 ± 0.0082 | 0.6186 ± 0.0044 | 0.6052 ± 0.0051 | 0.6394 ± 0.0013 | 0.7430 ± 0.0030 | 0.6252 ± 0.0042 | 0.6092 ± 0.0012 | 0.6086 ± 0.0040 | 0.6336 ± 0.0012 | 0.7375 ± 0.0013 | 0.7402 ± 0.0021 |

## Paired deltas against alpha=0

Each cell is mean paired delta ± sample std; count is seeds whose delta improves the metric (MAE improves when negative).

| Alpha | Segment | Metric | Delta | Direction |
|---:|---|---|---:|---:|
| 0.25 | clean | accuracy | -0.0055 ± 0.0024 | 0/3 |
| 0.25 | clean | macro_f1 | -0.0032 ± 0.0057 | 1/3 |
| 0.25 | clean | mae | -0.0012 ± 0.0024 | 2/3 |
| 0.25 | clean | pearson | -0.0001 ± 0.0010 | 1/3 |
| 0.25 | clean | selection_score | -0.0021 ± 0.0022 | 1/3 |
| 0.25 | mean_missing | accuracy | -0.0031 ± 0.0005 | 0/3 |
| 0.25 | mean_missing | macro_f1 | -0.0001 ± 0.0044 | 1/3 |
| 0.25 | mean_missing | mae | -0.0015 ± 0.0024 | 2/3 |
| 0.25 | mean_missing | pearson | 0.0000 ± 0.0011 | 1/3 |
| 0.25 | mean_missing | selection_score | -0.0008 ± 0.0013 | 1/3 |
| 0.25 | robust_score | robust_score | -0.0014 ± 0.0018 | 1/3 |
| 0.5 | clean | accuracy | -0.0069 ± 0.0027 | 0/3 |
| 0.5 | clean | macro_f1 | -0.0010 ± 0.0104 | 1/3 |
| 0.5 | clean | mae | -0.0020 ± 0.0039 | 2/3 |
| 0.5 | clean | pearson | -0.0004 ± 0.0017 | 1/3 |
| 0.5 | clean | selection_score | -0.0019 ± 0.0036 | 1/3 |
| 0.5 | mean_missing | accuracy | -0.0057 ± 0.0013 | 0/3 |
| 0.5 | mean_missing | macro_f1 | 0.0000 ± 0.0066 | 1/3 |
| 0.5 | mean_missing | mae | -0.0023 ± 0.0041 | 2/3 |
| 0.5 | mean_missing | pearson | -0.0005 ± 0.0019 | 1/3 |
| 0.5 | mean_missing | selection_score | -0.0014 ± 0.0018 | 1/3 |
| 0.5 | robust_score | robust_score | -0.0017 ± 0.0026 | 1/3 |
| 0.75 | clean | accuracy | -0.0069 ± 0.0024 | 0/3 |
| 0.75 | clean | macro_f1 | 0.0008 ± 0.0108 | 1/3 |
| 0.75 | clean | mae | -0.0025 ± 0.0049 | 2/3 |
| 0.75 | clean | pearson | -0.0008 ± 0.0024 | 1/3 |
| 0.75 | clean | selection_score | -0.0015 ± 0.0037 | 1/3 |
| 0.75 | mean_missing | accuracy | -0.0073 ± 0.0018 | 0/3 |
| 0.75 | mean_missing | macro_f1 | 0.0003 ± 0.0075 | 1/3 |
| 0.75 | mean_missing | mae | -0.0027 ± 0.0052 | 2/3 |
| 0.75 | mean_missing | pearson | -0.0012 ± 0.0027 | 1/3 |
| 0.75 | mean_missing | selection_score | -0.0018 ± 0.0020 | 1/3 |
| 0.75 | robust_score | robust_score | -0.0016 ± 0.0028 | 1/3 |
| 1.0 | clean | accuracy | -0.0055 ± 0.0048 | 0/3 |
| 1.0 | clean | macro_f1 | 0.0047 ± 0.0132 | 2/3 |
| 1.0 | clean | mae | -0.0028 ± 0.0055 | 2/3 |
| 1.0 | clean | pearson | -0.0015 ± 0.0032 | 1/3 |
| 1.0 | clean | selection_score | -0.0003 ± 0.0048 | 1/3 |
| 1.0 | mean_missing | accuracy | -0.0097 ± 0.0033 | 0/3 |
| 1.0 | mean_missing | macro_f1 | -0.0000 ± 0.0086 | 2/3 |
| 1.0 | mean_missing | mae | -0.0025 ± 0.0062 | 2/3 |
| 1.0 | mean_missing | pearson | -0.0023 ± 0.0034 | 1/3 |
| 1.0 | mean_missing | selection_score | -0.0026 ± 0.0024 | 0/3 |
| 1.0 | robust_score | robust_score | -0.0014 ± 0.0036 | 1/3 |

## Per-modality mean-missing results

| Alpha | Modality | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---:|---|---:|---:|---:|---:|---:|
| 0.0 | text | 0.6268 ± 0.0073 | 0.6017 ± 0.0069 | 0.6158 ± 0.0094 | 0.6282 ± 0.0032 | 0.7350 ± 0.0009 |
| 0.0 | audio | 0.6400 ± 0.0051 | 0.6139 ± 0.0088 | 0.6083 ± 0.0103 | 0.6408 ± 0.0043 | 0.7433 ± 0.0017 |
| 0.0 | vision | 0.6388 ± 0.0059 | 0.6130 ± 0.0076 | 0.6080 ± 0.0099 | 0.6405 ± 0.0043 | 0.7427 ± 0.0012 |
| 0.25 | text | 0.6255 ± 0.0054 | 0.6028 ± 0.0053 | 0.6140 ± 0.0076 | 0.6285 ± 0.0024 | 0.7350 ± 0.0009 |
| 0.25 | audio | 0.6348 ± 0.0074 | 0.6112 ± 0.0030 | 0.6070 ± 0.0078 | 0.6409 ± 0.0038 | 0.7413 ± 0.0006 |
| 0.25 | vision | 0.6358 ± 0.0077 | 0.6136 ± 0.0048 | 0.6068 ± 0.0067 | 0.6403 ± 0.0030 | 0.7421 ± 0.0024 |
| 0.5 | text | 0.6240 ± 0.0039 | 0.6033 ± 0.0035 | 0.6124 ± 0.0062 | 0.6284 ± 0.0020 | 0.7349 ± 0.0003 |
| 0.5 | audio | 0.6324 ± 0.0075 | 0.6123 ± 0.0026 | 0.6063 ± 0.0063 | 0.6405 ± 0.0033 | 0.7410 ± 0.0019 |
| 0.5 | vision | 0.6327 ± 0.0051 | 0.6133 ± 0.0011 | 0.6067 ± 0.0046 | 0.6394 ± 0.0020 | 0.7411 ± 0.0012 |
| 0.75 | text | 0.6238 ± 0.0030 | 0.6049 ± 0.0022 | 0.6112 ± 0.0050 | 0.6283 ± 0.0016 | 0.7352 ± 0.0000 |
| 0.75 | audio | 0.6310 ± 0.0078 | 0.6127 ± 0.0025 | 0.6057 ± 0.0055 | 0.6401 ± 0.0026 | 0.7407 ± 0.0022 |
| 0.75 | vision | 0.6310 ± 0.0045 | 0.6133 ± 0.0009 | 0.6071 ± 0.0036 | 0.6382 ± 0.0015 | 0.7406 ± 0.0014 |
| 1.0 | text | 0.6207 ± 0.0012 | 0.6035 ± 0.0026 | 0.6107 ± 0.0040 | 0.6278 ± 0.0012 | 0.7341 ± 0.0007 |
| 1.0 | audio | 0.6319 ± 0.0093 | 0.6160 ± 0.0057 | 0.6055 ± 0.0050 | 0.6394 ± 0.0019 | 0.7417 ± 0.0035 |
| 1.0 | vision | 0.6282 ± 0.0044 | 0.6121 ± 0.0018 | 0.6080 ± 0.0034 | 0.6366 ± 0.0016 | 0.7393 ± 0.0017 |

## vision_all_zero subset

| Alpha | Segment | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---:|---|---:|---:|---:|---:|---:|
| 0.0 | clean | 0.5111 ± 0.0385 | 0.5185 ± 0.0642 | 0.5571 ± 0.0122 | 0.3325 ± 0.0399 | 0.6508 ± 0.0307 |
| 0.0 | mean_missing | 0.5066 ± 0.0432 | 0.5090 ± 0.0639 | 0.5662 ± 0.0088 | 0.3247 ± 0.0369 | 0.6459 ± 0.0310 |
| 0.25 | clean | 0.4000 ± 0.0667 | 0.3949 ± 0.0603 | 0.6342 ± 0.0184 | 0.2322 ± 0.0533 | 0.5763 ± 0.0293 |
| 0.25 | mean_missing | 0.4037 ± 0.0420 | 0.3855 ± 0.0494 | 0.6323 ± 0.0127 | 0.2391 ± 0.0410 | 0.5758 ± 0.0254 |
| 0.5 | clean | 0.4222 ± 0.0770 | 0.4064 ± 0.1218 | 0.6557 ± 0.0177 | 0.1875 ± 0.0517 | 0.5783 ± 0.0519 |
| 0.5 | mean_missing | 0.4049 ± 0.0680 | 0.3842 ± 0.0952 | 0.6524 ± 0.0122 | 0.1961 ± 0.0386 | 0.5696 ± 0.0428 |
| 0.75 | clean | 0.4222 ± 0.0770 | 0.4064 ± 0.1218 | 0.6609 ± 0.0177 | 0.1760 ± 0.0503 | 0.5766 ± 0.0522 |
| 0.75 | mean_missing | 0.4045 ± 0.0731 | 0.3851 ± 0.1019 | 0.6577 ± 0.0122 | 0.1836 ± 0.0379 | 0.5679 ± 0.0454 |
| 1.0 | clean | 0.4222 ± 0.0770 | 0.4064 ± 0.1218 | 0.6626 ± 0.0180 | 0.1714 ± 0.0496 | 0.5760 ± 0.0524 |
| 1.0 | mean_missing | 0.4091 ± 0.0713 | 0.3901 ± 0.1032 | 0.6598 ± 0.0123 | 0.1776 ± 0.0380 | 0.5695 ± 0.0455 |

## Per-seed scores

| Training seed | Alpha | Clean score | Mean-missing score | Robust score |
|---:|---:|---:|---:|---:|
| 42 | 0.00 | 0.7413 | 0.7392 | 0.7402 |
| 42 | 0.25 | 0.7418 | 0.7400 | 0.7409 |
| 42 | 0.50 | 0.7434 | 0.7397 | 0.7416 |
| 42 | 0.75 | 0.7441 | 0.7395 | 0.7418 |
| 42 | 1.00 | 0.7458 | 0.7386 | 0.7422 |
| 43 | 0.00 | 0.7436 | 0.7397 | 0.7416 |
| 43 | 0.25 | 0.7402 | 0.7380 | 0.7391 |
| 43 | 0.50 | 0.7389 | 0.7379 | 0.7384 |
| 43 | 0.75 | 0.7401 | 0.7377 | 0.7389 |
| 43 | 1.00 | 0.7435 | 0.7376 | 0.7406 |
| 44 | 0.00 | 0.7450 | 0.7413 | 0.7432 |
| 44 | 0.25 | 0.7415 | 0.7400 | 0.7407 |
| 44 | 0.50 | 0.7417 | 0.7383 | 0.7400 |
| 44 | 0.75 | 0.7411 | 0.7376 | 0.7393 |
| 44 | 1.00 | 0.7398 | 0.7361 | 0.7379 |

## Findings

1. No nonzero alpha improves robust score in all three seeds. Each tested nonzero alpha improves seed 42, but lowers robust score in seeds 43 and 44; their per-seed robust-optimal alpha is 0.
2. Mean-missing selection score does not improve across seeds: seeds 43 and 44 are below alpha=0 for every nonzero alpha, so the seed-42 gains do not reproduce. The three-seed mean-missing score is lower at every nonzero alpha.
3. Accuracy decreases for all seeds on both clean and mean-missing sets at every nonzero alpha. Macro-F1 is mixed. On mean-missing, MAE improves in seeds 42 and 44 but worsens in seed 43 at every nonzero alpha. Pearson improves only in seed 42 and declines in seeds 43/44.
4. Text missing has no stable gain. The mean text score changes only slightly (about 0.7350 at alpha=0 to at most 0.7352 at alpha=0.75), with mixed paired directions.
5. The vision_all_zero mean-missing subset is harmed in all three seeds at every nonzero alpha: Accuracy and Macro-F1 fall, MAE rises, and its selection score falls.
6. There is no globally defensible nonzero alpha across seeds. Keep alpha=0 / B0-WCE as the stable choice and terminate the reconstruction main line; do not proceed to B4.
