# Q2 B4' reliability-aware dynamic fusion

Validation uses the unchanged B2 benchmark (54 missing scenarios + clean); seed 42 only.
All B0 prediction parameters are frozen. Gate input uses only current observations and padding.

## Main comparison

| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Robust score |
|---|---|---:|---:|---:|---:|---:|---:|
| B0-WCE | clean | 0.6456 | 0.6040 | 0.6158 | 0.6368 | 0.7413 | 0.7402 |
| B0-WCE | mean_missing | 0.6421 | 0.6016 | 0.6183 | 0.6320 | 0.7392 | 0.7402 |
| B4' | clean | 0.6497 | 0.6148 | 0.6159 | 0.6373 | 0.7451 | 0.7440 |
| B4' | mean_missing | 0.6462 | 0.6119 | 0.6182 | 0.6324 | 0.7428 | 0.7440 |

## Missing scene groups

### by_modality

| Group | Accuracy | Macro-F1 | MAE | Pearson | Score | Score Δ vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.6374 | 0.6022 | 0.6216 | 0.6248 | 0.7371 | +0.0022 |
| audio | 0.6501 | 0.6152 | 0.6163 | 0.6372 | 0.7453 | +0.0038 |
| vision | 0.6525 | 0.6193 | 0.6159 | 0.6369 | 0.7469 | +0.0054 |

### by_ratio

| Group | Accuracy | Macro-F1 | MAE | Pearson | Score | Score Δ vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.6493 | 0.6134 | 0.6166 | 0.6364 | 0.7445 | +0.0040 |
| 0.2 | 0.6488 | 0.6139 | 0.6172 | 0.6354 | 0.7444 | +0.0040 |
| 0.3 | 0.6482 | 0.6145 | 0.6178 | 0.6334 | 0.7441 | +0.0038 |
| 0.4 | 0.6450 | 0.6110 | 0.6179 | 0.6321 | 0.7423 | +0.0035 |
| 0.5 | 0.6419 | 0.6084 | 0.6200 | 0.6276 | 0.7402 | +0.0037 |

### by_location

| Group | Accuracy | Macro-F1 | MAE | Pearson | Score | Score Δ vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| early | 0.6441 | 0.6089 | 0.6199 | 0.6299 | 0.7412 | +0.0039 |
| middle | 0.6483 | 0.6157 | 0.6142 | 0.6335 | 0.7446 | +0.0036 |
| late | 0.6463 | 0.6111 | 0.6205 | 0.6337 | 0.7427 | +0.0034 |

### double_stress

| Group | Accuracy | Macro-F1 | MAE | Pearson | Score | Score Δ vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| text+audio | 0.6397 | 0.6057 | 0.6214 | 0.6260 | 0.7387 | +0.0007 |
| text+vision | 0.6387 | 0.6039 | 0.6217 | 0.6253 | 0.7379 | +0.0009 |
| audio+vision | 0.6538 | 0.6209 | 0.6162 | 0.6368 | 0.7476 | +0.0070 |

## Gate distributions

| Group | Gate | Mean | Sample std | P05 | P25 | P50 | P75 | P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean | text | 1.1014 | 0.0519 | 1.0317 | 1.0616 | 1.0932 | 1.1381 | 1.1942 |
| clean | audio | 1.0786 | 0.0669 | 0.9670 | 1.0316 | 1.0769 | 1.1279 | 1.1912 |
| clean | vision | 0.7944 | 0.1035 | 0.6100 | 0.7174 | 0.8099 | 0.8732 | 0.9470 |
| mean_missing | text | 1.1022 | 0.0519 | 1.0308 | 1.0628 | 1.0936 | 1.1377 | 1.1957 |
| mean_missing | audio | 1.0775 | 0.0666 | 0.9657 | 1.0295 | 1.0764 | 1.1260 | 1.1899 |
| mean_missing | vision | 0.7942 | 0.1034 | 0.6107 | 0.7183 | 0.8073 | 0.8729 | 0.9472 |
| text_missing | text | 1.1037 | 0.0520 | 1.0321 | 1.0646 | 1.0951 | 1.1394 | 1.1972 |
| text_missing | audio | 1.0775 | 0.0663 | 0.9662 | 1.0294 | 1.0765 | 1.1260 | 1.1887 |
| text_missing | vision | 0.7923 | 0.1035 | 0.6082 | 0.7158 | 0.8059 | 0.8710 | 0.9458 |
| audio_missing | text | 1.1017 | 0.0519 | 1.0318 | 1.0620 | 1.0936 | 1.1384 | 1.1952 |
| audio_missing | audio | 1.0784 | 0.0669 | 0.9667 | 1.0312 | 1.0767 | 1.1278 | 1.1913 |
| audio_missing | vision | 0.7941 | 0.1036 | 0.6083 | 0.7173 | 0.8094 | 0.8733 | 0.9468 |
| vision_missing | text | 1.1009 | 0.0518 | 1.0291 | 1.0615 | 1.0926 | 1.1357 | 1.1942 |
| vision_missing | audio | 1.0768 | 0.0666 | 0.9652 | 1.0287 | 1.0759 | 1.1250 | 1.1894 |
| vision_missing | vision | 0.7964 | 0.1030 | 0.6133 | 0.7226 | 0.8084 | 0.8748 | 0.9490 |
| text+audio | text | 1.1035 | 0.0520 | 1.0322 | 1.0648 | 1.0945 | 1.1389 | 1.1966 |
| text+audio | audio | 1.0776 | 0.0665 | 0.9659 | 1.0294 | 1.0764 | 1.1259 | 1.1882 |
| text+audio | vision | 0.7924 | 0.1036 | 0.6077 | 0.7156 | 0.8063 | 0.8712 | 0.9461 |
| text+vision | text | 1.1029 | 0.0518 | 1.0312 | 1.0642 | 1.0944 | 1.1378 | 1.1963 |
| text+vision | audio | 1.0763 | 0.0663 | 0.9655 | 1.0287 | 1.0755 | 1.1241 | 1.1876 |
| text+vision | vision | 0.7943 | 0.1030 | 0.6139 | 0.7212 | 0.8059 | 0.8728 | 0.9464 |
| audio+vision | text | 1.1015 | 0.0519 | 1.0299 | 1.0623 | 1.0933 | 1.1367 | 1.1945 |
| audio+vision | audio | 1.0770 | 0.0668 | 0.9648 | 1.0286 | 1.0760 | 1.1258 | 1.1899 |
| audio+vision | vision | 0.7956 | 0.1033 | 0.6155 | 0.7214 | 0.8068 | 0.8738 | 0.9484 |

## Damaged modality gate by rho

| Modality | Rho | Mean gate | Sample std |
|---|---:|---:|---:|
| text | 0.1 | 1.1010 | 0.0517 |
| text | 0.2 | 1.1018 | 0.0517 |
| text | 0.3 | 1.1031 | 0.0519 |
| text | 0.4 | 1.1050 | 0.0522 |
| text | 0.5 | 1.1073 | 0.0525 |
| audio | 0.1 | 1.0785 | 0.0669 |
| audio | 0.2 | 1.0785 | 0.0669 |
| audio | 0.3 | 1.0784 | 0.0669 |
| audio | 0.4 | 1.0784 | 0.0669 |
| audio | 0.5 | 1.0782 | 0.0670 |
| vision | 0.1 | 0.7947 | 0.1033 |
| vision | 0.2 | 0.7951 | 0.1033 |
| vision | 0.3 | 0.7959 | 0.1031 |
| vision | 0.4 | 0.7972 | 0.1029 |
| vision | 0.5 | 0.7988 | 0.1022 |

## vision_all_zero subset

Count: 15

| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|
| B0-WCE | clean | 0.5333 | 0.5556 | 0.5445 | 0.3371 |
| B0-WCE | mean_missing | 0.5407 | 0.5563 | 0.5574 | 0.3297 |
| B4' | clean | 0.5333 | 0.5556 | 0.5564 | 0.3237 |
| B4' | mean_missing | 0.5346 | 0.5514 | 0.5666 | 0.3166 |

### vision_all_zero gates: clean

| Gate | Mean | Sample std | P05 | P25 | P50 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 1.0805 | 0.0100 | 1.0644 | 1.0742 | 1.0798 | 1.0882 | 1.0935 |
| audio | 1.0192 | 0.0464 | 0.9565 | 0.9806 | 1.0220 | 1.0499 | 1.0862 |
| vision | 0.8714 | 0.0387 | 0.8185 | 0.8398 | 0.8822 | 0.8981 | 0.9301 |

### vision_all_zero gates: mean_missing

| Gate | Mean | Sample std | P05 | P25 | P50 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 1.0812 | 0.0113 | 1.0633 | 1.0732 | 1.0801 | 1.0893 | 1.0974 |
| audio | 1.0186 | 0.0445 | 0.9547 | 0.9773 | 1.0218 | 1.0529 | 1.1110 |
| vision | 0.8709 | 0.0382 | 0.8122 | 0.8371 | 0.8781 | 0.9004 | 0.9323 |

## Verification and timing

Identity with B0: True; frozen tensor equality: True; cached/full validation: True.
Parameters: 188,743 total, 25,283 trainable.
Epoch: 2 best robust, 1 best clean, 12 run.
Training 25.05s; validation cache setup 4.07s; total cached validation 9.05s; mean/epoch 0.75s; normal verification 4.80s.
