# B5-N1 text-only train-stat feature-wise Z-score

Seed42 only. N0 reuses the historical B0-WCE checkpoint; N1 fully retrains B0 with text-only normalization.
Validation only: clean + frozen 54 scenarios (seed 20260923). The monolithic pickle container is deserialized, but its test key is not indexed, made into a Dataset, evaluated, or used for any decision; attachment3 is not accessed.

Benchmark SHA256: `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`. Synthetic zeroing order: `raw feature -> train-stat transform -> apply_blocks overwrite to exact zero`.

## N0 / N1 main comparison

| Model/checkpoint | Condition | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Robust score |

|---|---|---:|---:|---:|---:|---:|---:|
| N0 B0-WCE historical | clean | 0.6456 | 0.6040 | 0.6158 | 0.6368 | 0.7413 | 0.7402 |
| N0 B0-WCE historical | mean_missing | 0.6421 | 0.6016 | 0.6183 | 0.6320 | 0.7392 | 0.7402 |
| N1 best-robust | clean | 0.6195 | 0.6113 | 0.6081 | 0.6297 | 0.7361 | 0.7345 |
| N1 best-robust | mean_missing | 0.6141 | 0.6064 | 0.6107 | 0.6256 | 0.7329 | 0.7345 |

Δrobust (N1−N0) = -0.005774; decision: **stop_normalization_route**.
N1 best-clean epoch 1, best-robust epoch 1; training time 69.6s.

## Neutral recall / F1

| Model | Condition | Recall | F1 |

|---|---|---:|---:|
| N0 | clean | 0.3315 | 0.4136 |
| N0 | mean_missing | 0.3370 | 0.4128 |
| N1 | clean | 0.5707 | 0.5036 |
| N1 | mean_missing | 0.5806 | 0.5027 |

## Missing subgroup comparison

Each value below is a scenario mean. Per-class full scenario metrics remain in JSON.

| Group | Model | Acc | Macro-F1 | MAE | Pearson | Score | Neutral recall | Neutral F1 |

|---|---|---:|---:|---:|---:|---:|---:|---:|
| modality: text | N0 | 0.6352 | 0.5955 | 0.6213 | 0.6247 | 0.7349 | 0.3384 | 0.4089 |
| modality: text | N1 | 0.6091 | 0.6018 | 0.6139 | 0.6193 | 0.7295 | 0.5862 | 0.5015 |
| modality: audio | N0 | 0.6460 | 0.6044 | 0.6161 | 0.6367 | 0.7415 | 0.3315 | 0.4135 |
| modality: audio | N1 | 0.6191 | 0.6109 | 0.6082 | 0.6297 | 0.7359 | 0.5707 | 0.5033 |
| modality: vision | N0 | 0.6456 | 0.6048 | 0.6163 | 0.6364 | 0.7415 | 0.3384 | 0.4157 |
| modality: vision | N1 | 0.6141 | 0.6066 | 0.6092 | 0.6292 | 0.7334 | 0.5815 | 0.5029 |
| rho: 0.1 | N0 | 0.6445 | 0.6024 | 0.6165 | 0.6359 | 0.7405 | 0.3291 | 0.4098 |
| rho: 0.1 | N1 | 0.6184 | 0.6103 | 0.6089 | 0.6289 | 0.7354 | 0.5719 | 0.5037 |
| rho: 0.2 | N0 | 0.6442 | 0.6026 | 0.6173 | 0.6349 | 0.7404 | 0.3327 | 0.4119 |
| rho: 0.2 | N1 | 0.6168 | 0.6083 | 0.6094 | 0.6279 | 0.7344 | 0.5731 | 0.5022 |
| rho: 0.3 | N0 | 0.6439 | 0.6037 | 0.6180 | 0.6330 | 0.7403 | 0.3388 | 0.4150 |
| rho: 0.3 | N1 | 0.6157 | 0.6079 | 0.6100 | 0.6264 | 0.7338 | 0.5797 | 0.5032 |
| rho: 0.4 | N0 | 0.6410 | 0.6011 | 0.6178 | 0.6318 | 0.7387 | 0.3388 | 0.4138 |
| rho: 0.4 | N1 | 0.6113 | 0.6039 | 0.6106 | 0.6253 | 0.7315 | 0.5833 | 0.5021 |
| rho: 0.5 | N0 | 0.6375 | 0.5981 | 0.6199 | 0.6274 | 0.7365 | 0.3412 | 0.4129 |
| rho: 0.5 | N1 | 0.6084 | 0.6016 | 0.6132 | 0.6219 | 0.7297 | 0.5894 | 0.5015 |
| location: early | N0 | 0.6393 | 0.5984 | 0.6205 | 0.6293 | 0.7372 | 0.3330 | 0.4075 |
| location: early | N1 | 0.6200 | 0.6112 | 0.6125 | 0.6221 | 0.7350 | 0.5833 | 0.5094 |
| location: middle | N0 | 0.6441 | 0.6056 | 0.6145 | 0.6333 | 0.7410 | 0.3518 | 0.4251 |
| location: middle | N1 | 0.6062 | 0.6001 | 0.6084 | 0.6274 | 0.7296 | 0.5882 | 0.4983 |
| location: late | N0 | 0.6429 | 0.6007 | 0.6198 | 0.6336 | 0.7393 | 0.3261 | 0.4058 |
| location: late | N1 | 0.6160 | 0.6081 | 0.6111 | 0.6272 | 0.7339 | 0.5704 | 0.5003 |
| double: text+audio | N0 | 0.6406 | 0.6022 | 0.6217 | 0.6260 | 0.7380 | 0.3460 | 0.4163 |
| double: text+audio | N1 | 0.6145 | 0.6072 | 0.6127 | 0.6202 | 0.7324 | 0.5888 | 0.5038 |
| double: text+vision | N0 | 0.6392 | 0.6000 | 0.6224 | 0.6253 | 0.7370 | 0.3424 | 0.4121 |
| double: text+vision | N1 | 0.6140 | 0.6071 | 0.6136 | 0.6197 | 0.7322 | 0.5906 | 0.5037 |
| double: audio+vision | N0 | 0.6442 | 0.6029 | 0.6165 | 0.6363 | 0.7406 | 0.3351 | 0.4116 |
| double: audio+vision | N1 | 0.6131 | 0.6053 | 0.6094 | 0.6291 | 0.7328 | 0.5797 | 0.5023 |

## Vision-all-zero subset (15 validation samples)

Diagnostic only; it does not alter training or selection.

| Model | Split | Acc | Macro-F1 | MAE | Pearson |

|---|---|---:|---:|---:|---:|
| N0 | clean | 0.5333 | 0.5556 | 0.5445 | 0.3371 |
| N0 | mean_missing | 0.5407 | 0.5563 | 0.5574 | 0.3297 |
| N1 | clean | 0.4667 | 0.4519 | 0.5487 | 0.4512 |
| N1 | mean_missing | 0.4494 | 0.4265 | 0.5563 | 0.4453 |

## Text scaler diagnostics

Fit split `train`, modality `text`, feature_dim=768, valid timesteps=83672, epsilon=1e-06.

Raw per-feature mean range [-3.54528, 0.708479]; raw std range [0.33875, 2.21308]. Zero/nearly-zero std dimensions: 0/0.

Train z-score per-feature mean |.| median/max = 3.07e-17/1.57e-16; nondegenerate std deviation from 1 median/max = 2.22e-16/7.77e-16.

Std quantiles are in the machine-readable metrics and exact 768-element scaler state is in `b5_n1_text_scaler_state.json`.

## Projection weight norms

| Modality | L2 norm of trained input projection weights |

|---|---:|
| text | 6.840378 |
| audio | 6.514998 |
| vision | 6.586823 |

## Checks

- N0 historical checkpoint reproduction: True; per-item results are in JSON.
- N0/N1 initial state equal: True (`6545bc82c1dc02051460e3f87dbf6b2a562f50ce6fdd03191ac24c6455fb4989`).
- N0/N1 train sample order hash matches: True (`803756f149492d081bb82381f4b2123df46387c3202686ef44857d081dfb6cd0`).
- Text scaler fit is train-only; valid positions counted: 83672.
- Padding mask remains unchanged and padded positions are excluded by masked pooling.
- Audio/vision unchanged; synthetic missing overwritten to exact zero after normalization: True.
- CPU/CUDA smoke: {'cpu': 'forward_backward_pass', 'cuda': 'forward_backward_pass'}; robust and clean checkpoint round-trip checks passed.
- The monolithic pickle container is deserialized to select train/valid. Its test key is not indexed or used; no test Dataset/evaluation is performed. Attachment3 is not accessed.
