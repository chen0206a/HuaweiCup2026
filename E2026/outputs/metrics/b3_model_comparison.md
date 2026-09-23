# Q2 B3 frozen validation comparison

B0-WCE, B2-B0 and B3-B0 all use training seed 42. Only B3 best robust checkpoint is compared.
Benchmark definition and SHA-256 are unchanged; test and attachment3 unused.

| Model | Clean Acc | Clean F1 | Clean MAE | Clean r | Missing Acc | Missing F1 | Missing MAE | Missing r | Robust score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-WCE | 0.6456 | 0.6040 | 0.6158 | 0.6368 | 0.6421 | 0.6016 | 0.6183 | 0.6320 | 0.7402 |
| B2-B0-BlockMask | 0.6360 | 0.6162 | 0.6232 | 0.6336 | 0.6295 | 0.6097 | 0.6252 | 0.6290 | 0.7393 |
| B3-B0-Reconstruction | 0.6181 | 0.6082 | 0.6245 | 0.6293 | 0.6119 | 0.6014 | 0.6263 | 0.6252 | 0.7323 |

## by_ratio

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | 0.1 | 0.6445 | 0.6024 | 0.6165 | 0.6359 | 0.7405 |
| B0-WCE | 0.2 | 0.6442 | 0.6026 | 0.6173 | 0.6349 | 0.7404 |
| B0-WCE | 0.3 | 0.6439 | 0.6037 | 0.6180 | 0.6330 | 0.7403 |
| B0-WCE | 0.4 | 0.6410 | 0.6011 | 0.6178 | 0.6318 | 0.7387 |
| B0-WCE | 0.5 | 0.6375 | 0.5981 | 0.6199 | 0.6274 | 0.7365 |
| B2-B0-BlockMask | 0.1 | 0.6346 | 0.6143 | 0.6238 | 0.6328 | 0.7403 |
| B2-B0-BlockMask | 0.2 | 0.6332 | 0.6128 | 0.6241 | 0.6317 | 0.7395 |
| B2-B0-BlockMask | 0.3 | 0.6308 | 0.6112 | 0.6247 | 0.6300 | 0.7382 |
| B2-B0-BlockMask | 0.4 | 0.6284 | 0.6090 | 0.6249 | 0.6289 | 0.7369 |
| B2-B0-BlockMask | 0.5 | 0.6239 | 0.6046 | 0.6276 | 0.6245 | 0.7340 |
| B3-B0-Reconstruction | 0.1 | 0.6152 | 0.6048 | 0.6253 | 0.6286 | 0.7325 |
| B3-B0-Reconstruction | 0.2 | 0.6142 | 0.6038 | 0.6256 | 0.6280 | 0.7319 |
| B3-B0-Reconstruction | 0.3 | 0.6136 | 0.6035 | 0.6257 | 0.6266 | 0.7315 |
| B3-B0-Reconstruction | 0.4 | 0.6093 | 0.5992 | 0.6248 | 0.6260 | 0.7293 |
| B3-B0-Reconstruction | 0.5 | 0.6090 | 0.5984 | 0.6280 | 0.6206 | 0.7283 |

## by_location

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | early | 0.6393 | 0.5984 | 0.6205 | 0.6293 | 0.7372 |
| B0-WCE | middle | 0.6441 | 0.6056 | 0.6145 | 0.6333 | 0.7410 |
| B0-WCE | late | 0.6429 | 0.6007 | 0.6198 | 0.6336 | 0.7393 |
| B2-B0-BlockMask | early | 0.6297 | 0.6103 | 0.6269 | 0.6258 | 0.7371 |
| B2-B0-BlockMask | middle | 0.6281 | 0.6093 | 0.6230 | 0.6302 | 0.7372 |
| B2-B0-BlockMask | late | 0.6306 | 0.6096 | 0.6257 | 0.6310 | 0.7379 |
| B3-B0-Reconstruction | early | 0.6087 | 0.5979 | 0.6299 | 0.6211 | 0.7280 |
| B3-B0-Reconstruction | middle | 0.6119 | 0.6022 | 0.6230 | 0.6260 | 0.7308 |
| B3-B0-Reconstruction | late | 0.6152 | 0.6043 | 0.6261 | 0.6284 | 0.7323 |

## by_modality

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | text | 0.6352 | 0.5955 | 0.6213 | 0.6247 | 0.7349 |
| B0-WCE | audio | 0.6460 | 0.6044 | 0.6161 | 0.6367 | 0.7415 |
| B0-WCE | vision | 0.6456 | 0.6048 | 0.6163 | 0.6364 | 0.7415 |
| B2-B0-BlockMask | text | 0.6224 | 0.6033 | 0.6268 | 0.6217 | 0.7330 |
| B2-B0-BlockMask | audio | 0.6359 | 0.6161 | 0.6234 | 0.6336 | 0.7412 |
| B2-B0-BlockMask | vision | 0.6322 | 0.6118 | 0.6249 | 0.6334 | 0.7392 |
| B3-B0-Reconstruction | text | 0.6011 | 0.5912 | 0.6257 | 0.6197 | 0.7245 |
| B3-B0-Reconstruction | audio | 0.6174 | 0.6078 | 0.6246 | 0.6293 | 0.7339 |
| B3-B0-Reconstruction | vision | 0.6182 | 0.6069 | 0.6273 | 0.6288 | 0.7338 |

## double_stress

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | text+audio | 0.6406 | 0.6022 | 0.6217 | 0.6260 | 0.7380 |
| B0-WCE | text+vision | 0.6392 | 0.6000 | 0.6224 | 0.6253 | 0.7370 |
| B0-WCE | audio+vision | 0.6442 | 0.6029 | 0.6165 | 0.6363 | 0.7406 |
| B2-B0-BlockMask | text+audio | 0.6250 | 0.6060 | 0.6263 | 0.6230 | 0.7345 |
| B2-B0-BlockMask | text+vision | 0.6218 | 0.6033 | 0.6272 | 0.6227 | 0.7330 |
| B2-B0-BlockMask | audio+vision | 0.6305 | 0.6100 | 0.6247 | 0.6334 | 0.7383 |
| B3-B0-Reconstruction | text+audio | 0.6026 | 0.5903 | 0.6293 | 0.6157 | 0.7240 |
| B3-B0-Reconstruction | text+vision | 0.6113 | 0.6001 | 0.6291 | 0.6205 | 0.7292 |
| B3-B0-Reconstruction | audio+vision | 0.6168 | 0.6064 | 0.6272 | 0.6280 | 0.7332 |

## Reconstruction diagnosis

| Modality | Masked positions | SmoothL1 | Cosine |
|---|---:|---:|---:|
| text | 117276 | 0.2224 | 0.5815 |
| audio | 117276 | 3.0729 | 0.8839 |
| vision | 117276 | 0.6902 | 0.4556 |

Detailed trigger counts and error by rho are in `b3_reconstruction_diagnostics.json`.
The 15 vision_all_zero validation samples are reported separately in `b3_summary.json`; no selection uses this subset.

## Same-checkpoint clean reconstruction ablation

This diagnostic disables reconstruction only at inference; it is not a checkpoint selection or training variant.
| State | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---:|---:|---:|---:|---:|
| enabled | 0.6181 | 0.6082 | 0.6245 | 0.6293 | 0.7342 |
| disabled | 0.6168 | 0.6072 | 0.6243 | 0.6296 | 0.7337 |
