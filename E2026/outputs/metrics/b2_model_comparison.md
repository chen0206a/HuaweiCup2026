# Q2 B2 fixed validation block benchmark

Project-internal score only; no attachment2 test or attachment3 used.
Fixed benchmark: `b2_benchmark_definition.json` (54 missing + clean).

| Model | Clean Acc | Clean F1 | Clean MAE | Clean r | Missing Acc | Missing F1 | Missing MAE | Missing r | Robust score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-WCE | 0.6456 | 0.6040 | 0.6158 | 0.6368 | 0.6421 | 0.6016 | 0.6183 | 0.6320 | 0.7402 |
| B1-WCE | 0.6236 | 0.6038 | 0.6145 | 0.6500 | 0.6119 | 0.5920 | 0.6213 | 0.6394 | 0.7338 |
| B2-B0-BlockMask | 0.6360 | 0.6162 | 0.6232 | 0.6336 | 0.6295 | 0.6097 | 0.6252 | 0.6290 | 0.7393 |
| B2-B1-BlockMask | 0.6195 | 0.6055 | 0.6011 | 0.6536 | 0.6160 | 0.6002 | 0.6093 | 0.6453 | 0.7361 |

Ratio and modality aggregates use the 45 single-modality scenes.
Location aggregates use all 54 missing scenes. Double stress uses the 9 two-modality scenes.
Averages are scenario means on the same 728 validation samples; no test split contributes.

## by_ratio

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | 0.1 | 0.6445 | 0.6024 | 0.6165 | 0.6359 | 0.7405 |
| B0-WCE | 0.2 | 0.6442 | 0.6026 | 0.6173 | 0.6349 | 0.7404 |
| B0-WCE | 0.3 | 0.6439 | 0.6037 | 0.6180 | 0.6330 | 0.7403 |
| B0-WCE | 0.4 | 0.6410 | 0.6011 | 0.6178 | 0.6318 | 0.7387 |
| B0-WCE | 0.5 | 0.6375 | 0.5981 | 0.6199 | 0.6274 | 0.7365 |
| B1-WCE | 0.1 | 0.6216 | 0.6020 | 0.6156 | 0.6486 | 0.7363 |
| B1-WCE | 0.2 | 0.6187 | 0.5992 | 0.6171 | 0.6457 | 0.7345 |
| B1-WCE | 0.3 | 0.6160 | 0.5961 | 0.6191 | 0.6428 | 0.7326 |
| B1-WCE | 0.4 | 0.6091 | 0.5886 | 0.6230 | 0.6372 | 0.7281 |
| B1-WCE | 0.5 | 0.6010 | 0.5812 | 0.6291 | 0.6274 | 0.7228 |
| B2-B0-BlockMask | 0.1 | 0.6346 | 0.6143 | 0.6238 | 0.6328 | 0.7403 |
| B2-B0-BlockMask | 0.2 | 0.6332 | 0.6128 | 0.6241 | 0.6317 | 0.7395 |
| B2-B0-BlockMask | 0.3 | 0.6308 | 0.6112 | 0.6247 | 0.6300 | 0.7382 |
| B2-B0-BlockMask | 0.4 | 0.6284 | 0.6090 | 0.6249 | 0.6289 | 0.7369 |
| B2-B0-BlockMask | 0.5 | 0.6239 | 0.6046 | 0.6276 | 0.6245 | 0.7340 |
| B2-B1-BlockMask | 0.1 | 0.6187 | 0.6046 | 0.6029 | 0.6527 | 0.7373 |
| B2-B1-BlockMask | 0.2 | 0.6174 | 0.6026 | 0.6051 | 0.6505 | 0.7361 |
| B2-B1-BlockMask | 0.3 | 0.6175 | 0.6022 | 0.6075 | 0.6478 | 0.7356 |
| B2-B1-BlockMask | 0.4 | 0.6146 | 0.5984 | 0.6101 | 0.6439 | 0.7333 |
| B2-B1-BlockMask | 0.5 | 0.6120 | 0.5947 | 0.6167 | 0.6348 | 0.7303 |

## by_location

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | early | 0.6393 | 0.5984 | 0.6205 | 0.6293 | 0.7372 |
| B0-WCE | middle | 0.6441 | 0.6056 | 0.6145 | 0.6333 | 0.7410 |
| B0-WCE | late | 0.6429 | 0.6007 | 0.6198 | 0.6336 | 0.7393 |
| B1-WCE | early | 0.6144 | 0.5943 | 0.6207 | 0.6380 | 0.7311 |
| B1-WCE | middle | 0.6103 | 0.5907 | 0.6214 | 0.6392 | 0.7293 |
| B1-WCE | late | 0.6109 | 0.5910 | 0.6217 | 0.6411 | 0.7297 |
| B2-B0-BlockMask | early | 0.6297 | 0.6103 | 0.6269 | 0.6258 | 0.7371 |
| B2-B0-BlockMask | middle | 0.6281 | 0.6093 | 0.6230 | 0.6302 | 0.7372 |
| B2-B0-BlockMask | late | 0.6306 | 0.6096 | 0.6257 | 0.6310 | 0.7379 |
| B2-B1-BlockMask | early | 0.6171 | 0.6017 | 0.6088 | 0.6434 | 0.7348 |
| B2-B1-BlockMask | middle | 0.6127 | 0.5965 | 0.6108 | 0.6442 | 0.7324 |
| B2-B1-BlockMask | late | 0.6181 | 0.6024 | 0.6084 | 0.6482 | 0.7358 |

## by_modality

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | text | 0.6352 | 0.5955 | 0.6213 | 0.6247 | 0.7349 |
| B0-WCE | audio | 0.6460 | 0.6044 | 0.6161 | 0.6367 | 0.7415 |
| B0-WCE | vision | 0.6456 | 0.6048 | 0.6163 | 0.6364 | 0.7415 |
| B1-WCE | text | 0.5964 | 0.5781 | 0.6242 | 0.6237 | 0.7206 |
| B1-WCE | audio | 0.6187 | 0.6013 | 0.6165 | 0.6482 | 0.7353 |
| B1-WCE | vision | 0.6248 | 0.6009 | 0.6217 | 0.6491 | 0.7367 |
| B2-B0-BlockMask | text | 0.6224 | 0.6033 | 0.6268 | 0.6217 | 0.7330 |
| B2-B0-BlockMask | audio | 0.6359 | 0.6161 | 0.6234 | 0.6336 | 0.7412 |
| B2-B0-BlockMask | vision | 0.6322 | 0.6118 | 0.6249 | 0.6334 | 0.7392 |
| B2-B1-BlockMask | text | 0.6070 | 0.5894 | 0.6195 | 0.6312 | 0.7272 |
| B2-B1-BlockMask | audio | 0.6193 | 0.6054 | 0.6019 | 0.6532 | 0.7378 |
| B2-B1-BlockMask | vision | 0.6219 | 0.6067 | 0.6040 | 0.6534 | 0.7387 |

## double_stress

| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |
|---|---|---:|---:|---:|---:|---:|
| B0-WCE | text+audio | 0.6406 | 0.6022 | 0.6217 | 0.6260 | 0.7380 |
| B0-WCE | text+vision | 0.6392 | 0.6000 | 0.6224 | 0.6253 | 0.7370 |
| B0-WCE | audio+vision | 0.6442 | 0.6029 | 0.6165 | 0.6363 | 0.7406 |
| B1-WCE | text+audio | 0.5943 | 0.5788 | 0.6215 | 0.6286 | 0.7210 |
| B1-WCE | text+vision | 0.6007 | 0.5782 | 0.6278 | 0.6288 | 0.7222 |
| B1-WCE | audio+vision | 0.6190 | 0.5984 | 0.6217 | 0.6476 | 0.7344 |
| B2-B0-BlockMask | text+audio | 0.6250 | 0.6060 | 0.6263 | 0.6230 | 0.7345 |
| B2-B0-BlockMask | text+vision | 0.6218 | 0.6033 | 0.6272 | 0.6227 | 0.7330 |
| B2-B0-BlockMask | audio+vision | 0.6305 | 0.6100 | 0.6247 | 0.6334 | 0.7383 |
| B2-B1-BlockMask | text+audio | 0.6103 | 0.5933 | 0.6178 | 0.6362 | 0.7297 |
| B2-B1-BlockMask | text+vision | 0.6154 | 0.5966 | 0.6201 | 0.6360 | 0.7317 |
| B2-B1-BlockMask | audio+vision | 0.6213 | 0.6069 | 0.6035 | 0.6533 | 0.7386 |
