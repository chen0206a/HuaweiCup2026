# Q2 expanded public architecture comparison

All values below are measured on Attachment2 validation (aligned_50.pkl SHA256 `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`). Twelve published model mechanisms were adapted to the aligned-50 interface and evaluated with the common classification/regression heads. They are **not** complete replications of each paper's original training pipeline. B0 and P2 use the existing CleanSelect checkpoint evaluation. Every seed is 42, 43, or 44; sample SD uses ddof=1. The 54 scenarios use the frozen SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`.

## Clean validation

| Model | Family | Params | Accuracy | Macro-F1 | MAE | Pearson |
|---|---|---:|---:|---:|---:|---:|
| TFN | standard fusion | 4,840,670 | 0.6099 ± 0.0076 | 0.5872 ± 0.0083 | 0.6336 ± 0.0102 | 0.6056 ± 0.0098 |
| LMF | standard fusion | 323,276 | 0.5925 ± 0.0114 | 0.5719 ± 0.0090 | 0.6682 ± 0.0196 | 0.5807 ± 0.0188 |
| MFN | standard fusion | 261,764 | 0.5966 ± 0.0080 | 0.5824 ± 0.0062 | 0.6418 ± 0.0124 | 0.5908 ± 0.0125 |
| MulT | standard fusion | 624,094 | 0.6172 ± 0.0148 | 0.5891 ± 0.0236 | 0.6221 ± 0.0260 | 0.6369 ± 0.0115 |
| MISA | standard fusion | 1,118,852 | 0.6131 ± 0.0083 | 0.6010 ± 0.0060 | 0.9175 ± 0.1005 | 0.6269 ± 0.0087 |
| Self-MM | standard fusion | 82,119 | 0.6355 ± 0.0052 | 0.6107 ± 0.0084 | 0.6099 ± 0.0066 | 0.6423 ± 0.0042 |
| MMIM | standard fusion | 163,204 | 0.6360 ± 0.0048 | 0.6110 ± 0.0007 | 0.6102 ± 0.0098 | 0.6375 ± 0.0054 |
| MAG-BERT | standard fusion | 477,188 | 0.6364 ± 0.0097 | 0.6172 ± 0.0072 | 0.6012 ± 0.0110 | 0.6672 ± 0.0035 |
| TFR-Net | missing-aware | 415,620 | 0.6374 ± 0.0071 | 0.6230 ± 0.0080 | 0.6145 ± 0.0118 | 0.6267 ± 0.0142 |
| MissModal | missing-aware | 163,204 | 0.6305 ± 0.0071 | 0.6108 ± 0.0069 | 0.6084 ± 0.0113 | 0.6431 ± 0.0006 |
| M3S | missing-aware | 323,276 | 0.6035 ± 0.0063 | 0.5775 ± 0.0169 | 0.6706 ± 0.0173 | 0.5793 ± 0.0171 |
| MMIN | missing-aware | 292,164 | 0.6383 ± 0.0062 | 0.6117 ± 0.0079 | 0.6279 ± 0.0247 | 0.6401 ± 0.0027 |
| B0 | project baseline | 163,460 | 0.6401 ± 0.0048 | 0.6139 ± 0.0091 | 0.6079 ± 0.0103 | 0.6409 ± 0.0043 |
| P2 | project model | 164,343 | 0.6415 ± 0.0050 | 0.6243 ± 0.0026 | 0.6052 ± 0.0063 | 0.6475 ± 0.0024 |

## Frozen 54-scenario mean

| Model | Training regime | Missing Accuracy | Missing Macro-F1 | Missing MAE | Missing Pearson |
|---|---|---:|---:|---:|---:|
| TFN | standard clean training | 0.5767 ± 0.0197 | 0.5582 ± 0.0127 | 0.6451 ± 0.0101 | 0.5784 ± 0.0070 |
| LMF | standard clean training | 0.5616 ± 0.0054 | 0.5392 ± 0.0066 | 0.6630 ± 0.0123 | 0.5539 ± 0.0185 |
| MFN | standard clean training | 0.5657 ± 0.0070 | 0.5532 ± 0.0081 | 0.6559 ± 0.0097 | 0.5540 ± 0.0092 |
| MulT | standard clean training | 0.6000 ± 0.0209 | 0.5749 ± 0.0230 | 0.6309 ± 0.0189 | 0.6232 ± 0.0057 |
| MISA | standard clean training | 0.6079 ± 0.0095 | 0.5958 ± 0.0080 | 0.9247 ± 0.1025 | 0.6198 ± 0.0077 |
| Self-MM | standard clean training | 0.6302 ± 0.0016 | 0.6053 ± 0.0042 | 0.6131 ± 0.0068 | 0.6371 ± 0.0040 |
| MMIM | standard clean training | 0.6302 ± 0.0041 | 0.6059 ± 0.0004 | 0.6135 ± 0.0101 | 0.6332 ± 0.0051 |
| MAG-BERT | standard clean training | 0.6159 ± 0.0065 | 0.5950 ± 0.0081 | 0.6150 ± 0.0055 | 0.6474 ± 0.0008 |
| TFR-Net | missing-aware training | 0.6336 ± 0.0099 | 0.6154 ± 0.0098 | 0.6232 ± 0.0146 | 0.6220 ± 0.0143 |
| MissModal | missing-aware training | 0.6260 ± 0.0096 | 0.6068 ± 0.0048 | 0.6111 ± 0.0112 | 0.6381 ± 0.0013 |
| M3S | missing-aware training | 0.5733 ± 0.0056 | 0.5493 ± 0.0102 | 0.6620 ± 0.0071 | 0.5581 ± 0.0192 |
| MMIN | missing-aware training | 0.6341 ± 0.0049 | 0.6072 ± 0.0039 | 0.6318 ± 0.0259 | 0.6346 ± 0.0037 |
| B0 | standard clean training | 0.6349 ± 0.0062 | 0.6092 ± 0.0075 | 0.6111 ± 0.0098 | 0.6359 ± 0.0038 |
| P2 | standard clean training | 0.6334 ± 0.0038 | 0.6163 ± 0.0013 | 0.6086 ± 0.0059 | 0.6417 ± 0.0024 |

## Interpretation

P2 rank among 14 models (Accuracy / Macro-F1 / MAE / Pearson): clean 1/1/2/2, missing 4/1/1/2. MAE ranks lower values first.
P2 uses 164,343 parameters; the median adapted public mechanism has 323,276, or 1.97× as many. Parameter efficiency is descriptive and does not remove regime or architecture differences.

The specialized missing-aware entries received synthetic missing during training; standard fusion entries were trained on clean samples. Both groups were selected on clean validation and evaluated on identical frozen missing scenarios. This regime difference is retained in the tables.
Across adapted public mechanisms, missing Accuracy spans 0.5616–0.6302 for standard clean training and 0.5733–0.6341 for missing-aware training. The ranges overlap; these model families also differ in architecture and cannot isolate a training-regime effect.

Text-missing sensitivity (clean minus 15 text-only scenario mean Accuracy), all 14 models:
- MFN: +0.0739 ± 0.0042
- LMF: +0.0726 ± 0.0174
- M3S: +0.0708 ± 0.0201
- TFN: +0.0591 ± 0.0197
- MulT: +0.0389 ± 0.0138
- MAG-BERT: +0.0292 ± 0.0043
- P2: +0.0194 ± 0.0036
- MISA: +0.0143 ± 0.0035
- Self-MM: +0.0137 ± 0.0078
- MMIM: +0.0137 ± 0.0001
- B0: +0.0133 ± 0.0025
- MMIN: +0.0112 ± 0.0058
- MissModal: +0.0107 ± 0.0075
- TFR-Net: +0.0107 ± 0.0060

Training time is recorded per seed in the seedwise CSV. P2 time covers residual training with a pre-existing frozen B0 and is labeled separately; it is not a from-scratch time comparison.

No Attachment2 test or Attachment3/4 results were used for model selection. Source details and implementation departures appear in the registry and per-model adaptation notes.
