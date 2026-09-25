# Q2 public baseline comparison

Attachment2 aligned-50 train (3395) and valid (728) only. Seeds 42/43/44; sample SD. The TFN, MulT, and MISA results were newly trained; B0-WCE and P2 are historical locked references.

| Model | Params | Clean Acc | Clean Macro-F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson | Robust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TFN | 4840670 | 0.6099 ± 0.0076 | 0.5872 ± 0.0083 | 0.6336 ± 0.0102 | 0.6056 ± 0.0098 | 0.5767 ± 0.0197 | 0.5582 ± 0.0127 | 0.6451 ± 0.0101 | 0.5784 ± 0.0070 | 0.7139 ± 0.0039 |
| MulT | 624094 | 0.6172 ± 0.0148 | 0.5891 ± 0.0236 | 0.6221 ± 0.0260 | 0.6369 ± 0.0115 | 0.6000 ± 0.0209 | 0.5749 ± 0.0230 | 0.6309 ± 0.0189 | 0.6232 ± 0.0057 | 0.7253 ± 0.0061 |
| MISA | 1118852 | 0.6131 ± 0.0083 | 0.6010 ± 0.0060 | 0.9175 ± 0.1005 | 0.6269 ± 0.0087 | 0.6079 ± 0.0095 | 0.5958 ± 0.0080 | 0.9247 ± 0.1025 | 0.6198 ± 0.0077 | 0.7168 ± 0.0019 |
| Historical B0-WCE | see lock manifest | 0.6401 ± 0.0048 | 0.6139 ± 0.0091 | 0.6079 ± 0.0103 | 0.6409 ± 0.0043 | 0.6349 ± 0.0062 | 0.6092 ± 0.0075 | 0.6111 ± 0.0098 | 0.6359 ± 0.0038 | 0.7417 ± 0.0015 |
| Locked Q2 attention residual | see lock manifest | 0.6419 ± 0.0057 | 0.6241 ± 0.0026 | 0.6059 ± 0.0066 | 0.6472 ± 0.0025 | 0.6340 ± 0.0049 | 0.6163 ± 0.0012 | 0.6091 ± 0.0060 | 0.6414 ± 0.0025 | 0.7448 ± 0.0015 |

Baseline checkpoint selection used the clean-validation project score. The historical B0-WCE checkpoint was also clean-score selected; the historical P2 checkpoint was robust-score selected. Thus the P2 missing-score comparison has a checkpoint-selection advantage and should be read descriptively. No test-based model choice occurred. Missing evaluation uses the unchanged 54-scenario benchmark definition (SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`).

## Source and adaptation

- TFN: [reference repo](https://github.com/Justin1904/TensorFusionNetworks) `ef0e78b5583159de9b74ef2cdef6031bd9f94b37`; independently implemented tensor outer-product fusion from the paper. The inspected repo has no LICENSE file, so no source was copied. Text uses a packed LSTM; audio and vision use valid-step mean for utterance input.
- MulT: [official repo](https://github.com/yaohungt/Multimodal-Transformer) `a670936824ee722c8494fd98d204977a1d663c7a` (MIT); six directed crossmodal attention streams and per-modality memory with sinusoidal positions; padding is excluded from attention keys and final valid-step selection.
- MISA: [official repo](https://github.com/declare-lab/MISA) `ec42faddde0d210cf7368aebf2118fe9570e7102` (MIT); shared/private decomposition with difference, CMD, and reconstruction terms. Precomputed BERT text features replace its tokenizer and BERT inference; audio and vision use valid-step means.

These are aligned-feature architecture adaptations, not direct execution of original repository training scripts. Original paper metrics are not mixed with these results.
No outside sentiment data or pretrained task weights were used. Attachment2 test, Attachment3, and Attachment4 were not evaluated for these baselines.

## Training cost and selection

| Model | Mean training seconds ± SD | Best epochs by seed 42/43/44 |
|---|---:|---|
| TFN | 18.7402 ± 1.4686 | 6 / 4 / 4 |
| MulT | 44.5175 ± 3.2250 | 5 / 3 / 3 |
| MISA | 34.7700 ± 5.3630 | 13 / 12 / 19 |

## Frozen missing benchmark groups

Per-seed and three-seed mean/SD metrics for modality, ratio, location, modality × ratio, modality × location, and double-modality conditions are in `baseline_missing_groups_per_seed.csv` and `baseline_missing_groups_mean_sd.csv`.

| Model | Group | Mean selection score ± SD |
|---|---|---:|
| TFN | text | 0.6840 ± 0.0116 |
| TFN | audio | 0.7159 ± 0.0046 |
| TFN | vision | 0.7227 ± 0.0020 |
| TFN | text+audio | 0.6658 ± 0.0165 |
| TFN | text+vision | 0.6791 ± 0.0172 |
| TFN | audio+vision | 0.7165 ± 0.0057 |
| MulT | text | 0.7063 ± 0.0113 |
| MulT | audio | 0.7290 ± 0.0065 |
| MulT | vision | 0.7296 ± 0.0043 |
| MulT | text+audio | 0.7058 ± 0.0178 |
| MulT | text+vision | 0.7104 ± 0.0093 |
| MulT | audio+vision | 0.7252 ± 0.0058 |
| MISA | text | 0.7085 ± 0.0041 |
| MISA | audio | 0.7185 ± 0.0012 |
| MISA | vision | 0.7186 ± 0.0018 |
| MISA | text+audio | 0.7102 ± 0.0047 |
| MISA | text+vision | 0.7105 ± 0.0048 |
| MISA | audio+vision | 0.7185 ± 0.0012 |

## Reproducibility and data boundary

Training host: NVIDIA GeForce RTX 3090 (24 GB), PyTorch 2.14.0+cu130. Attachment2 `aligned_50.pkl` SHA256 on both local and server: `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`. Model adapter SHA256: `f7e5ee67c265d52f9d8d8086b0d1e047d6c106ac8f1d59d7305b72c4b7eafaa8`; training script SHA256: `e10170957b5ba3c790e4bf54faddde485f8e07fe06ba7f4d76f000b44765ccd2`.

The pickle is a monolithic container, so it is deserialized as a whole. The loader constructs only train/valid datasets (`include_test=False`); test samples and labels are never indexed, evaluated, or used for selection. All class weights use train counts only. The benchmark definition SHA is verified before every run. Nine per-seed checkpoints, logs, and metrics are preserved locally; checkpoint hashes are in `baseline_checkpoint_manifest.json`.
