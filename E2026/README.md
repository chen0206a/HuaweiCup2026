# 2026 华为杯 E 题 Q2

赛题原文保存在 `references/problem_original.docx`，Q2 简报保存在项目根目录。当前实现读取附件 2 已预计算的 aligned-50 特征；源 pickle 只读。

## Data and mask contract

`padding_mask[b,t]` is true for a real aligned utterance timestep. It is built from the audited binary valid-prefix signal `text_bert[:,1,:]`, and loading rechecks that audio and vision have no nonzero feature vectors in its padding suffix. B0 mean pooling uses this mask for all three modalities.

`availability_mask[b,m,t]` records deliberate observation availability. Every raw complete example starts with all entries true. A later artificial masking stage may set selected entries false; padding remains a separate concept and must still be handled with `padding_mask`.

`native_zero_mask[b,m,t]` records that the original feature vector was all zero at a valid timestep. It is a content or data quality flag only. It must not be used to remove a timestep or to infer artificial missingness. The 153 samples whose vision is all zero over valid timesteps remain in the dataset and are identified by `vision_all_zero`.

Batch keys are `id`, `text [B,50,768]`, `audio [B,50,74]`, `vision [B,50,35]` (all float32), `padding_mask [B,50]` (bool), `availability_mask [B,3,50]` (bool), `native_zero_mask [B,3,50]` (bool), `cls_label [B]` (long), `reg_label [B]` (float32), and `vision_all_zero [B]` (bool).

## Preprocessing and B0

The default is `normalization: none`, which uses the supplied processed features after dtype conversion. Optional `train_featurewise` fits per-feature means and standard deviations using valid train positions only; valid native-zero vectors participate, padding does not, and the same train scaler is applied to valid and test. The original arrays and native-zero flags are retained.

B0 independently projects each modality's masked mean, concatenates the projections, and predicts the 3-class label and regression score with a small MLP. Its loss is cross entropy plus configurable `lambda_reg * SmoothL1`; no class weighting is used. Train optimizes parameters, valid selects the checkpoint, and test is evaluated only after selection as a holdout.

`configs/b0_weighted_ce.yaml` defines a diagnostic ablation with the same B0 configuration and balanced class weights calculated from train labels only. It disables test loading and evaluation. Run it with `python -m src.training.train --config configs/b0_weighted_ce.yaml`, then build a validation-only comparison with `python scripts/compare_b0_weighted_validation.py`. `python scripts/analyze_b0_validation_selection.py` compares B0's validation metric-optimal epochs with its valid-loss-selected epoch.

On the server, run tests and implementation checks from this directory:

```bash
pytest -q
python scripts/smoke_b0.py
python scripts/overfit32.py
python -m src.training.train --config configs/b0.yaml
```
