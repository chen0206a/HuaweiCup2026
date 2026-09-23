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

## B1 temporal encoder

B1 uses an independent input projection and 2-layer, 4-head Transformer encoder for each modality. Fixed sinusoidal positions encode temporal order. `padding_mask` is passed to each encoder as `src_key_padding_mask=~padding_mask`; masked mean pooling also excludes padded outputs. Encoded modalities are pooled independently and concatenated, without cross-modal attention. B1-CE and B1-WeightedCE share all settings except classification-loss weighting; balanced weights use train labels only. Both use train/valid only and save separate best-valid-loss and best-selection-score checkpoints.

The project-only validation score is `0.25*Accuracy + 0.25*Macro-F1 + 0.25*(1-MAE/6) + 0.25*(Pearson+1)/2`; it is not the competition's official score. Horizontal B0/B1 comparison uses the checkpoint selected by this score. Run `python scripts/smoke_b1.py`, `python scripts/overfit32_b1.py`, then `python -m src.training.train_b1 --config configs/b1_ce.yaml` and the weighted config. `python scripts/compare_b0_b1_validation.py` writes the validation-only comparison.

On the server, run tests and implementation checks from this directory:

```bash
pytest -q
python scripts/smoke_b0.py
python scripts/overfit32.py
python -m src.training.train --config configs/b0.yaml
```

## B2 continuous block missing benchmark

`src/data/block_mask.py` draws blocks only within the true valid prefix. The block length is `max(1, round(rho * valid_length))`, with Python's half-to-even tie rule. Artificially missing feature vectors become zero, but their `padding_mask` entries remain true, so B0 pooling and B1 attention receive the zero vectors. The generator marks those coordinates false in `availability_mask` for bookkeeping only. Predictors receive exactly `text`, `audio`, `vision`, and `padding_mask`; neither `availability_mask` nor `native_zero_mask` is passed as a predictor feature. Original native-zero vectors remain untouched. The source pickle is read only.

Training keeps about half of samples complete. The other half get one modality block with probability 0.75, or two independently sampled modality blocks with probability 0.25. Modalities and ratios in `{0.1,0.2,0.3,0.4,0.5}` are sampled uniformly; starts are uniform among valid positions. The fixed validation benchmark has 45 single modality and 9 double modality scenes, plus clean. Definitions and seed are saved in `outputs/metrics/b2_benchmark_definition.json` and must be reused for B3/B4. The benchmark uses train/valid only, never attachment2 test or attachment3.

Run `python -m pytest tests/test_block_mask.py -q`, then `python -m src.training.run_b2 inherent` before any B2 training. Train with `python -m src.training.run_b2 train --config configs/b2_b0_blockmask.yaml` and the B1 config. Finally run `python -m src.training.run_b2 compare`. The project-only robust checkpoint score is half the clean selection score plus half the mean score across 54 missing scenes; B2 horizontal comparisons use best robust checkpoints. Both best clean and best robust checkpoints are saved.
