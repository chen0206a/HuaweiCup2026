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

## B2.1 training-seed stability

The original four B0/B1/B2 runs used training seed 42; `20260923` is the separate, fixed missing-benchmark seed. B2.1 reuses seed 42 and repeats the unchanged training configurations at seeds 43 and 44. For each seed, B0/B1 use best validation selection-score checkpoints and B2 uses best robust-score checkpoints. `scripts/run_b21_seed.py` permits only the training seed and output filenames to differ from each frozen base config, checks the benchmark definition SHA-256, and never constructs the test split or reads attachment3. `scripts/summarize_b21.py` reports sample standard deviations over 42/43/44 and paired score differences. Run one seed at a time with `python -m scripts.run_b21_seed --seed 43` and `--seed 44`, then `python -m scripts.summarize_b21 --seeds 42 43 44`.

## B3 same-position latent reconstruction

B3 starts from the seed-42 B0-WCE best selection-score checkpoint. B0's affine modality projection runs per timestep and is then masked-mean pooled; its LayerNorm, GELU, Dropout, fusion, and prediction heads stay in their B0 positions. With reconstruction disabled, this is checked against the actual B0 checkpoint on all 728 complete validation samples before training. Three independent small MLPs reconstruct each modality's 128-dimensional affine latent from the other two at the same aligned position. Source latents at raw all-zero valid positions are explicitly set to zero before entering a reconstructor, so an affine projection bias cannot impersonate observed source information.

The predictor observes only features and `padding_mask`; it detects `zero_trigger` from all-zero raw input vectors at valid positions. Synthetic `availability_mask` is used only by the SmoothL1 reconstruction and observed identity losses and by diagnostics. The reconstruction targets come from the full pre-mask features through the same affine projection and are detached. Native-zero vectors can trigger reconstruction on clean validation; `native_zero_mask`, `availability_mask`, and `zero_trigger` remain separate. Training uses the frozen B2 augmentation, weighted CE plus SmoothL1 regression, reconstruction weight 1.0, identity weight 0.1, and seed 42. The frozen 54-scenario benchmark selects the best robust checkpoint; a best clean checkpoint is also saved. Run `python -m pytest tests/test_b3_reconstruction.py -q`, `python -m src.training.run_b3 equivalence`, `python -m src.training.run_b3 train`, then `python -m src.training.run_b3 compare`. Test and attachment3 remain unused.

`python -m scripts.plot_b3` renders the saved validation score and reconstruction-quality curves from `b3_summary.json` and the frozen scenario CSV files; it does not rerun inference or change model selection.

## B3.1 frozen-backbone conservative reconstruction

B3.1 initializes from `b0_weighted_ce_best_selection_score.pt` (training seed 42) and freezes every B0 modality projection, LayerNorm, fusion, and prediction-head tensor (`requires_grad=False`, B0 modules kept in eval mode). Only the three same-position reconstructors train, with the existing B2 block-mask augmentation, weighted classification plus regression task loss, and `lambda_rec=1`; `lambda_id=0`. The task-gradient check and exact tensor comparison against the source B0 checkpoint are recorded in `outputs/metrics/b31_training_metrics.json`.

At inference, the existing raw-zero trigger is retained. Reconstructed latents are blended only at triggered valid positions as `(1-alpha)*H_base + alpha*H_hat`; padding stays excluded. Alpha is one global fixed value, evaluated on `{0,0.25,0.5,0.75,1.0}` using the unchanged 54-scenario validation benchmark. Alpha 0 is checked for bitwise-identical logits and regression output to B0. The initial experiment's report is `outputs/metrics/b31_report.md`; machine-readable full results are `b31_alpha_sweep.json` and `b31_b3_checkpoint_enabled_disabled.json`. Test and attachment3 are not loaded.

## B3.2 frozen reconstruction multi-seed stability

B3.2 reuses the seed-42 B3.1 run and trains only the missing seeds 43 and 44 from their corresponding B0-WCE best-selection checkpoints. The frozen benchmark hash and alpha grid are checked; only train/valid data are loaded. The multi-alpha evaluator computes the modality latents and three reconstructors once per scenario batch, then reuses them across all five alphas. It was checked against the original seed-42 evaluation at every alpha and all 55 clean/missing scenarios with zero mismatched fields. `scripts/summarize_b32.py` reports mean ± sample standard deviation and paired per-seed deltas. Current three-seed results do not show a consistent positive robust or mean-missing gain; the vision-all-zero subgroup is harmed at every nonzero alpha. The supported choice remains B0-WCE; stop reconstruction and do not proceed to B4.

## B4' reliability-aware dynamic fusion (separate branch)

`src/models/reliability_gate.py` freezes the seed-42 B0-WCE modality projections, LayerNorms, fusion, and prediction heads. A small joint router reads the three unchanged B0 projected pooled embeddings plus observable `zero_ratio[3]`, `longest_zero_run_ratio[3]`, and valid-length ratio. These statistics count all-zero feature vectors only where `padding_mask` is true; zero values can be native content and are never treated as an oracle missing label. Neither `availability_mask` nor `native_zero_mask` enters the predictor. The router outputs three gates `2*sigmoid(a)`, initialized exactly to one, and scales the B0 embeddings before the original frozen fusion/head. This branch does not use reconstruction.

Run `python -m pytest tests/test_b4_reliability.py -q`, then `python -m src.training.run_b4_reliability --config configs/b4_reliability_gate.yaml` on the GPU server. Training uses the frozen B2 augmentation, train-only balanced class weights, and seed 42. Validation uses the unchanged 54-scenario benchmark and seed 20260923; frozen B0 embeddings and the observable statistics are cached once. Initial and final cached validation matched normal full forward exactly on all 55 rows and compared fields. Best clean and robust checkpoints are saved separately; the robust checkpoint is the main comparison. The seed-42 screening result is in `outputs/metrics/b4_reliability_report.md`; it warrants a three-seed stability check but does not establish a general gain. Attachment2 test and attachment3 were not loaded.

## B5-P pooling ablation

The original B0 first masked-means raw valid timesteps for each modality, then applies a separate Linear/LayerNorm/GELU/Dropout projection to 128 dimensions. P1/P2 add a residual to these projected mean representations. P1 masked-max pools raw valid timesteps; P2 applies a single scalar attention scorer per modality, masking padding before softmax. Each auxiliary pooled vector uses the unchanged B0 projection to obtain a 128-dimensional residual, scaled by a separate zero-initialized gamma. All native-zero valid timesteps remain in pooling. The original seed-42 B0-WCE checkpoint initializes P1/P2, and its prediction parameters stay frozen; only gamma (P1) or gamma plus attention scorers (P2) train. The original B0 clean-only training recipe, weighted CE, SmoothL1, AdamW settings, normalization, and frozen 54-scenario validation benchmark remain in force. Neither BlockMask augmentation nor B4 gating is added.

Run `python -m pytest tests/test_b5_pooling.py -q`, then `python -m src.training.run_b5_pooling --config configs/b5_pooling.yaml` on the GPU server. P0 re-evaluates the unmodified B0 checkpoint and checks exact agreement with the saved B2 benchmark. P1/P2 verify gamma-zero identity against B0 before training, then save separate best-clean and best-robust checkpoints. `python -m scripts.compare_b5_pooling_checkpoints` evaluates both selection rules on the same frozen validation benchmark. The report records subgroup performance, gamma values, and P2 attention concentration. Test and attachment3 are excluded.

Seed-42 screening gave robust scores P0 0.740248, P1 0.744320, and P2 0.745847. These are validation findings only. P2 loses on all four metrics in the 15-example native vision-all-zero subset; P1 worsens its regression metrics. The pooling line requires cross-seed confirmation before any general claim, and no fusion experiment is included in B5-P.

### B5-P2 paired multi-seed replication

`configs/b5_p2_multiseed.yaml` and `src/training/run_b5_p2_multiseed.py` reproduce P2 only at training seeds 43 and 44; seed42 is reused from exp_008. Each P2 run starts from its matching seed's B0-WCE best-selection checkpoint. B0 parameters stay frozen and the frozen projection/fusion/heads are explicitly in eval mode throughout training. The script checks zero-gamma initial identity, exact frozen-tensor equality, checkpoint round-trip, and the frozen benchmark SHA256 before comparing validation results. It loads train/valid only, never attachment2 test or attachment3. Benchmark seed 20260923 and all 54 scenario definitions remain frozen.

Paired robust deltas P2−B0 are +0.005600/+0.003791/−0.000090 for seeds 42/43/44; mean +0.003100 ± 0.002907 sample SD. Two seeds are positive while seed44 is effectively neutral, so retain P2 as a seed-sensitive candidate and avoid claiming a consistent three-seed gain. Full results, subgroup tables, gamma/attention diagnostics, and vision-all-zero subset are in `outputs/metrics/b5_p2_multiseed_report.md` and `b5_p2_multiseed_summary.json`; formal record is `experiments/exp_012_b5_p2_multiseed/`. No z-score or module combination is included.

### B5-N1 text-only feature-wise z-score

`configs/b5_n1_text_zscore.yaml` and `src/training/run_b5_n1_text_zscore.py` compare N1 against the reused seed42 B0-WCE checkpoint (N0). N1 fully retrains the original B0 using only train-valid-position statistics for the 768 text dimensions; audio and vision are only dtype-converted. The epsilon floor is `1e-6`; padding is excluded from fit and masked pooling. Dataset transform precedes `apply_blocks`, which re-overwrites synthetic missing spans with exact zero. The benchmark definition and SHA256 remain unchanged; validation is clean + 54 scenarios. Both best-clean and best-robust checkpoints are saved.

N0 clean selection score/robust score are 0.741331/0.740248 and are reproduced exactly by the current evaluator. N1 best-clean and best-robust both select epoch 1; robust is 0.734474 (Δ vs N0 = −0.005774). Neutral recall/F1 rise, but accuracy and subgroup selection scores decline. Stop the normalization route; do not add modality normalization, epsilon sweeps, or combine with P2. Full report, scaler state, metrics, and formal record are in `outputs/metrics/b5_n1_text_zscore_report.md`, `b5_n1_text_scaler_state.json`, and `experiments/exp_013_b5_n1_text_zscore/`.

## B5-F1 low-rank fusion ablation

F1 is independent of P1/P2 and starts again from the original seed-42 B0-WCE checkpoint. `src/models/fusion_interaction.py` freezes all B0 prediction parameters and holds their Dropout modules in eval mode. It uses three bias-free 128→16 projections, pairwise Hadamard interactions, and a zero-initialized bias-free 48→128 projection to add a residual to the original B0 fusion output. F1 starts with predictions exactly equal to B0. Only the 12,288 interaction parameters train. The original B5-P clean training recipe and the frozen B2 validation benchmark are reused. Frozen train/validation B0 encodings are cached; cached and full validation metrics match exactly.

Run `python -m pytest tests/test_b5_fusion.py -q`, then `python -m src.training.run_b5_fusion --config configs/b5_fusion.yaml` on the GPU server. Both best-clean and best-robust checkpoints are saved. The seed-42 robust score is 0.738850 versus F0 0.740248, so the prespecified screening rule stops this fusion branch without further tuning or seed43/44 runs. Results and pairwise residual diagnostics are in `outputs/metrics/b5_fusion_report.md` and `outputs/metrics/b5_fusion_metrics.json`. Test and attachment3 remain excluded.


## B5-L1 task-balance screening

`configs/b5_l1_lambda.yaml` runs the original B0-WCE architecture from scratch with seed 42 and lambda values 0.25, 0.5, 1.0, and 2.0. It uses the same fresh initialization, train order, train-derived balanced CE weights, optimizer, and frozen validation benchmark across candidates. The benchmark file is hash-checked. Test and attachment 3 are not loaded. Both clean-selected and robust-selected checkpoints are saved separately. See `outputs/metrics/b5_l1_lambda_report.md` and `experiments/exp_010_b5_l1_lambda/` for results and configuration.

Seed-42 screening found robust deltas vs the same-protocol lambda=1 reference of −0.010943, −0.002920, 0, and +0.000808 for lambda .25/.5/1/2. The slight lambda=2 gain is below the prespecified +.002 threshold; its increased Neutral recall/F1 comes with reduced Accuracy. Keep lambda_reg=1.0 and stop the grid without further tuning.


## B5-H0 head consistency diagnostic

`python -m scripts.run_b5_h0_head_diagnostic` performs read-only inference with the historical seed-42 B0-WCE checkpoint on valid clean + the frozen 54-scenario benchmark. It exports per-sample JSONL/CSV and computes sign-map head agreement, descriptive tau sweep, class/regression distributions, confidence bins, scenario and paired clean→missing diagnostics, and oracle union upper bounds. Test and attachment3 are not evaluated. See `outputs/metrics/b5_h0_head_diagnostic.md` and `experiments/exp_011_b5_h0_head_diagnostic/`. The result indicates partial complementarity with substantial shared errors; no follow-up training is included.
