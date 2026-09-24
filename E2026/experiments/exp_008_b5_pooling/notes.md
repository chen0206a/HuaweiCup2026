# B5-P: strong-B0 pooling ablation, seed 42

## Why run it

Test whether B0's masked mean pooling limits clean or continuous block missing validation performance. B4' follow-up seeds are paused. This experiment changes only the pooling path and does not use B4 gating, reconstruction, a temporal encoder, test, or attachment3.

## Actual B0 and intervention

`src/models/baseline.py` applies a masked mean to each raw modality feature sequence (768/74/35), then a separate `Linear(D,128) → LayerNorm(128) → GELU → Dropout(0.1)` projection. It concatenates three 128-dimensional representations, applies a `Linear(384,128) → LayerNorm → GELU → Dropout` fusion, then three-class and regression heads. Padding is excluded from the mean; native all-zero valid steps remain included.

P0 is the unchanged seed-42 B0-WCE best-selection checkpoint (historical epoch 3). P1 adds `gamma_m * B0Projection_m(masked_max(raw_m))` to each projected mean representation; P2 replaces max with a scalar per-modality masked attention pool. Gamma starts exactly zero. Each candidate initializes all B0 weights from P0 and freezes them; P1 trains three gamma scalars, P2 trains three gamma scalars and three `Linear(D,1)` attention scorers. Existing B0 projection, fusion, and heads remain exact and in eval mode. Training uses clean train examples, train-only balanced CE plus SmoothL1, AdamW LR 0.001/weight decay 0.0001, batch 128, seed 42, max 80 epochs/patience 12. No BlockMask train augmentation is added. The clean plus 54 missing validation scenes and benchmark hash are unchanged.

## Verification

P0 exactly reproduces saved B2 benchmark rows. Before training, P1 and P2 logits and regression predictions match P0 bitwise on all 728 validation examples. After training, all 20 B0 tensors match the source checkpoint exactly in each model; checkpoint round trips preserve predictions. CPU/CUDA tests cover padding exclusion, single-step valid sequences, native all-zero vision, forward/backward, and save/load. Both checkpoint selection rules were independently reevaluated on the same validation benchmark. No attachment2 test or attachment3 data were constructed.

## Result and interpretation

P0 robust score is 0.740248. P1 best robust is 0.744320 at epoch 22 (+0.004072); P2 is 0.745847 at epoch 15 (+0.005600). P1 best-clean and best-robust checkpoints coincide at epoch 22. P2 best-clean is epoch 20; its robust score is 0.745683, retaining a +0.005435 margin over P0 under clean selection. P1's clean and mean-missing Accuracy/Macro-F1 rise, while MAE and Pearson each deteriorate slightly. P2 improves clean Macro-F1, MAE and Pearson, and mean-missing Macro-F1, MAE and Pearson; mean-missing Accuracy falls from 0.642094 to 0.639677. All single-modality and rho-group aggregate scores improve, but text+audio double stress falls by 0.000958 for P1 and 0.002321 for P2. P2 text+vision double stress falls by 0.001301.

The 15 `vision_all_zero` validation examples are a serious caveat. P1 preserves clean classification but degrades clean MAE/Pearson from 0.5445/0.3371 to 0.5667/0.2976. P2 reduces clean Accuracy/Macro-F1 from 0.5333/0.5556 to 0.4667/0.4973 and degrades clean MAE/Pearson to 0.5888/0.2438. Missing-scenario subset averages also worsen for both models. This is a small subgroup, but the same failure direction spans all four P2 metrics.

P1 gamma is text +0.194826, audio +0.090689, vision -0.194600. P2 gamma is text +0.364504, audio -0.261424, vision -0.122185. P2's clean normalized attention entropy (L>1) is text 0.593, audio 0.239, vision 0.865; fractions with max timestep weight ≥0.9 are 2.75%, 0%, and 0.27%, respectively. This does not indicate a general single-timestep attention collapse, though audio attention is comparatively concentrated.

## Paper status / next action

The seed-42 gain is screening evidence, not a stable improvement claim. Both P1 and P2 meet the prespecified aggregate robust-score screening threshold without broad clean/mean-missing four-metric collapse, so both qualify for seed43/44 stability checks. P1 is the safer priority because P2 has a marked native-zero subgroup regression and classification drop; a follow-up should state the subgroup guardrail before inspecting new results. P1's native-zero regression harm must also be checked across seeds. Do not proceed to fusion in this phase. Full per-scenario metrics and history are in `outputs/metrics/b5_pooling_metrics.json`, `b5_pooling_history.json`, and `b5_pooling_report.md`.
