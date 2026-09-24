# B5-F1 low-rank pairwise fusion, seed 42

## Why run it

Test whether explicit text–audio, text–vision and audio–vision multiplicative interactions add information beyond the frozen B0 fusion. B5-P pooling candidates are not included, and B4' follow-up remains paused.

## Frozen protocol and model

Before the run, the B5-P training code was audited: `model.train()` is followed in every epoch by `eval()` for the frozen B0 modality projections, fusion and heads. Its configuration sets `freeze_b0_parameters: true`; runtime inspection showed all three projection Dropout modules and fusion Dropout had `training=False`, and all original B0 parameters had `requires_grad=False`. F1 follows this same freeze/eval protocol.

F0 is the original seed-42 B0-WCE best-selection checkpoint. F1 starts from it and adds three bias-free `Linear(128,16)` projections, pairwise Hadamard products concatenated to 48 dimensions, and a bias-free `Linear(48,128)` residual on the frozen B0 fusion output. The three U projections use Xavier initialization and W is all zero, giving exact B0 predictions initially. Only 12,288 interaction parameters train; total parameters are 175,748. The B5-P training recipe remains: clean train, batch 128, train-only balanced CE plus SmoothL1, AdamW LR 0.001/weight decay 0.0001, no scheduler, max 80 epochs/patience 12, seed 42, no BlockMask augmentation. Validation is the unchanged clean plus 54 missing scenes, benchmark seed 20260923 and hash `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`. No test or attachment3 data were loaded.

The frozen B0 train and validation encodings were cached. Cached validation and normal full forward matched exactly on all 55 metric rows before and after training. The F1 initial logits/regression exactly matched F0 on all 728 clean validation samples. All 20 B0 tensors remained bitwise equal to the source checkpoint. CPU/CUDA gradient, shape, zero-W/nonzero-U, gradient-flow and checkpoint round-trip tests passed.

## Result

The best clean and best robust checkpoints both occur at epoch 5. F1 robust score is **0.738850** versus F0 **0.740248**, a change of **−0.001398**. Clean Accuracy falls from 0.645604 to 0.633242 while Macro-F1 rises from 0.603955 to 0.619080; clean MAE improves slightly (0.615772 to 0.613773) and Pearson falls (0.636786 to 0.632657). Mean-missing Accuracy falls from 0.642094 to 0.624440, Macro-F1 rises from 0.601590 to 0.609150, MAE improves from 0.618273 to 0.616128, and Pearson falls from 0.632044 to 0.627087. Text missing and vision missing score averages fall by 0.005270 and 0.002220; text+audio and text+vision double-stress scores fall by 0.007762 and 0.008067. The 15 native vision-all-zero validation examples show lower clean Macro-F1 and Pearson, and worse mean-missing classification and Pearson.

The residual norm ratio `||delta_z||/(||z_B0||+1e-8)` is 0.1561±0.0892 clean and 0.1568±0.0894 mean missing; clean P05/P50/P95 are 0.0503/0.1359/0.3391. Average pair contribution norms clean are TA 0.6041, TV 0.5737, AV 0.3689; mean missing 0.6061/0.5741/0.3682. These are diagnostics only, not selection criteria. Training took 14.36 s for 17 epochs with cached validation averaging 0.746 s/epoch; a normal full validation check took 4.39 s.

## Paper status / next action

The prespecified screening rule says robust score at or below F0 stops this fusion route. F1 is a negative seed-42 result; do not tune rank, losses or architecture to rescue it, and do not run seed43/44. Do not combine it with P1/P2. Full scenario metrics are in `outputs/metrics/b5_fusion_metrics.json`, with a readable table in `b5_fusion_report.md`.
