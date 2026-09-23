# B2-B0-BlockMask

Why: isolate continuous block augmentation on the unchanged B0 weighted-CE backbone.

Data/protocol: attachment2 aligned-50 train for optimization and valid for checkpoint selection; normalization=none; 54 frozen missing scenarios plus clean; test and attachment3 unused.

Result: best robust epoch 2; clean selection score 0.741275; mean missing score 0.737375; change vs B0-WCE -0.001790. Full metrics are in metrics.json.

Paper status: diagnostic B2 ablation; avoid a broad robustness claim from one seed and one validation split.

Next action: stop at B2; await further instructions.
