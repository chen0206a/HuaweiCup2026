# Q2 B3 B0 latent reconstruction

Why: test whether same-position cross-modal latent reconstruction improves robustness to continuous local missingness, without changing B0 fusion or adding a reliability gate.

Protocol: initialize from seed-42 B0-WCE best-selection checkpoint; seed 42 training; frozen B2 block augmentation and 54-scene validation benchmark; best-robust checkpoint for comparison; attachment2 test and attachment3 unused. B0 affine/mean equivalence passed on all 728 validation examples.

Result: B3 clean selection score 0.734221, mean-missing score 0.730388, robust score 0.732304 at epoch 2. B0-WCE robust score was 0.740248. Clean Accuracy fell from 0.6456 to 0.6181 and mean-missing Accuracy from 0.6421 to 0.6119. Single-modality text, audio and vision missing scores were all lower than B0-WCE. Full scenario, reconstruction, native-zero and 15-sample vision-all-zero diagnostics are in outputs/metrics/b3_summary.json and the generated comparison report.

Interpretation: this one-seed B3 configuration does not support a robustness benefit. Clean-input reconstruction toggling changes only five validation class predictions, so the overall decline cannot be assigned solely to native-zero triggering; training-related backbone drift may contribute. This is a diagnostic inference, not a causal decomposition.

Paper status: negative B3 ablation may be reported with exact validation protocol. Do not claim a significant effect. Do not enter B4 or add a gate without a separately agreed objective.
