# B3.1 frozen-backbone conservative reconstruction

## Why run it
B3 jointly fine-tuned the B0 backbone and fell below B0-WCE robust score. This run isolates reconstruction by restoring seed-42 B0-WCE best-selection weights and freezing every B0 prediction parameter; only three reconstructors train.

## Procedure
One seed-42 run, 80 epochs, original B2 block-mask training generator, lambda_rec=1.0, lambda_id=0, global inference alpha grid {0, 0.25, 0.5, 0.75, 1.0}. Evaluation uses the unchanged 54-scenario validation benchmark (seed 20260923) plus clean validation. No test or attachment3 data were loaded.

## Result
Frozen B0 tensors passed exact equality (20 tensors; also identical to source checkpoint). Alpha 0 produced bitwise-identical B0 logits and regression outputs. Robust score rose from 0.74025 at alpha 0 to 0.74219 at alpha 1; every tested alpha above zero exceeded alpha 0. Clean selection score also rose. However, the 15 vision_all_zero validation samples degraded sharply for every nonzero alpha (clean accuracy 0.5333 to 0.3333), and text-only missing selection score did not improve materially. Aggregate gains therefore do not yet establish a safe general reconstruction rule.

## Paper status / next action
Useful as a controlled positive aggregate signal, with an explicit severe subgroup failure. Do not proceed to reconstruction-based B4 from this alone. Preserve the frozen B0 route and native-zero subgroup finding for method discussion; no additional model or alpha fitting was performed.
