# B5-P2 attention residual pooling: multi-seed replication

## Why

Seed42 screening showed a robust-score gain for P2 over B0-WCE. This experiment tests whether it replicates with paired, seed-specific B0 checkpoints at training seeds 43 and 44. Seed42 is reused from `exp_008_b5_pooling`.

## Protocol

P2 architecture, training settings, freeze/eval behavior, losses, optimizer, benchmark, and selection rules are unchanged from `exp_008_b5_pooling`. Each new P2 starts from its same-seed B0-WCE best-selection checkpoint. All B0 prediction tensors are frozen and frozen modules remain in eval mode after `model.train()`; only attention scorers and gamma train. P2 uses best-robust checkpoints for the paired main comparison and also saves best-clean checkpoints. Validation uses clean plus the frozen 54 missing scenarios (benchmark seed 20260923; SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`). Test and attachment3 are excluded.

## Result

Paired robust deltas (P2 minus same-seed B0) are +0.005600 (seed42), +0.003791 (seed43), and -0.000090 (seed44). Mean paired delta is +0.003100 with sample SD 0.002907. The result is positive in two of three seeds; seed44 is effectively neutral. This is seed-sensitive evidence, not a stable all-seed gain. Do not promote it to a general improvement claim or add modules based on this screening alone.

The full metric tables, per-seed group breakdowns, attention diagnostics, and vision-all-zero subset results are in `outputs/metrics/b5_p2_multiseed_report.md` and the machine-readable summary. Both new runs passed initial-identity, frozen-parameter equality, and checkpoint round-trip checks.

## Next action

Stop B5-P2 replication here. No z-score, P2 combination, or follow-on module is included in this experiment.
