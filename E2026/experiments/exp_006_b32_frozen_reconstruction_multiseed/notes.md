# B3.2 frozen reconstruction multi-seed stability

## Why run it
B3.1 seed 42 showed a small aggregate robust-score gain but a severe vision_all_zero subgroup regression. This run checks whether those effects reproduce under the same protocol.

## Procedure
Seed 42 reuses B3.1; seeds 43 and 44 use their corresponding B0-WCE best-selection checkpoints and train only three reconstructors for 80 epochs. Architecture, optimizer, losses, B2 block-mask generation, 54 validation scenarios, benchmark seed/hash, and the global alpha grid {0,0.25,0.5,0.75,1.0} are fixed. The optimized evaluator shares one latent/reconstructor forward per scenario batch across all alphas. Test and attachment3 were not loaded.

## Verification
The optimized evaluator was compared to the pre-optimization seed-42 results across 5 alphas x 55 clean/missing rows and all stored metric, per-class, confusion-matrix, and vision_all_zero fields: 0 mismatches. Alpha=0 is exact B0 for all three seeds. Every B0 frozen-tensor equality check passed (20 tensors per seed).

## Result
No nonzero alpha improves robust score or mean-missing score consistently across the three seeds. Seeds 43 and 44 prefer alpha=0 for robust score, and the three-seed mean robust/mean-missing scores are lower at every nonzero alpha. Accuracy decreases on clean and missing validation for all seeds. Text missing has no stable score gain. The vision_all_zero mean-missing subset worsens on Accuracy, Macro-F1, MAE, Pearson, and selection score at every nonzero alpha in all three seeds.

## Paper status / decision
The reconstruction main line is not stable enough to continue. Return to B0-WCE and do not proceed to reconstruction-based B4. Treat the seed-42 gain as seed-specific validation behavior; do not claim a general reconstruction benefit.
