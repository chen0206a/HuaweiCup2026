# Q2 LTARP component ablation

## Purpose

Measure the contribution of temporal attention, the mean branch, residual versus concatenation fusion, and each modality's attention branch on the existing MMP backbone.

## Frozen protocol

- Seeds: 42, 43, 44.
- Attachment2 aligned_50; train=3395, valid=728. Test is excluded from Dataset construction, checkpoint selection, and evaluation.
- Each run starts from the matching seed's CleanSelect MMP checkpoint. MMP modality projections, fusion, classification head, and regression head are frozen and held in eval mode. Only variant-specific new parameters are trained.
- Weighted CE + SmoothL1 (`lambda_reg=1.0`), AdamW (`lr=0.001`, `weight_decay=0.0001`), batch 128, maximum 80 epochs, patience 12, no normalization or augmentation.
- Checkpoints are selected by complete-input validation `S_val`; the selected checkpoint is then fixed for clean and the original 54 missing scenarios.
- Benchmark seed 20260923; benchmark SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`.
- A0 reuses corresponding CleanSelect MMP checkpoints. A4 reuses the clean-selected full LTARP checkpoint for each seed; for seeds 43/44 the stored checkpoint epoch equals the recorded clean-best epoch.

## Variants

A0 MMP; A1 attention-only; A2 mean-attention concatenation with a per-modality `Linear(256,128)`; A3 fixed residual coefficient 1; A4 formal full LTARP; A5 text-only attention residual; A6 audio-only attention residual; A7 vision-only attention residual.

## Outputs

Results are written to `outputs/final/q2/ltarp_component_ablation/`. This experiment does not edit locked model checkpoints, historical metrics, benchmark definition, or paper files.

## Results (2026-09-26)

- The reused A0 and A4 checkpoints for seeds 42/43/44 passed hash and CleanSelect metric reproduction checks; maximum absolute metric error was below `8.3e-9`. Frozen MMP tensors were unchanged for all evaluated checkpoints.
- Mean robust score (mean ± sample SD): A0 `0.74168 ± 0.00146`; A1 `0.74618 ± 0.00016`; A2 `0.73812 ± 0.00437`; A3 `0.74469 ± 0.00164`; A4 `0.74472 ± 0.00144`; A5 `0.74480 ± 0.00139`; A6 `0.74300 ± 0.00063`; A7 `0.74338 ± 0.00187`.
- A1 attention-only improved Macro-F1, MAE, and Pearson over MMP on both clean and mean-missing validation; mean-missing Accuracy was slightly lower. Its paired robust-score gain was `+0.00450 ± 0.00160` over MMP across all three seeds.
- Adding the mean branch in A4 gave small additional Macro-F1 gains over A1 (`+0.00062` clean, `+0.00126` mean-missing), while robust score fell by `0.00146 ± 0.00134`; the gain is confined to selected raw metrics rather than a consistent aggregate improvement.
- Learnable gamma (A4) and fixed gamma=1 (A3) had nearly identical mean robust score difference (`+0.00003 ± 0.00258`), with learnable gamma improving mean Macro-F1 and MAE but slightly lowering Accuracy/Pearson.
- A2 concatenation had the largest parameter increase (`+99,568`) and the most variable robust score; A5 text-only attention had the largest modality-only robust gain over MMP (`+0.00312 ± 0.00264`).
- All detailed seedwise metrics, 54-scenario breakdowns, paired deltas, checkpoint hashes, and selected epochs are in `outputs/final/q2/ltarp_component_ablation/ablation_metrics.json` and the accompanying CSVs. No test split, Attachment3, or Attachment4 was used.
