# B5-L1 multi-task loss balance screening

## Result

The actual B0-WCE configuration uses `lambda_reg=1.0`, weighted cross entropy with train-only balanced class weights, and SmoothL1. Four from-scratch seed-42 runs used the identical initialization (20/20 tensors), train order, class weights, optimizer, and frozen clean-plus-54-scenario benchmark. Test and attachment 3 were not loaded.

The lambda 1.0 rerun exactly reproduced the historical seed-42 B0 clean selection score (0.741330793) and robust score (0.740247931). Relative to this same-protocol reference, robust-score deltas were: lambda 0.25 −0.010943, 0.5 −0.002920, 1.0 0, and 2.0 +0.000808. No non-baseline setting met the prespecified +0.002 screening threshold, so retain lambda 1.0 and do not refine the grid.

Lambda 2.0 improved MAE/Pearson and Macro-F1 relative to lambda 1.0, but reduced accuracy; its robust gain was weak and does not establish a joint improvement. The lower lambdas did not help classification: lower lambda also had worse clean and missing Accuracy/Macro-F1 on the selected checkpoints, despite worse regression at 0.25/0.5. At lambda 2.0 Neutral recall/F1 increased substantially, explaining much of the Macro-F1 movement while accuracy fell. Per-group scenario metrics, loss contributions, the fixed-batch gradient diagnostic, and initialization hashes are in `metrics.json` and `E2026/outputs/metrics/b5_l1_lambda_report.md`.

Initial fusion-layer gradient norms on the shared initial model were CE 1.59935, SmoothL1 2.43345, cosine −0.03458 (descriptive only; no online weighting). Training time was 76.0–81.2 seconds per run. Best clean and robust epochs coincided for these four seed-42 runs: 2, 2, 3, and 3.

## Next action

Keep `lambda_reg=1.0`. Stop the lambda sweep; no seed43/44 follow-up is warranted under the screening rule.
