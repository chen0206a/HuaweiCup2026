# Q2 public baseline integration check

Date: 2026-09-25

## Gate A: MISA regression sanity

- `MISA_REGRESSION_SANITY = PASS`
- `BASELINE_TABLE_LOCKED = YES`
- Regression targets for train and validation come from the same raw `regression_labels` field as the other Q2 models. Dataset targets equal the raw labels after the training pipeline's float32 cast. Normalization is none; no inverse transform is needed.
- The MISA regression head is `Linear(384, 1)`, has no output activation, and returns `[B]`, matching the `[B]` target. The loss is Weighted CE + SmoothL1 + 0.3 × DifferenceLoss + CMD + ReconstructionLoss, with regression weight 1.0.
- Regression head and shared-representation regression gradients were nonzero on fixed train-only batches. Auxiliary losses were also nonzero at the shared representation. No loss-scale or target-field mismatch was found.
- The same validation loader and evaluator reproduce saved checkpoint MAE/Pearson within 1.4e-8. On fixed random 20-sample subsets, NumPy MAE and SciPy Pearson match the project implementation within 1.2e-16.
- Across seeds 42/43/44, MISA clean validation MAE is 1.0288 / 0.8906 / 0.8332 and Pearson is 0.6362 / 0.6256 / 0.6188. Prediction means show positive bias and prediction spread exceeds target spread; errors occur across all target-intensity bins. This indicates weak absolute calibration for this adapted model, not a target-scale or evaluation bug.
- Existing checkpoints, source pickle and baseline result files were not modified. Attachment2 test, Attachment3 and Attachment4 were not evaluated.

## Locked baseline values

Clean validation, three-seed mean ± sample SD:

| Method | Parameters | Accuracy ↑ | Macro-F1 ↑ | MAE ↓ | Pearson ↑ |
|---|---:|---:|---:|---:|---:|
| TFN | 4,840,670 | 0.6099 ± 0.0076 | 0.5872 ± 0.0083 | 0.6336 ± 0.0102 | 0.6056 ± 0.0098 |
| MulT | 624,094 | 0.6172 ± 0.0148 | 0.5891 ± 0.0236 | 0.6221 ± 0.0260 | 0.6369 ± 0.0115 |
| MISA | 1,118,852 | 0.6131 ± 0.0083 | 0.6010 ± 0.0060 | 0.9175 ± 0.1005 | 0.6269 ± 0.0087 |
| 本文模型 | 164,343 | 0.6419 ± 0.0057 | 0.6241 ± 0.0026 | 0.6059 ± 0.0066 | 0.6472 ± 0.0025 |

The clean values and MISA results are retained from the existing baseline summary; the sanity check did not trigger retraining.

## Paper integration

- Q2 LaTeX source: `paper/sections/q2.tex`
- Bibliography: `paper/refs.bib`; all three references have verified publication metadata. TFN is ACL Anthology D17-1115, MulT is ACL Anthology P19-1656, and MISA is ACM MM 2020, DOI 10.1145/3394171.3413678.
- Added model adaptation/training notes, clean results table and a 54-scenario descriptive missing-results table. The missing comparison explicitly notes that the newly trained public baselines were clean-validation selected while the locked model used the fixed missing-scenario validation criterion. No strict robustness ranking is claimed.
- Final clean comparison is Table 1 on PDF page 3. The missing comparison is Table 2 on PDF page 3. The Q2 section starts on page 2 and continues on page 3.
- PDF: `paper/main.pdf`
- Compile: XeLaTeX and BibTeX completed successfully. `latexmk` was unavailable because MiKTeX could not find Perl, so the equivalent compile steps were run directly. Final log has no unresolved citations/references and no overfull/underfull box warnings. Rendered page 3 was visually checked; both tables fit within the text width and Chinese line breaks are readable.
- Q2 section contains no TODO. TODO placeholders remain in the abstract and Q1/Q3/Q4 (and other unfinished global paper sections); they were outside this Q2 task and were left untouched.

## Final status

`PUBLIC_BASELINE_EXPERIMENT = COMPLETE`

`NO_MORE_BASELINE_TRAINING_RECOMMENDED = YES`
