# Figure Reorganization Audit

## Scope

Only figure structure, figure captions, figure references, and the two requested descriptive figures were changed. No experiment was added, no model was trained, and generated table contents were not changed.

## Figure order and compiled pages

The compiled manuscript has body figures numbered continuously from 1 to 16. Figure numbers and PDF pages were read from `build/main.aux` after the final compilation.

| Figure | Content | PDF page |
|---:|---|---:|
| 1 | Q1 overview | 3 |
| 2 | Q1 text feature extraction placeholder | 3 |
| 3 | Q1 audio feature extraction placeholder | 3 |
| 4 | Q1 video feature extraction placeholder | 4 |
| 5 | Q1 Attachment 1 descriptive statistics | 5 |
| 6 | Q1 temporal alignment | 7 |
| 7 | Q1 alignment-method comparison | 11 |
| 8 | Q1 typical-sample source trace | 13 |
| 9 | Q2 model architecture | 14 |
| 10 | Q2 Attachment 2 descriptive statistics | 15 |
| 11 | Q2 missing-ratio metrics | 17 |
| 12 | Q2 missing-modality/location stress test | 18 |
| 13 | Q2 seed stability and error analysis | 19 |
| 14 | Q3 HEAF framework | 22 |
| 15 | Q3 modality dominance and deletion validation | 24 |
| 16 | Q3 case explanations | 26 |
| A1 | Q1 modality-combination auxiliary results | 29 |

Q1 Figure 7 follows Table 3 on page 11. Figure 8 is declared in the typical-sample subsection. The appendix contains only the former Appendix Figure A2, renumbered as Figure A1. Q1 Figure 2/3/4 placeholders remain unchanged.

## Descriptive-figure source checks

### Figure 5: Q1

- Source manifest: `outputs/q1_final_local/05_appendix/q1_sample_summary_100.csv`.
- Label source: the Attachment 1 `label-100.xlsx` label sheet.
- The manifest has 100 rows and 100 unique sample IDs. A one-to-one merge on `(video_id, clip_id)` matched all 100 rows; both inputs had unique merge keys.
- Workbook annotation counts: Negative 18, Neutral 25, Positive 57.
- Per-sample valid-bin statistics read directly from the final manifest: text mean 41.21 (range 13–49), audio mean 49.55 (range 47–50), vision mean 50 (range 50–50).
- Text sources: official transcript 75 (73 with no fallback record, 2 with fallback records); media ASR 25 (20 with no fallback record, 5 with fallback records). Fallback-record counts per sample were 0: 93, 1: 4, 4: 1, 5: 1, and 8: 1.
- The 100-row figure source is retained as `figures/q1/fig05_q1_data_statistics_source.csv`; source paths and SHA256 digests are recorded in the adjacent JSON file.

### Figure 10: Q2

- Source: `E2026/data/raw/aligned_50.pkl`.
- Split sizes were read from each split's ID list: train 3395, valid 728, test 727.
- Train/valid class counts read from `classification_labels`: train Negative/Neutral/Positive = 967/758/1670; valid = 206/184/338.
- Train/valid continuous-label distributions read from `regression_labels`; plotted range is fixed to [-3, 3].
- Valid lengths were counted from `text_bert[:, 1, :]`: train min/median/max = 3/22/50; valid = 3/23.5/50. The valid field was checked to be a prefix-valid mask.
- Test contributes only its sample count to the split-size panel. Test labels and predictions are absent from the plotted train/valid source CSV and were not used for any label or performance analysis.
- Train/valid source rows, split counts, and source-file SHA256 are retained next to Figure 10.

## Reference, file, and layout checks

- Main-text figure labels resolve to 1–16 in order; the sole appendix figure resolves to A1.
- No stale Q1 alignment A1, modality A2, case A3, or appendix A2/A3 figure labels remain in manuscript section or appendix TeX sources.
- Figure paths in the TeX source use the renamed assets. Existing PNG/PDF/SVG/PPTX companion assets were renamed where present.
- PDF page count: 29. Visual inspection covered the figure pages, including the new Figures 5 and 10, Q1 Figures 6–8, the Q2/Q3 figure sequence, and Appendix Figure A1; no clipping or overlap was found.
- The `generated/*.tex` table-file SHA256 values match the pre-build values; table content is unchanged.
- Final `build/main.log`: undefined references 0; undefined citations 0; missing characters 0; overfull hboxes 0; underfull hboxes 0.
- The final PDF was compiled with XeLaTeX and is `build/main.pdf`.
