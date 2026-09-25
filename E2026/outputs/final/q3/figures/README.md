# E2026 Q3 Figures 7–9

## Figure 7 v2 — Q3 framework from the Q2 architecture layout

`figure7_q3_framework_zh_v2.pptx` is an editable copy of the user-provided Q2 architecture slide. Its four dashed stage containers retain the source layout, while their contents show Q3 sentiment prediction, HEAF explanation and evidence output. The left text quote, audio waveform and original video scene all come from paired Attachment4 sample 09, which differs from the video in the Q2 source slide and from the two current Figure 9 cases. The scene is not an HEAF key frame; audio/visual raw-time grounding remains unverified. See `figure7_q3_framework_zh_v2_README.md` and `figure7_q3_framework_zh_v2_sources.json` for the source hashes and extraction rule. PNG/PDF/SVG are exported from the PPTX; earlier Figure 7 files remain unchanged.

These figures use the frozen Q3 results. The current Figure 8 and Figure 9 v2 are Chinese paper figures with a shared muted palette, dark bordered dotted bars, white backgrounds and restrained axes. Figure 8 uses **Attachment2 validation** results; Figure 9 retains the two fixed **Attachment4** cases (14 and 02) and original-video scene images. Attachment4 has no labels, so Figure 9 is case interpretation only and is not a performance figure. Earlier versions remain in `archive/`.

## Reproduce

From `D:/华为杯/E2026` with Python 3 and `matplotlib`, `numpy`:

```powershell
python outputs/final/q3/figures/scripts/figure8_q3_faithfulness_zh.py
python outputs/final/q3/figures/scripts/extract_figure9_context_frames.py
python outputs/final/q3/figures/scripts/figure9_q3_case_studies_v2.py
```

The first script reads the frozen validation metrics and generates the current Figure 8. The last two commands regenerate the fixed-position scene frames and Figure 9 v2; run the frame extractor before its plotter. The original scripts are retained as legacy sources: running `figure8_q3_faithfulness.py` would overwrite the current Figure 8 files, while `figure9_q3_case_studies.py` writes the separate v1 filenames. Both final figures use the user-selected pale text/audio/vision colors (`#BFDCE6`, `#EEE7B0`, `#E9C9CC`), deep-blue trend line (`#4F7F95`), Microsoft YaHei, light gray guides and dotted bar hatches.

## Figure 8 — 中文版解释有效性与主导模态统计

**Source files:**

- `E2026/outputs/q3/heaf_validation_metrics.json` — audit deletion curves, bootstrap intervals, validation primary-modality counts, group counts.
- `E2026/outputs/q3/heaf_validation_report.md` — interpretation limits and validation context.
- `E2026/configs/final/q3_heaf_validation.yaml` — frozen validation protocol reference.

| Panel | Content and data source | Asset/style reference |
|---|---|---|
| A | Classification vs regression primary-modality counts over all 728 Attachment2 valid samples: Text 656/620, Vision 65/104, Audio 7/4. Pale modality colors and sparse/dense dotted hatches identify the two heads. | `assets/figures/GroupedBarChart/plot_GroupedBarChartv1.py`; grouped bar spacing and direct count labels adapted. |
| B | Top-interval vs same-length random deletion class-margin drop at 10/20/30/40%; means and video-group bootstrap 95% CIs. Audit group: 396 clips, 126 video IDs, 1,000 bootstrap replicates. | `assets/figures/LineTrend/plot_trend.py`; line/marker and sparse-axis parameters inherited. |
| C | Top-minus-random class-margin mean difference with video-group bootstrap 95% CIs. | `assets/figures/BarComparison/plot_comparison_Trajectory.py`; direct comparison and zero-reference styling adapted. |

Intermediate data remain `data/figure8_deletion_curve.csv`, `data/figure8_primary_modality_counts.csv`, and `data/figure8_margin_gain_summary.csv`. Confidence bands in B use the separately reported top/random grouped-bootstrap intervals; C uses the directly reported grouped-bootstrap interval for the paired top-minus-random difference. There are no p-values or significance tests in this figure; the plotted quantities are descriptive intervention summaries with bootstrap uncertainty. The Chinese plotter reads `heaf_validation_metrics.json` directly and does not rewrite these source tables.

The top interval is selected as the maximum over candidate windows, so B's top-vs-random advantage has a selection advantage by construction. The figure caption/report should describe this as a perturbation-consistency check, not independent proof of explanation quality or a causal effect. This is Attachment2 validation evidence; no Attachment4 output contributes to Figure 8. See `figure8_zh_README.md` for the proposed Chinese caption and QA summary.

## Figure 9 v2 — Attachment4 typical explanation cases

The paper figure has two rows: **A, Sample 14** (text-primary, high faithfulness, verified text grounding) and **B, Sample 02** (vision-primary, raw visual grounding unverified). Each row shows the frozen prediction, classification Shapley values, and the stored feature-slot temporal importance curve. A shows the exact verified text fragment from `raw_text`. B shows only the feature-space interval; no video time, feature-aligned frame, or raw visual location is claimed. The original-language text fragment is preserved verbatim even though the explanatory labels are Chinese.

**Suggested Chinese caption:**

> 图9 Attachment4典型样本的多层级解释结果。(a) 样本14为文本主导案例，给出三模态Shapley贡献、时间遮挡曲线及经验证的原始文本证据。(b) 样本02为视觉主导案例，HEAF能够定位视觉特征空间中的关键区间，但由于附件未提供可核验的视觉特征槽位—视频时间映射，原始视觉位置保持未验证。图中视频帧仅用于展示对应样本的原始场景，不表示HEAF定位的关键视觉帧。

Video frames come from the paired MP4s via `extract_figure9_context_frames.py`. Sample 14 requests 50% of reported duration; Sample 02 requests 25%, 50%, and 75%. Selection reads only MP4 metadata and decoded frames, never HEAF scores or temporal peaks. `data/figure9_video_frame_manifest.json` records source MP4 path/hash, requested relative position, actual decoded zero-based frame index, nominal time derived from index/FPS when OpenCV PTS is unreliable, and the required `context_only` / `not_keyframe_mapping` status. The decoder stops before the MP4-reported tail in both files (Sample 14: 308/415 frames, Sample 02: 101/108 frames); all four requested positions were nevertheless directly decoded. Timing is scene-selection metadata, not a slot-to-time mapping.

| Scene frame | Requested | Decoded index (zero-based) | Index/FPS time |
|---|---:|---:|---:|
| Sample 14 | 50% | 208 | 6.933 s |
| Sample 02 | 25% | 27 | 0.900 s |
| Sample 02 | 50% | 54 | 1.800 s |
| Sample 02 | 75% | 81 | 2.700 s |

The v2 plotter validates that both case records come from `attachment4_explanations.jsonl`, their fixed primary modalities and raw grounding statuses match, and Sample 14's displayed fragment equals the `[0,38)` raw-text span in the original `14.pkl`. Shapley values, predictions, and temporal curve values are read verbatim from the final JSONL. The shaded intervals are **feature-slot** intervals `[1,12)` and `[1,7)`, not media times. No model inference or result recalculation occurs.

The Figure 9 v1 PNG/PDF/SVG remain at their original filenames and have identical copies in `archive/`. Sample 16, Sample 19, pair interactions, regression Shapley detail, and other full explanation fields remain in the v1 figure, four case-card JSON files, and original final results; they are omitted only from the v2 paper figure.

**v2 QA:** Both plotted case records were loaded from the final JSONL; no prediction, Shapley or curve values were regenerated. The Sample 14 fragment was checked against the original `14.pkl` raw-text span. Sample 02 retains null raw visual timing and an explicit unverified label. All screenshots were selected from fixed MP4 duration fractions independent of HEAF. The old Figure 9 files and archive copies have matching SHA256 values; the final prediction/explanation files were not modified. The SVG parses as XML; the current PDF has one 183 × 140 mm page and the PNG is 300 dpi. The labels are legible at the intended 183 mm width. This v2 is ready for a Huawei Cup Chinese manuscript draft with the caption above.

The Figure 9 v2 Scheme C update changed only the text/audio/vision hues, temporal line weight, white-background interval tint, and thin gray grid. Its earlier two-case raster/vector files are in `archive/figure9_q3_case_studies_v2_before_scheme_c.*`. The paired MP4 screenshots, explanations and grounding statuses were not modified.

The current pale-color update preserves the same two cases and video frames. It adds dark outlines and dotted hatches to Shapley bars, uses a common deep-blue temporal line and a pale orange feature-slot highlight, and moves the two short grounding notes into light boxes. The preceding Scheme C exports are in `archive/figure9_q3_case_studies_v2_scheme_c.*`. The raw English text fragment remains verbatim evidence; all explanatory labels are Chinese. Figure 9 stays at two cases under the previously agreed case selection, since this revision changes presentation only.

## Figure 9 v1 — archived four-case version

**Source files:**

- `E2026/outputs/q3/final/attachment4_explanations.jsonl` — frozen predictions, Shapley values, interactions, slot-level temporal curves, faithfulness interventions and raw-evidence status.
- `E2026/outputs/q3/final/attachment4_summary.json` — deterministic candidate IDs and selection criteria.
- `E2026/outputs/q3/final/attachment4_predictions_explanations.csv` and `attachment4_delivery_check.md` — delivery/QA cross-check.

| Panel | Fixed case and displayed evidence | Selection/grounding |
|---|---|---|
| A | Sample 14: prediction card; classification Shapley and three pair interactions; classification-primary temporal curve with key interval; regression Shapley values as a compact secondary annotation; verified raw-text fragment and character offsets. | Automatically nominated `text_primary_high_faithfulness`; text row → verified `text_bert` token slot → tokenizer offset → raw-text span. |
| B | Sample 02: prediction, classification Shapley, vision-primary temporal curve, explicit unverified raw-media note. | Automatically nominated `vision_primary`; no frame timestamp or media time is supplied. |
| C | Sample 16: prediction, all three classification pair interactions, temporal curve, and strongest-pair annotation. | Automatically nominated `strong_interaction`; maximum absolute classification pair interaction. |
| D | Sample 19: prediction, classification Shapley, temporal curve and q=10% top-minus-random margin difference. | Automatically nominated `weak_or_failure_boundary`; minimum q=10% top-minus-random value. |

Case files `data/figure9_sample14_card.json`, `sample02`, `sample16`, and `sample19` retain the original JSONL explanation fields and add only figure-panel/selection provenance. They are not new model outputs. The selected cases are not manually replaced. Audio/vision raw-media timing remains unverified; intervals in those modalities are feature-slot evidence only. Verified text is shown only for Sample 14 in the hero panel; it is a secondary evidence display and does not alter the selected primary modality.

The temporal curves plot the locked continuous-window class-margin drop at each valid window center; the shaded band marks the reported feature-slot interval `[start, end)`. Slot indices do not encode seconds or frame timestamps. Figure 9 is a deterministic case display (n=4 cases), with no aggregate statistics or label-based performance measures.

**Figure 9 style references:** `assets/figures/LineTrend/plot_trend.py` for temporal line treatment; `assets/figures/GroupedBarChart/plot_GroupedBarChartv1.py` for modality bars; `assets/figures/BarComparison/plot_comparison_Trajectory.py` for direct signed comparisons. The local `assets/figures/multipanel/` directory contains no production plot script, so the asymmetric mixed composition uses Matplotlib `GridSpec`. The heatmap asset was reviewed but not used: three signed pairwise values are more directly read as bars.

The style inspiration is the public [academic-figure-skill repository](https://github.com/TingxiYu/academic-figure-skill), including its `LineTrend`, `GroupedBarChart`, `BarComparison`, typography, palette, and vector-export conventions. Figure 8 uses Python/Matplotlib instead of the requested initial R/ggplot2 preference because no R runtime is installed here; after checking, the user authorized the Python backend. No R script or silent backend substitution is involved.

## Outputs

- `figure8_q3_faithfulness.png`, `.pdf`, `.svg`
- `archive/figure8_q3_faithfulness_v1.png`, `.pdf`, `.svg` (English original)
- `archive/figure8_q3_faithfulness_scheme_c.png`, `.pdf`, `.svg` (previous Chinese palette)
- `figure8_zh_README.md` (Chinese caption, source and QA)
- `figure9_q3_case_studies.png`, `.pdf`, `.svg`
- `figure9_q3_case_studies_v2.png`, `.pdf`, `.svg` (300 dpi PNG; editable vector PDF/SVG)
- `archive/figure9_q3_case_studies.png`, `.pdf`, `.svg` (unchanged v1 copies)
- `archive/figure9_q3_case_studies_v2_before_scheme_c.png`, `.pdf`, `.svg`
- `archive/figure9_q3_case_studies_v2_scheme_c.png`, `.pdf`, `.svg`
- `data/sample14_context_frame.png`, `data/sample02_context_25.png`, `data/sample02_context_50.png`, `data/sample02_context_75.png`
- `data/figure9_video_frame_manifest.json`
- 300 dpi raster previews duplicated in `preview/`
- reproducible plotting scripts in `scripts/`

PNG previews were rendered and visually reviewed after the final layout adjustments. PDF and SVG exports retain vector drawing/text, with video stills as embedded raster images. Figure 8 is a double-column-width quantitative multipanel figure; Figure 9 v2 is a double-column-width, two-case explanation figure at 183 mm width. Check the final manuscript export if the document editor rescales it below this width.
