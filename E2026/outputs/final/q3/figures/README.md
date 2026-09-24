# E2026 Q3 Figures 8–9

These figures use the frozen Q3 results. Figure 8 summarizes **Attachment2 validation audit** interventions; Figure 9 shows four fixed, automatically nominated **Attachment4** explanation cards. Attachment4 has no labels, so Figure 9 is case interpretation only and is not a performance figure.

## Reproduce

From `D:/华为杯/E2026` with Python 3 and `matplotlib`, `numpy`:

```powershell
python outputs/final/q3/figures/scripts/figure8_q3_faithfulness.py
python outputs/final/q3/figures/scripts/figure9_q3_case_studies.py
```

The scripts regenerate the intermediate CSV/JSON files, the PNG/PDF/SVG outputs, and copies of the PNG files under `preview/`. Both scripts set the same Arial-first typography, semantic modality colors, line weights, spine rules, and panel-label style.

## Figure 8 — Q3 faithfulness and modality evidence

**Source files:**

- `E2026/outputs/q3/heaf_validation_metrics.json` — audit deletion curves, bootstrap intervals, validation primary-modality counts, group counts.
- `E2026/outputs/q3/heaf_validation_report.md` — interpretation limits and validation context.
- `E2026/configs/final/q3_heaf_validation.yaml` — frozen validation protocol reference.

| Panel | Content and data source | Asset/style reference |
|---|---|---|
| A | Top-interval vs same-length random deletion class-margin drop at 10/20/30/40%; means and video-group bootstrap 95% CIs. Audit group: 396 clips, 126 video IDs, 1,000 bootstrap replicates. | `assets/figures/LineTrend/plot_trend.py`; line/marker and sparse-axis parameters inherited. |
| B | Classification vs regression primary-modality counts over all 728 Attachment2 valid samples: Text 656/620, Audio 7/4, Vision 65/104. | `assets/figures/GroupedBarChart/plot_GroupedBarChartv1.py`; grouped bar spacing and direct count labels adapted. |
| C | Top-minus-random class-margin mean difference with video-group bootstrap 95% CIs. | `assets/figures/BarComparison/plot_comparison_Trajectory.py`; direct comparison and zero-reference styling adapted. |

Intermediate data are `data/figure8_deletion_curve.csv`, `data/figure8_primary_modality_counts.csv`, and `data/figure8_margin_gain_summary.csv`. Confidence bands in A use the separately reported top/random grouped-bootstrap intervals; C uses the directly reported grouped-bootstrap interval for the paired top-minus-random difference. There are no p-values or significance tests in this figure; the plotted quantities are descriptive intervention summaries with bootstrap uncertainty.

The top interval is selected as the maximum over candidate windows, so A's top-vs-random advantage has a selection advantage by construction. The figure caption/report should describe this as a perturbation-consistency check, not independent proof of explanation quality or a causal effect. This is Attachment2 validation evidence; no Attachment4 output contributes to Figure 8.

## Figure 9 — Attachment4 explanation case studies

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
- `figure9_q3_case_studies.png`, `.pdf`, `.svg`
- 300 dpi raster previews duplicated in `preview/`
- reproducible plotting scripts in `scripts/`

PNG previews were rendered and visually reviewed after the final layout adjustments. PDF and SVG exports retain vector drawing/text. At final manuscript sizing, Figure 8 is a double-column-width quantitative multipanel figure; Figure 9 is a double-column-width, taller asymmetric case-study figure. The high-density bottom case panels in Figure 9 use compact labels; verify them after any manuscript software rescales the figures.
