# Q2 Figure 4/5 per-scenario completion check

Status: **COMPLETE**
Run date: 2026-09-24
Evaluation scope: Attachment2 valid only; no training, tuning, test, or Attachment3.

## Preflight

- B0 seed43 checkpoint SHA256: `cf51892f79f135faf5e38ff338363552b52aa68b5616c99ea6489ca73f52a3de` — manifest match.
- B0 seed44 checkpoint SHA256: `45d60973f3e1b0312fbc88e9f45cbe0a594b75288c59a27fbd781699aeeccc4a` — manifest match.
- Attachment2 valid samples: 728; valid timesteps: 18,628; feature dimensions: text 768 / audio 74 / vision 35.
- Benchmark definition SHA256: `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`; seed 20260923; 54 missing conditions (45 single-modality + 9 double-modality) plus clean.
- Evaluation reused `src/evaluation/missing_benchmark.py`; continuous-block semantics, valid-prefix mask, padding preservation and native-zero handling were unchanged.
- Each new seed produced clean + 54 conditions (55 rows). Clean metrics matched that seed's checkpoint manifest within absolute tolerance 1e-6.

## Completed data

| Output | Rows | Aggregation |
|---|---:|---|
| `scenario_details_complete.csv` | 330 | 2 models × 3 seeds × 55 conditions |
| `fig4_modality_by_rho_per_seed_complete.csv` | 90 | model × seed × modality × rho; averages three positions within each condition |
| `fig4_modality_by_rho_mean_sd.csv` | 30 | mean and sample SD across 3 seeds |
| `fig5_modality_by_location_per_seed_complete.csv` | 54 | model × seed × modality × position; averages five ratios |
| `fig5_modality_by_location_mean_sd.csv` | 18 | mean and sample SD across 3 seeds |
| `fig5_double_modality_location_per_seed.csv` | 54 | TA/TV/AV × model × seed × position; frozen rho=0.3 |
| `fig5_double_modality_location_mean_sd.csv` | 18 | TA/TV/AV mean and sample SD across 3 seeds |

Mean/SD treats seeds as the independent repeats (`ddof=1`); scenario conditions are not treated as independent replicates. Existing incomplete source CSVs were retained unchanged.

**FIG4_DATA_COMPLETE = YES**
**FIG5_DATA_COMPLETE = YES**
