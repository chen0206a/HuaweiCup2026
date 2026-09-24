# Q3-1 HEAF validation

Status: **HEAF_VALIDATION_PASSED**.

Only Attachment2 valid was indexed and evaluated. No training or Attachment3/4 use.
The official aligned pickle is monolithic, so loading valid deserializes the container; test is never indexed or evaluated.

## Frozen predictor gate

Passed: **True** (absolute tolerance 1e-06).
Actual: Accuracy 0.648351648, Macro-F1 0.623434797, MAE 0.608260350, Pearson 0.646170374.
Maximum locked-metric delta: 4.01e-09.

## Grouped validation split

Design: 332 clips / 113 video IDs; audit: 396 clips / 126 video IDs. No video ID overlap.

## Exact coalitions

Eight coalitions/sample; maximum efficiency errors: class 1.78e-15, regression 4.44e-16. Mask semantics: True across 728 samples.
Negative pair values are reported only as negative interactions.

Classification primary counts: {'text': 656, 'vision': 65, 'audio': 7}; regression primary counts: {'text': 620, 'vision': 104, 'audio': 4}; agreement 0.839.

| Target | Pair | Mean interaction | Negative fraction |
|---|---|---:|---:|
| class | text_audio | -0.070613 | 0.694 |
| class | text_vision | 0.078082 | 0.376 |
| class | audio_vision | -0.037247 | 0.644 |
| reg | text_audio | -0.159219 | 0.857 |
| reg | text_vision | 0.050273 | 0.326 |
| reg | audio_vision | 0.059207 | 0.139 |

## Window selection (design subset only)

Chosen ρ: **0.30**; stride 1; 32 random same-length draws, seed 20260924.
| ρ | Mean top−random class-margin drop |
|---:|---:|
| 0.10 | 0.268837 |
| 0.20 | 0.366684 |
| 0.30 | 0.440487 |

## Faithfulness (audit subset only)

| Outcome | Top mean | Random mean | Top−random mean | Grouped 95% CI | Fraction top>random | Negative top fraction |
|---|---:|---:|---:|---|---:|---:|
| class_margin | 0.482771 | 0.082036 | 0.400735 | [0.363538062876364, 0.43724302017754724] | 1.000 | 0.000 |
| confidence | 0.093286 | 0.015567 | 0.077718 | [0.07116985105471103, 0.0842816588296985] | 1.000 | 0.000 |
| class_logit | 0.292982 | 0.053294 | 0.239688 | [0.21945767878490963, 0.2602978374003637] | 0.995 | 0.005 |
| regression | 0.002082 | 0.020177 | -0.018095 | [-0.052631934178229284, 0.0141006960480883] | 0.523 | 0.487 |
| regression_absolute | 0.202760 | 0.099215 | 0.103545 | [0.08829899487173987, 0.12026171643550057] | 0.894 | 0.000 |

The interval selected by the largest margin drop is compared with random intervals from that same evaluated window set. Its margin advantage is partly built into selection; treat it as a selection sanity check, not independent proof of explanation quality.

### Deletion curve

| Deleted valid positions | Mean margin top−random | Grouped 95% CI | Top>random fraction | Negative top fraction | Mean confidence top−random |
|---:|---:|---|---:|---:|---:|
| 10% | 0.108296 | [0.09247486761436913, 0.1250765755751227] | 0.841 | 0.114 | 0.020766 |
| 20% | 0.237758 | [0.21213547683515188, 0.26541115712748936] | 0.919 | 0.056 | 0.045785 |
| 30% | 0.418076 | [0.3814606575783341, 0.45444965700824885] | 1.000 | 0.000 | 0.081359 |
| 40% | 0.465465 | [0.4217788883568801, 0.5093878674357768] | 0.987 | 0.003 | 0.091018 |

Grouped intervals and sample-level results are in the JSON/JSONL artifacts.

## Seed sensitivity on fixed audit subset

ρ and audit IDs were not reselected. Seed42 remains the main predictor.

- Seed 43: predicted-class agreement 0.896; class primary agreement 0.869; regression primary agreement 0.778; class/regression Shapley rank correlation 0.758/0.745; mean same-modality interval IoU 0.579; top−random direction agreement 1.000.
- Seed 44: predicted-class agreement 0.896; class primary agreement 0.831; regression primary agreement 0.783; class/regression Shapley rank correlation 0.694/0.760; mean same-modality interval IoU 0.527; top−random direction agreement 1.000.

Rank comparisons may involve different predicted classes; conditional same-class values are in metrics.json. Direction agreement for the selected interval has the same selection limitation as the main interval check.

## Representative audit cases

Positive interval cases (largest top−random margin):
- `258654$_$3` (text): +3.0246
- `101787$_$5` (text): +2.2662
- `273539$_$9` (text): +2.1132
- `266791$_$4` (text): +2.0937
- `273539$_$3` (text): +1.9684

Failure cases (smallest 10% deletion top−random margin):
- `80855$_$10` (text): -0.1657
- `273539$_$2` (text): -0.1548
- `5i6AgQSjXO4$_$2` (text): -0.1538
- `dt641WxonBI$_$1` (text): -0.1345
- `241629$_$5` (text): -0.1247


Interpretation: these are frozen-model zero-intervention explanations, not real-world causal effects or raw-media grounding.
