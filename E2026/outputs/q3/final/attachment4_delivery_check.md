# Attachment4 final HEAF delivery check

**Status: `Q3_FINAL_INFERENCE_COMPLETE`.**

Coverage: 20/20; unique IDs: True; exact feature↔MP4 pairs: 20/20.
Checkpoint SHA256 unchanged: `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`; method schema: 0.2.0 / `25abe930caad7f0b4a3cb867716bf00a4b58ea554af96cd66e89f5b21809e2d2`.
Input shapes, finite values, masks and file inventory hashes passed. Maximum Shapley efficiency errors: class `8.88e-16`, regression `4.44e-16`.
All 20 explanation cards validated against the locked JSON Schema. Text fragments were rechecked against exact raw_text character spans. A/V grounding remains unverified; A/V time/frame fields are null in JSONL and `NA` in CSV.
Attachment4 contains no labels; no accuracy, F1, MAE, Pearson, correctness, Attachment2 test, or Attachment3 result is included or used.
Faithfulness values describe frozen-model intervention effects only; they do not establish prediction correctness or real-world causal effects.
Typical case nominations use the fixed statistical rules recorded in `attachment4_summary.json`; no manual visual selection was used.

## Per-modality and grounding summary

```json
{
  "predicted_class_counts_and_proportions": {
    "Negative": {
      "count": 7,
      "proportion": 0.35
    },
    "Neutral": {
      "count": 5,
      "proportion": 0.25
    },
    "Positive": {
      "count": 8,
      "proportion": 0.4
    }
  },
  "classification_primary_modality_counts": {
    "text": 19,
    "vision": 1
  },
  "regression_primary_modality_counts": {
    "text": 18,
    "vision": 2
  },
  "primary_agreement": {
    "count": 17,
    "n": 20,
    "proportion": 0.85
  },
  "grounding_status_counts": {
    "verified": 19,
    "unverified": 1
  },
  "verified_text_explanation_count": 19,
  "unverified_av_primary_count": 1,
  "representative_candidates": {
    "text_primary_high_faithfulness": {
      "sample_id": "14",
      "criterion": "maximum q=10% top-minus-random class-margin drop among classification text-primary samples; ID tie-break",
      "value": 0.40079098542954716
    },
    "vision_primary": {
      "sample_id": "02",
      "criterion": "maximum absolute classification Shapley for vision among vision-primary samples; ID tie-break",
      "phi": 0.8293745537061559
    },
    "audio_primary": null,
    "strong_interaction": {
      "sample_id": "16",
      "pair": "text_vision",
      "criterion": "maximum absolute classification pair-interaction value; sample ID then fixed pair order tie-break",
      "value": -0.7237452453956196
    },
    "weak_or_failure_boundary": {
      "sample_id": "19",
      "top_minus_random_margin_drop": -0.11243989743420653,
      "primary_modality_classification": "text",
      "criterion": "minimum q=10% top-minus-random class-margin drop; sample ID tie-break"
    }
  }
}
```

## Files

- `attachment4_predictions_explanations.csv` (CSV null rule: explicit `NA`)
- `attachment4_explanations.jsonl` (JSON null rule: `null`; locked schema 0.2.0)
- `attachment4_summary.json` (unlabeled distributions and deterministic representative candidates)
