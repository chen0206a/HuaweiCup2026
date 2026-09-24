# Q3-2.6 Text Feature Row Identity Audit

## Purpose

Test whether `Attachment4 text[i]` is the BERT token hidden-state row for `text_bert` slot `i`. No P2 model, HEAF output, label, or training data was loaded. Only 20 aligned Attachment4 samples were read for this provenance check.

## Public preprocessing evidence

- MMSA `BertTextEncoder` constructs `BertModel.from_pretrained("bert-base-uncased")` and returns output tuple `[0]`, preserving the token dimension.
- Self-MM `DataPre.py` tokenizes raw text with `[CLS]`/`[SEP]`, and its `__getTextEmbedding` returns `model(input_ids)[0]` directly with no pooling.
- MMSA-FET's BERT extractor likewise returns `BertModel.last_hidden_state` directly.
- The reviewed paths support the final hidden layer only. None supports layer averaging or pooled sentence representations. The public pipeline names `bert-base-uncased` but does not pin the exact historical checkpoint revision for Attachment4.

## Reconstruction and checks

Used provided `text_bert` IDs and token type IDs for each valid prefix, kept `[CLS]` and `[SEP]`, ran the pinned `bert-base-uncased` revision in eval/no-grad mode, and compared `last_hidden_state` row-by-row to the provided `text` feature. The reviewed Self-MM extractor does not call `.eval()` explicitly; Hugging Face `from_pretrained` instantiated in eval mode in this run, and the audit asserted eval mode explicitly. The model's valid prefix was unpadded, matching the public extractor call. No alternate representation was searched.

- 20 samples, 604 valid token rows.
- Same-index row was the full cosine-matrix argmax for **604/604 rows**.
- Mean same-index cosine: **0.999999994**.
- Mean best off-diagonal cosine: **0.690571460**.
- Mean diagonal margin: **0.309428544**.
- Global maximum absolute difference: **3.0041e-5**.
- Row-weighted RMSE: **8.3977e-7**.

The numerical match is at float32 precision and the public extractors return the sequence of token states without reordering or pooling. Together with the previous exact reconstruction of `text_bert` IDs/masks/type IDs, this verifies `text feature row i ↔ text_bert token slot i`. The audit does not prove which historical copy of the named checkpoint was used, but this uncertainty does not change the token row ordering.

## Grounding consequence

Text grounding is now `VERIFIED`: the existing pinned tokenizer rechecks IDs, attention mask, and type IDs per sample, then uses its `offset_mapping` to return raw-text character spans. `[CLS]`/`[SEP]` slots are excluded from the fragment; a special-token-only interval returns no fragment. Audio and vision stay `UNVERIFIED` and are not promoted.

Overall status: **`PARTIAL_GROUNDING_READY`**. This is an interface/provenance result only. No full Attachment4 HEAF inference or final explanation cards were generated.

## Files and reproducibility

- `metrics.json`: per-sample and aggregate numerical diagnostics, pinned candidate revision/hash, and source references.
- `row_cosine_matrices.npz`: compressed full valid-row cosine matrices for all 20 samples.
- `figures/sample_01_text_row_cosine_heatmap.png`: deterministic sample-01 provenance heatmap; not a paper main figure.
- `../../scripts/run_q3_text_row_identity.py`: reruns the audit with the same pinned candidate.
- `src/q3/text_grounding.py` and `src/q3/evidence_grounding.py`: consume the verified row identity to expose verified text fragments.

## Risks / limits

- Historical source does not pin the exact BERT weight revision; report this explicitly.
- Numerical differences are nonzero but tiny; report the values rather than claim bitwise identity.
- This audit says nothing about audio/video feature timing, and does not authorize full Attachment4 explanations.
