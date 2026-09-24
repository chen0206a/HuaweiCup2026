# Q3-2.7 Approximate Audio/Visual Media Grounding Audit

## Decision

**Stop before time mapping.** Public sources show compatible feature schemas and describe possible aligned extraction pipelines, but the available evidence does not prove that Attachment4's `text[i]`, `audio[i]`, and `vision[i]` are the same aligned semantic position. Therefore no forced alignment or A/V feature-slot time mapping was run.

## Evidence reviewed

- Q3-2.6 verifies Attachment4 `text[i] ↔ text_bert` token slot `i` for the pinned BERT final-layer representation.
- Existing Attachment4 interface audit records 20/20 exact feature ID to paired MP4 stem matches, shapes `[50,768]`, `[50,74]`, `[50,35]`, `[3,50]`, and valid lengths 13–50.
- Attachment4 metadata contains no `align` records, slot timestamps, audio intervals, frame indices/PTS, original source IDs, producer config, or hash lineage to a public feature release.
- MMSA documents a compatible aligned_50 schema and BERT text features; its loader consumes the precomputed arrays. This establishes format compatibility, not the producer or exact row semantics of these files.
- MMSA-FET documents one explicit alternative: align raw media to words, map words to BERT WordPieces, copy pooled A/V word features across token rows, and retain `align_result`. Attachment4 has neither those records nor provenance proving this pipeline was used.
- CMU-MultimodalSDK alignment operates against a named reference sequence with timestamp intervals. Attachment4 has no source CSD ID or sequence intervals to connect it to that interface.

The BERT row match, dimensions, common valid prefix, zero padding, and exact clip filename pairing are individually useful, but none proves the required A/V row-to-token correspondence for these exact files. Dimensions and array positions alone cannot identify word/segment semantics.

## Gate outcome

`aligned_semantic_position_supported_for_attachment4 = false`.

The audit stopped before MP4 audio extraction, forced alignment, video decoding/PTS extraction, interval construction, and per-slot grounding. This follows the required gate and avoids assigning approximate times to feature rows whose relation to text slots is unestablished. No samples were selected using HEAF outputs; no predictions or labels were read.

## Status

- Text: `VERIFIED`
- Audio: `UNVERIFIED`
- Vision: `UNVERIFIED`
- Overall: `PARTIAL_GROUNDING_READY`

The existing `evidence_grounding.py` already fails closed for A/V without a reviewed mapping. No code change was needed, and no approximate mapping was introduced.

## Reopen condition

Revisit this gate only if a producer manifest/source record links the exact Attachment4 features to a preprocessing version and supplies or reconstructs a per-slot link from the verified token/word positions to A/V feature rows. The record must include the word-to-WordPiece expansion rule. Only after that link is established should a fixed forced aligner and fixed video timestamp method be evaluated for approximate timing.
