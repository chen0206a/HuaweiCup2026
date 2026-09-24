# Q3-2 Attachment4 grounding and interface audit

**Conclusion: `BLOCKED_BY_GROUNDING`; grounding level: `UNVERIFIED`.**

HEAF was locked before Attachment4 content inspection. This audit did not load the predictor or generate final explanations/CSV.

## Aligned files and ID correspondence

Aligned set: 20 pkl + 20 mp4; exact feature-ID→video-stem matches 20.
Missing feature/video: 0/0; duplicate feature IDs 0; ambiguous mappings 0; filename/embedded-ID mismatches 0.
All aligned files matched the pre-existing size/SHA256 inventory: True.

## Feature interface

Every sample has keys ['audio', 'id', 'raw_text', 'text', 'text_bert', 'vision']; required shapes text/audio/vision/text_bert = [50,768]/[50,74]/[50,35]/[3,50].
Numeric finite and binary valid-prefix mask checks: True; valid length range 13–50. Padding comes only from text_bert[1]; native zero vectors are kept as data.
raw_text present in 20/20 samples; no label fields. Actual timestamp/alignment metadata keys: [].
text_bert permits exact recovery of integer token IDs and mask; all samples begin/end with observed IDs [101]/[102]. The supplied files have no tokenizer vocabulary, token offsets, word timestamps or feature-slot→media table. Slot counts differ from whitespace word counts in 19 samples, which rules out a direct one-word-per-slot assumption.

## What the videos provide

All 20 paired MP4s opened in OpenCV. Nominal duration range from reported frame_count/FPS: 3.600–24.100 seconds. Sequential decoding gives 93–639 frames and last-frame positions 3.067–21.300 seconds; 16 files have a reported/decoded frame-count mismatch.
This environment has no ffprobe executable or raw PTS API in OpenCV. Decoder-reported frame positions are not an alignment from feature slots to video/audio seconds.

## Grounding decision

The pkl gives whole-sample raw_text and BERT token IDs, but no per-slot text span. Audio/vision arrays give no timestamp or source frame index. MP4 timing cannot resolve which feature slot came from which time. We therefore return `unverified` for interval grounding: text_fragment, audio_time_range and video_frame_time stay null. Exact feature ID→MP4 identity is verified separately.
No `duration/50`, character-proportional text slicing or Q1 uniform-bin rule was used.

## At most two candidate ways forward (not implemented)

1. **Recover authoritative preprocessing provenance.** Obtain the source aligned-feature generation record with per-slot token offsets, word/audio times and visual frame/PTS indices, linked by exact sample ID and compatible with these file hashes. Validate every mapping against raw_text and MP4. If complete, interval grounding could be VERIFIED; missing offsets or undocumented resampling remain errors.
2. **Reconstruct and validate approximate alignment from provided media.** Fix the exact BERT tokenizer/version, align raw_text to the paired audio and video PTS using a documented forced-alignment/decoding pipeline, then compare reconstructed slot indices with feature preprocessing. Word alignment, subword splits, truncation, ASR mismatch, video edits and decoder timing create uncertainty. Report sample-specific error bounds and use APPROXIMATE only after independent spot checks; otherwise remain UNVERIFIED.

The second route is a proposal, not an automatically accepted replacement for absent official metadata. No HEAF parameter may be retuned from Attachment4.
