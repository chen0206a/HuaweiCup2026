# Q3-2.7 Approximate Audio/Visual Media Grounding Audit

**Gate result: `FAILED — STOPPED BEFORE TIME MAPPING`.**  
**TEXT = `VERIFIED`; AUDIO = `UNVERIFIED`; VISION = `UNVERIFIED`.**  
**Overall: `PARTIAL_GROUNDING_READY`.**

## Why the gate failed

Q3-2.6 established `text[i] ↔ text_bert token slot i`. The Attachment4 inventory also establishes exact sample ID to paired MP4 filename matching for all 20 files. Those facts do not establish that `audio[i]` and `vision[i]` represent that same token or word position.

The reviewed Attachment4 records have `[50,768]` text, `[50,74]` audio, `[50,35]` vision, and `[3,50]` `text_bert`; the valid prefix comes from the text mask. The audit found no per-slot `align` data, audio time intervals, video frame indexes or PTS, original source IDs, producer configuration, or hash lineage linking these sample files to a public feature release.

The public MMSA schema documents an `aligned_50` feature package and says its MOSEI text features are pre-extracted BERT; its loader consumes prebuilt arrays. This establishes interface compatibility, not the provenance or row semantics of these exact Attachment4 files. [MMSA README](https://github.com/thuiar/MMSA/blob/a94e65d07fa1ae0d44e552390074b29b0898edfd/README.md), [MMSA data loader](https://github.com/thuiar/MMSA/blob/a94e65d07fa1ae0d44e552390074b29b0898edfd/src/MMSA/data_loader.py).

MMSA-FET demonstrates one specific word-to-token alignment method: align media to words, map words to BERT WordPieces, expand pooled audio/video word features to token rows, and retain `align_result`. Attachment4 contains no such alignment record or producer marker tying it to that pipeline. [MMSA-FET alignment implementation](https://github.com/thuiar/MMSA-FET/blob/f8fbd2d88d4f77580ea1ded0b3469073488c5c19/src/MSA_FET/dataset.py).

The CMU-MultimodalSDK aligns sequences using an explicit named reference sequence and timestamp intervals. Attachment4 has no source CSD ID or intervals with which to identify that reference or join its rows. [CMU-MultimodalSDK alignment implementation](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/mmsdk/mmdatasdk/dataset/dataset.py).

So the public sources establish plausible aligned data conventions, but they do not establish shared semantic positions for this attachment. Matching dimensions, valid-prefix lengths, or zero suffixes cannot fill that provenance gap.

## Scope and stop point

No MP4 audio extraction, forced alignment, video frame decoding, PTS extraction, or slot-to-time mapping was performed. No HEAF output or prediction was used to select cases. The existing grounding API remains fail-closed for A/V; no approximate mappings were added.

The gate can be reopened if a producer record tied to these exact files supplies the preprocessing version and a row mapping from verified token/word positions to audio and vision features, including the word-to-WordPiece expansion rule. See [machine-readable audit](q3_av_grounding_audit.json) and [experiment evidence](../../experiments/q3/exp_0027_av_grounding/notes.md).
