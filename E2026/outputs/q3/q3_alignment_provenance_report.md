# Q3-2.5 Alignment Provenance Recovery

**Overall status: `BLOCKED_BY_GROUNDING`.** Per-modality result: TEXT `UNVERIFIED` (BERT token-to-raw-span submapping `VERIFIED`; text feature row link `UNVERIFIED`), AUDIO `UNVERIFIED`, VISION `UNVERIFIED`.

The distinction matters: P2 consumes the 768-D `text` rows. Exact reconstruction of `text_bert` proves token identity and raw character offsets, but Attachment4 does not identify the generation or row ordering of those 768-D vectors. The text evidence field for P2 explanations therefore remains withheld.

## Attachment4 provenance

The sample dictionaries strongly resemble public MMSA/Self-MM MOSEI aligned data (fields, dimensions, fixed 50 positions, BERT ID/mask/type rows). This establishes a **candidate format**, not source provenance. Public releases are split-level pickles with their own whole-file checksums and public IDs in `video_id$_$clip_id` form; Attachment4 instead contains 20 single-sample pickles with IDs `01`–`20`. No source checkpoint, preprocessing manifest, original source ID, or checksum link binds these files to a specific public release.

## Text token replay

Pinned `bert-base-uncased` fast tokenizer reproduced input IDs, attention masks, and token type IDs exactly for **20/20** samples, including the padded arrays. Settings: lowercasing on, `[CLS]`/`[SEP]` enabled, max length 50, truncate on the right, right-pad with ID 0. All observed valid prefixes start with ID 101 `[CLS]` and end with ID 102 `[SEP]`; IDs in mask-zero padding match as well. Punctuation, apostrophe and WordPiece cases are included in these full-sequence comparisons.
Tokenizer artifact: revision `86b5e0934494bd15c9632b12f734a8a67f723594`; SHA256 `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98`. The selected configuration has no mismatch. Left truncation matches 18/20; no special tokens matches 0/20; case-sensitive tokenization matches 0/20; no truncation/padding matches 0/20.
The fast tokenizer offsets support exact `text_bert` slot → token → raw-text character span. `[CLS]`, `[SEP]`, and padding slots are excluded from text evidence. Contextual BERT vectors are not described as independent word semantics.

## Feature row and nonverbal alignment

**`TOKEN_MAPPING_VERIFIED`; `FEATURE_ROW_MAPPING_UNVERIFIED`.** Public MMSA code distinguishes precomputed BERT features from `text_bert` IDs and derives aligned audio/vision valid lengths from the text mask, but its loader consumes already-created pickle files. Self-MM's preprocessing source generates BERT hidden states and token IDs separately from different calls and pads sequences by a dataset-level length rule; it is not linked to these Attachment4 hashes. MMSA-FET demonstrates a separate compatible design that extracts BERT hidden states, token IDs and maps word-aligned A/V features to tokenizer `word_ids`; it preserves the alignment records. Those projects explain plausible semantics, but do not prove that this competition pickle uses their exact pipeline.

Audio 74 / vision 35 are consistent with the common COVAREP 74-D / FACET 35-D feature family used for MOSEI. MMSA's config also lists `(768,74,35)` for MOSEI. This dimensional agreement alone does not prove attachment feature extractor identity or row alignment. The pkl contains no interval/frame metadata, and audio/vision zero suffixes only identify padding, not time.

## Time recovery

CMU-MultimodalSDK stores computational sequence intervals alongside features and describes word-level alignment by choosing a reference sequence and collapsing other features (often with a mean) within reference intervals. Its CMU-MOSEI registry includes timestamped word vectors, COVAREP and FACET/OpenFace sequences. Such metadata could restore times if the attachment sample retained its original `video_id$_$clip_id` and the exact source version. Attachment4 has only generic IDs `01`–`20`, so this key join cannot be made. No large external dataset was downloaded and no labels were used.

MMSA-FET provides a candidate reconstruction route: its Wav2vec CTC aligner emits word intervals, then its alignment function selects feature timestamps inside each word interval, averages audio/video descriptors, and expands pooled values to BERT tokens via word IDs. It retains the align records in output. It is a different extraction pipeline and cannot be applied to Attachment4 as verified provenance without evidence that its feature generation matches.

## Grounding decision

1. **Attachment4 structure vs MMSA/Self-MM:** high structural similarity; provenance is not confirmed.
2. **BERT tokenizer replay:** `bert-base-uncased` reproduces 20/20 IDs, masks and token types exactly.
3. **Slot → raw_text span:** verified for `text_bert` token slots. The P2 `text` feature-row link remains unverified.
4. **Text feature row semantics:** not proven for these exact files.
5. **Audio/vision alignment and timestamps:** public methods are known, but Attachment4 provenance and original IDs are missing; per-row mapping is unverified.
6. **Public timestamps for these exact samples:** not currently recoverable through a keyed provenance join.

Final status: `TEXT = UNVERIFIED` for P2 evidence attribution (token submapping verified); `AUDIO = UNVERIFIED`; `VISION = UNVERIFIED`; overall `BLOCKED_BY_GROUNDING`. The feature-grounding blocker remains because `text` feature slots cannot yet be tied to the verified token slots.

## Reviewed public sources

- [MMSA README at `a94e65d07fa1ae0d44e552390074b29b0898edfd`](https://github.com/thuiar/MMSA/blob/a94e65d07fa1ae0d44e552390074b29b0898edfd/README.md): release checksums, schema and feature-field descriptions.
- [MMSA data loader](https://github.com/thuiar/MMSA/blob/a94e65d07fa1ae0d44e552390074b29b0898edfd/src/MMSA/data_loader.py): precomputed arrays loaded; BERT feature selection and aligned lengths derived from mask.
- [Self-MM README](https://github.com/thuiar/Self-MM/blob/1786283c81eeb507f317fa1c70a3faf77e67cee0/README.md), [DataPre.py](https://github.com/thuiar/Self-MM/blob/1786283c81eeb507f317fa1c70a3faf77e67cee0/data/DataPre.py), [MOSEI config](https://github.com/thuiar/Self-MM/blob/1786283c81eeb507f317fa1c70a3faf77e67cee0/config/config_regression.py): release checksums, BERT/token generation and sequence padding; main commit `1786283c81eeb507f317fa1c70a3faf77e67cee0`.
- [MMSA-FET dataset alignment](https://github.com/thuiar/MMSA-FET/blob/f8fbd2d88d4f77580ea1ded0b3469073488c5c19/src/MSA_FET/dataset.py), [BERT extractor](https://github.com/thuiar/MMSA-FET/blob/f8fbd2d88d4f77580ea1ded0b3469073488c5c19/src/MSA_FET/extractors/text/bert.py), [Wav2vec CTC aligner](https://github.com/thuiar/MMSA-FET/blob/f8fbd2d88d4f77580ea1ded0b3469073488c5c19/src/MSA_FET/aligner/default.py), example config uses `bert-base-uncased`; commit `f8fbd2d88d4f77580ea1ded0b3469073488c5c19`.
- [CMU-MultimodalSDK README](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/README.md), [alignment implementation](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/mmsdk/mmdatasdk/dataset/dataset.py), [MOSEI sequence registry](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/4f2eadcd7e7b9e20e83b868cdad385b41830285c/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py); commit `4f2eadcd7e7b9e20e83b868cdad385b41830285c`.
- [MulT paper (ACL Anthology)](https://aclanthology.org/P19-1656/): public description of word-level alignment by P2FA and mean pooling of audio/video within word intervals; explains the common COVAREP/FACET family, not Attachment4 provenance.

## Smoke cases and scope

Two samples were passed through the integrated evidence-grounding API. It returns the verified token mapping as a diagnostic while retaining outer `text_fragment=null` because the feature-row link is unresolved. No HEAF or P2 inference was run; no training, tuning, labels, Attachment2 test, Attachment3, or final explanation cards were used.

Candidate next steps: recover an official attachment-to-source-ID and feature-hash manifest, or separately review a reconstruction proposal that includes exact feature-preprocessing compatibility and independent alignment checks. Neither candidate was implemented in this audit.
