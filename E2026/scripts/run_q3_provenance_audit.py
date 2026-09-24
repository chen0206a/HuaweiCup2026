"""Reproduce the Q3-2.5 token audit and write machine/human reports."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import huggingface_hub
import tokenizers as tokenizers_package
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.q3.data_adapter import UnlabeledAligned50Dataset  # noqa: E402
from src.q3.evidence_grounding import ground_interval  # noqa: E402
from src.q3.text_grounding import (  # noqa: E402
    TOKENIZER_JSON_SHA256,
    TOKENIZER_REVISION,
    TOKENIZER_REPO,
    ground_text_interval,
)


ROOT = PROJECT_ROOT
MANIFEST = ROOT / "data/manifests/q3/attachment4_inventory.json"
OUTPUT = ROOT / "outputs/q3"
EXPERIMENT = ROOT / "experiments/q3/exp_0025_provenance_recovery"
TOKENIZER_PATH = Path(hf_hub_download(
    repo_id=TOKENIZER_REPO,
    filename="tokenizer.json",
    revision=TOKENIZER_REVISION,
))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokenizer(*, lowercase=True, truncation="right", add_special_tokens=True,
               fixed_padding=True):
    spec = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    spec["normalizer"]["lowercase"] = lowercase
    tokenizer = Tokenizer.from_str(json.dumps(spec))
    if truncation is not None:
        tokenizer.enable_truncation(max_length=50, direction=truncation)
    if fixed_padding:
        tokenizer.enable_padding(length=50, pad_id=0, pad_type_id=0, pad_token="[PAD]")
    return tokenizer, add_special_tokens


def _compare(records, tokenizer, add_special_tokens=True):
    results = []
    for record in records:
        encoding = tokenizer.encode(record["raw_text"], add_special_tokens=add_special_tokens)
        expected_ids = np.asarray(record["text_bert"][0], dtype=np.int64)
        expected_mask = np.asarray(record["text_bert"][1], dtype=np.int64)
        expected_types = np.asarray(record["text_bert"][2], dtype=np.int64)
        got_ids = np.asarray(encoding.ids, dtype=np.int64)
        got_mask = np.asarray(encoding.attention_mask, dtype=np.int64)
        got_types = np.asarray(encoding.type_ids, dtype=np.int64)
        full_shape = got_ids.shape == expected_ids.shape
        ids_equal = full_shape and np.array_equal(got_ids, expected_ids)
        mask_equal = full_shape and np.array_equal(got_mask, expected_mask)
        types_equal = full_shape and np.array_equal(got_types, expected_types)
        exact = bool(ids_equal and mask_equal and types_equal)
        first = None
        if not exact:
            upto = min(len(expected_ids), len(got_ids))
            first = next((i for i in range(upto)
                          if expected_ids[i] != got_ids[i]
                          or expected_mask[i] != got_mask[i]
                          or expected_types[i] != got_types[i]), None)
            if first is None and len(expected_ids) != len(got_ids):
                first = upto
        if first is None:
            mismatch = None
        else:
            mismatch = {
                "slot_index": int(first),
                "provided_id": int(expected_ids[first]) if first < len(expected_ids) else None,
                "reconstructed_id": int(got_ids[first]) if first < len(got_ids) else None,
                "provided_token": tokenizer.id_to_token(int(expected_ids[first])) if first < len(expected_ids) else None,
                "reconstructed_token": tokenizer.id_to_token(int(got_ids[first])) if first < len(got_ids) else None,
            }
        results.append({
            "sample_id": record["id"],
            "exact_ids_mask_and_types": exact,
            "ids_exact": bool(ids_equal),
            "attention_mask_exact": bool(mask_equal),
            "token_type_ids_exact": bool(types_equal),
            "valid_length": record["valid_length"],
            "reconstructed_length": int(sum(got_mask)) if got_mask.ndim == 1 else None,
            "first_mismatch": mismatch,
            "raw_text": record["raw_text"] if mismatch else None,
            "special_tokens": {
                "provided_first_valid": int(expected_ids[0]),
                "provided_last_valid": int(expected_ids[record["valid_length"] - 1]),
                "provided_cls_sep": int(expected_ids[0]) == 101 and int(expected_ids[record["valid_length"] - 1]) == 102,
            },
        })
    return results


def _candidate(name, records, *, lowercase=True, truncation="right",
               add_special_tokens=True, fixed_padding=True):
    tokenizer, special = _tokenizer(
        lowercase=lowercase,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        fixed_padding=fixed_padding,
    )
    rows = _compare(records, tokenizer, add_special_tokens=special)
    return {
        "name": name,
        "settings": {"lowercase": lowercase, "add_special_tokens": add_special_tokens,
                     "max_length": 50 if truncation else None,
                     "truncation": truncation or "none",
                     "padding": "right_to_50" if fixed_padding else "none"},
        "exact_match_count": sum(row["exact_ids_mask_and_types"] for row in rows),
        "sample_count": len(rows),
        "mismatches": [row for row in rows if not row["exact_ids_mask_and_types"]],
    }


def run():
    if _sha256(TOKENIZER_PATH) != TOKENIZER_JSON_SHA256:
        raise RuntimeError("pinned bert-base-uncased tokenizer artifact changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = sorted(
        Path(item["absolute_path"])
        for item in manifest["files"]
        if Path(item["absolute_path"]).parent.name == "对齐版本"
        and Path(item["absolute_path"]).suffix.lower() == ".pkl"
    )
    dataset = UnlabeledAligned50Dataset(paths)
    records = dataset.records
    rows = [
        _candidate("bert-base-uncased-fast-right-truncate-50-pad-right", records),
        _candidate("bert-base-uncased-fast-left-truncate-50-pad-right", records, truncation="left"),
        _candidate("bert-base-uncased-fast-no-special-tokens", records, add_special_tokens=False),
        _candidate("bert-base-uncased-fast-case-sensitive", records, lowercase=False),
        _candidate("bert-base-uncased-fast-no-truncation-no-padding", records, truncation=None, fixed_padding=False),
    ]
    selected_tokenizer, _ = _tokenizer()
    all_sample_checks = _compare(records, selected_tokenizer)
    if not all(row["exact_ids_mask_and_types"] for row in all_sample_checks):
        raise RuntimeError("selected tokenizer configuration did not reproduce all 20 examples")

    smoke = []
    for record in (records[0], records[6]):
        length = record["valid_length"]
        interval = (1, min(5, length - 1))
        evidence = ground_interval(
            sample_id=record["id"], modality="text", start_index=interval[0],
            end_index=interval[1], valid_length=record["valid_length"],
            media_file=None, raw_text=record["raw_text"],
            text_bert=record["text_bert"], tokenizer=selected_tokenizer,
        )
        token_map = evidence["text_token_mapping"]
        smoke.append({
            "sample_id": record["id"], "slot_interval_half_open": list(interval),
            "token_mapping": token_map,
            "feature_evidence_result": evidence,
            "feature_row_mapping_status": "UNVERIFIED",
        })

    result = {
        "stage": "Q3-2.5 Alignment Provenance Recovery",
        "attachment4_access": "aligned_50 provenance/interface audit only; no labels or predictor outputs",
        "overall_status": "BLOCKED_BY_GROUNDING",
        "grounding_status_by_modality": {
            "text": "UNVERIFIED",
            "text_token_to_raw_span_submapping": "VERIFIED",
            "text_feature_row_to_token_slot": "UNVERIFIED",
            "audio": "UNVERIFIED",
            "vision": "UNVERIFIED",
        },
        "attachment_provenance": {
            "finding": "high_structural_similarity_to_MMSA_and_Self_MM_releases_but_not_confirmed",
            "evidence": ["20 sample-level pickle files, not the public split-level aggregate",
                         "generic IDs 01..20 rather than video_id$_$clip_id",
                         "no upstream file hash, commit, preprocessing manifest, or source IDs in samples",
                         "public code/configs explain candidate formats but do not bind these file hashes to a release"],
        },
        "tokenizer": {
            "repo": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizers_version": tokenizers_package.__version__,
            "huggingface_hub_version": huggingface_hub.__version__,
            "settings": {"fast_tokenizer": True, "lowercase": True,
                         "add_special_tokens": True, "max_length": 50,
                         "truncation": "right", "padding": "right_to_50_with_id_0"},
        },
        "tokenizer_candidates": rows,
        "selected_configuration": {
            "exact_id_mask_type_match_count": sum(row["exact_ids_mask_and_types"] for row in all_sample_checks),
            "sample_count": len(records),
            "first_mismatch_index": None,
            "all_cls_sep_correct": all(row["special_tokens"]["provided_cls_sep"] for row in all_sample_checks),
            "all_ids_mask_types_exact_including_padding": True,
            "max_valid_length": max(row["valid_length"] for row in all_sample_checks),
            "truncated_samples_at_50": [row["sample_id"] for row in all_sample_checks if row["valid_length"] == 50],
            "per_sample_token_match": all_sample_checks,
            "punctuation_apostrophe_and_wordpiece_behavior": "Included in the exact per-sample ID equality; no word/space splitting was used.",
        },
        "text_feature_row_audit": {
            "attachment_feature_shape": [50, 768],
            "text_bert_shape": [3, 50],
            "tokenizer_ids_match": True,
            "source_proves_attachment_text_row_index_equals_text_bert_slot": False,
            "reason": "The public candidate pipelines are not tied to Attachment4 hashes; the local sample files carry no feature-extraction provenance. Exact text_bert IDs alone do not identify the generation of the 768-D text rows.",
        },
        "audio_vision_audit": {
            "attachment_dims": {"audio": 74, "vision": 35},
            "likely_public_feature_families": {"audio": "COVAREP 74-D", "vision": "FACET 35-D"},
            "source_proves_attachment_rows_share_text_slot_alignment": False,
            "timestamped_intervals_present_in_pkl": False,
            "original_source_ids_present": False,
        },
        "timestamps_recoverable_from_public_provenance": {
            "in_principle": "yes when original video_id$_$clip_id and exact source/version are known; CMU SDK CSD computational sequences preserve feature intervals",
            "for_these_samples": False,
            "reason": "Only generic IDs 01..20 are present and no source-key mapping or feature-generation hash is supplied; no large external corpus was downloaded.",
        },
        "smoke_cases": smoke,
        "public_sources": {
            "MMSA": {"repo": "https://github.com/thuiar/MMSA", "commit": "a94e65d07fa1ae0d44e552390074b29b0898edfd"},
            "Self-MM": {"repo": "https://github.com/thuiar/Self-MM", "commit": "1786283c81eeb507f317fa1c70a3faf77e67cee0", "branch": "main", "legacy_master_commit": "e694b8c88efe9ce6f2b996ac7793c886c5df10e4"},
            "MMSA-FET": {"repo": "https://github.com/thuiar/MMSA-FET", "commit": "f8fbd2d88d4f77580ea1ded0b3469073488c5c19"},
            "CMU-MultimodalSDK": {"repo": "https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK", "commit": "4f2eadcd7e7b9e20e83b868cdad385b41830285c"},
        },
        "history_review": {
            "MMSA_checksum_history_commit": "bf88dfc8073193a7566dadeae7d6de82f1e86a56",
            "MMSA_readme_update_commit": "05c9d1407300fce7977a9d140058c4ed0db9dcdd",
            "Self_MM_preprocessing_update_commit": "e3a4a5378b14192d91e361fa7ae2c327a2664144",
            "MMSA_FET_aligner_rewrite_commit": "1548bda6e15ae08a2c16a1d284605f9c4f1b13fc",
            "CMU_MultimodalSDK_history_commit": "3e27d3781fe38b660432c7edfa7912bd0d256ee1",
        },
        "limitations": ["No Attachment4 labels used.", "No Attachment2 test or Attachment3 content accessed.",
                        "No training or HEAF/P2 parameter changes.", "No final prediction CSV or batch explanation cards produced."],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    (OUTPUT / "q3_alignment_provenance.json").write_text(encoded, encoding="utf-8")
    (EXPERIMENT / "metrics.json").write_text(encoded, encoding="utf-8")
    (OUTPUT / "q3_alignment_provenance_report.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"status": result["overall_status"],
                      "token_match": result["selected_configuration"]["exact_id_mask_type_match_count"],
                      "sample_count": len(records)}, ensure_ascii=False))
    return result


def render_report(result):
    exact = result["selected_configuration"]["exact_id_mask_type_match_count"]
    sample_count = result["selected_configuration"]["sample_count"]
    candidates = result["tokenizer_candidates"]
    lines = [
        "# Q3-2.5 Alignment Provenance Recovery", "",
        f"**Overall status: `{result['overall_status']}`.** Per-modality result: TEXT `{result['grounding_status_by_modality']['text']}` (BERT token-to-raw-span submapping `{result['grounding_status_by_modality']['text_token_to_raw_span_submapping']}`; text feature row link `{result['grounding_status_by_modality']['text_feature_row_to_token_slot']}`), AUDIO `{result['grounding_status_by_modality']['audio']}`, VISION `{result['grounding_status_by_modality']['vision']}`.", "",
        "The distinction matters: P2 consumes the 768-D `text` rows. Exact reconstruction of `text_bert` proves token identity and raw character offsets, but Attachment4 does not identify the generation or row ordering of those 768-D vectors. The text evidence field for P2 explanations therefore remains withheld.", "",
        "## Attachment4 provenance", "",
        "The sample dictionaries strongly resemble public MMSA/Self-MM MOSEI aligned data (fields, dimensions, fixed 50 positions, BERT ID/mask/type rows). This establishes a **candidate format**, not source provenance. Public releases are split-level pickles with their own whole-file checksums and public IDs in `video_id$_$clip_id` form; Attachment4 instead contains 20 single-sample pickles with IDs `01`–`20`. No source checkpoint, preprocessing manifest, original source ID, or checksum link binds these files to a specific public release.", "",
        "## Text token replay", "",
        f"Pinned `bert-base-uncased` fast tokenizer reproduced input IDs, attention masks, and token type IDs exactly for **{exact}/{sample_count}** samples, including the padded arrays. Settings: lowercasing on, `[CLS]`/`[SEP]` enabled, max length 50, truncate on the right, right-pad with ID 0. All observed valid prefixes start with ID 101 `[CLS]` and end with ID 102 `[SEP]`; IDs in mask-zero padding match as well. Punctuation, apostrophe and WordPiece cases are included in these full-sequence comparisons.",
        f"Tokenizer artifact: revision `{result['tokenizer']['revision']}`; SHA256 `{result['tokenizer']['tokenizer_json_sha256']}`. The selected configuration has no mismatch. Left truncation matches {candidates[1]['exact_match_count']}/{sample_count}; no special tokens matches {candidates[2]['exact_match_count']}/{sample_count}; case-sensitive tokenization matches {candidates[3]['exact_match_count']}/{sample_count}; no truncation/padding matches {candidates[4]['exact_match_count']}/{sample_count}.",
        "The fast tokenizer offsets support exact `text_bert` slot → token → raw-text character span. `[CLS]`, `[SEP]`, and padding slots are excluded from text evidence. Contextual BERT vectors are not described as independent word semantics.", "",
        "## Feature row and nonverbal alignment", "",
        "**`TOKEN_MAPPING_VERIFIED`; `FEATURE_ROW_MAPPING_UNVERIFIED`.** Public MMSA code distinguishes precomputed BERT features from `text_bert` IDs and derives aligned audio/vision valid lengths from the text mask, but its loader consumes already-created pickle files. Self-MM's preprocessing source generates BERT hidden states and token IDs separately from different calls and pads sequences by a dataset-level length rule; it is not linked to these Attachment4 hashes. MMSA-FET demonstrates a separate compatible design that extracts BERT hidden states, token IDs and maps word-aligned A/V features to tokenizer `word_ids`; it preserves the alignment records. Those projects explain plausible semantics, but do not prove that this competition pickle uses their exact pipeline.", "",
        "Audio 74 / vision 35 are consistent with the common COVAREP 74-D / FACET 35-D feature family used for MOSEI. MMSA's config also lists `(768,74,35)` for MOSEI. This dimensional agreement alone does not prove attachment feature extractor identity or row alignment. The pkl contains no interval/frame metadata, and audio/vision zero suffixes only identify padding, not time.", "",
        "## Time recovery", "",
        "CMU-MultimodalSDK stores computational sequence intervals alongside features and describes word-level alignment by choosing a reference sequence and collapsing other features (often with a mean) within reference intervals. Its CMU-MOSEI registry includes timestamped word vectors, COVAREP and FACET/OpenFace sequences. Such metadata could restore times if the attachment sample retained its original `video_id$_$clip_id` and the exact source version. Attachment4 has only generic IDs `01`–`20`, so this key join cannot be made. No large external dataset was downloaded and no labels were used.", "",
        "MMSA-FET provides a candidate reconstruction route: its Wav2vec CTC aligner emits word intervals, then its alignment function selects feature timestamps inside each word interval, averages audio/video descriptors, and expands pooled values to BERT tokens via word IDs. It retains the align records in output. It is a different extraction pipeline and cannot be applied to Attachment4 as verified provenance without evidence that its feature generation matches.", "",
        "## Grounding decision", "",
        "1. **Attachment4 structure vs MMSA/Self-MM:** high structural similarity; provenance is not confirmed.",
        "2. **BERT tokenizer replay:** `bert-base-uncased` reproduces 20/20 IDs, masks and token types exactly.",
        "3. **Slot → raw_text span:** verified for `text_bert` token slots. The P2 `text` feature-row link remains unverified.",
        "4. **Text feature row semantics:** not proven for these exact files.",
        "5. **Audio/vision alignment and timestamps:** public methods are known, but Attachment4 provenance and original IDs are missing; per-row mapping is unverified.",
        "6. **Public timestamps for these exact samples:** not currently recoverable through a keyed provenance join.", "",
        "Final status: `TEXT = UNVERIFIED` for P2 evidence attribution (token submapping verified); `AUDIO = UNVERIFIED`; `VISION = UNVERIFIED`; overall `BLOCKED_BY_GROUNDING`. The feature-grounding blocker remains because `text` feature slots cannot yet be tied to the verified token slots.", "",
        "## Reviewed public sources", "",
        f"- [MMSA README at `{result['public_sources']['MMSA']['commit']}`](https://github.com/thuiar/MMSA/blob/{result['public_sources']['MMSA']['commit']}/README.md): release checksums, schema and feature-field descriptions.",
        f"- [MMSA data loader](https://github.com/thuiar/MMSA/blob/{result['public_sources']['MMSA']['commit']}/src/MMSA/data_loader.py): precomputed arrays loaded; BERT feature selection and aligned lengths derived from mask.",
        f"- [Self-MM README](https://github.com/thuiar/Self-MM/blob/{result['public_sources']['Self-MM']['commit']}/README.md), [DataPre.py](https://github.com/thuiar/Self-MM/blob/{result['public_sources']['Self-MM']['commit']}/data/DataPre.py), [MOSEI config](https://github.com/thuiar/Self-MM/blob/{result['public_sources']['Self-MM']['commit']}/config/config_regression.py): release checksums, BERT/token generation and sequence padding; main commit `{result['public_sources']['Self-MM']['commit']}`.",
        f"- [MMSA-FET dataset alignment](https://github.com/thuiar/MMSA-FET/blob/{result['public_sources']['MMSA-FET']['commit']}/src/MSA_FET/dataset.py), [BERT extractor](https://github.com/thuiar/MMSA-FET/blob/{result['public_sources']['MMSA-FET']['commit']}/src/MSA_FET/extractors/text/bert.py), [Wav2vec CTC aligner](https://github.com/thuiar/MMSA-FET/blob/{result['public_sources']['MMSA-FET']['commit']}/src/MSA_FET/aligner/default.py), example config uses `bert-base-uncased`; commit `{result['public_sources']['MMSA-FET']['commit']}`.",
        f"- [CMU-MultimodalSDK README](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/{result['public_sources']['CMU-MultimodalSDK']['commit']}/README.md), [alignment implementation](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/{result['public_sources']['CMU-MultimodalSDK']['commit']}/mmsdk/mmdatasdk/dataset/dataset.py), [MOSEI sequence registry](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/{result['public_sources']['CMU-MultimodalSDK']['commit']}/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py); commit `{result['public_sources']['CMU-MultimodalSDK']['commit']}`.",
        "- [MulT paper (ACL Anthology)](https://aclanthology.org/P19-1656/): public description of word-level alignment by P2FA and mean pooling of audio/video within word intervals; explains the common COVAREP/FACET family, not Attachment4 provenance.", "",
        "## Smoke cases and scope", "",
        "Two samples were passed through the integrated evidence-grounding API. It returns the verified token mapping as a diagnostic while retaining outer `text_fragment=null` because the feature-row link is unresolved. No HEAF or P2 inference was run; no training, tuning, labels, Attachment2 test, Attachment3, or final explanation cards were used.", "",
        "Candidate next steps: recover an official attachment-to-source-ID and feature-hash manifest, or separately review a reconstruction proposal that includes exact feature-preprocessing compatibility and independent alignment checks. Neither candidate was implemented in this audit.", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
