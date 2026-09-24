"""Attachment4 aligned interface and grounding audit; no predictor inference."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.q3.data_adapter import UnlabeledAligned50Dataset
from src.q3.evidence_grounding import ground_interval


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/q3/attachment4_inventory.json"
LOCK = ROOT / "outputs/q3/q3_method_lock.json"
OUTPUT = ROOT / "outputs/q3"
EXPERIMENT = ROOT / "experiments/q3/exp_002_grounding_audit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _aligned_entries(manifest: dict) -> tuple[list[dict], list[dict]]:
    pkl = []
    videos = []
    for item in manifest["files"]:
        path = Path(item["absolute_path"])
        if path.parent.name == "对齐版本" and path.suffix.lower() == ".pkl":
            pkl.append(item)
        elif path.parent.name == "videos" and path.parent.parent.name == "对齐版本" and path.suffix.lower() == ".mp4":
            videos.append(item)
    return sorted(pkl, key=lambda x: x["absolute_path"]), sorted(videos, key=lambda x: x["absolute_path"])


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"invalid video FPS/frame count: {path}")
        first_ok = capture.grab()
        if not first_ok:
            raise RuntimeError(f"cannot decode first video frame: {path}")
        first_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        last_time = first_time
        decoded_count = 1
        # Seeking to CAP_PROP_FRAME_COUNT-1 fails on some supplied clips. Walk
        # decoded frame positions instead; do not inspect frame pixels.
        while capture.grab():
            decoded_count += 1
            last_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        return {"opened": True, "backend": capture.getBackendName(),
                "fps_reported": fps, "frame_count_reported": frames,
                "nominal_duration_seconds_frame_count_over_fps": frames / fps,
                "decoded_frame_count": decoded_count,
                "decoded_count_differs_from_reported": decoded_count != frames,
                "first_decoded_frame_pos_msec_seconds": first_time,
                "last_decoded_frame_pos_msec_seconds": last_time,
                "width": width, "height": height,
                "timing_caveat": "OpenCV decoder-reported frame positions and FPS/count estimates; not a feature-slot mapping or independently certified container PTS."}
    finally:
        capture.release()


def render_report(result: dict) -> str:
    matching = result["id_matching"]
    features = result["feature_interface"]
    videos = result["video_metadata"]
    lines = ["# Q3-2 Attachment4 grounding and interface audit", "",
             f"**Conclusion: `{result['conclusion']}`; grounding level: `{result['grounding_status']}`.**",
             "", "HEAF was locked before Attachment4 content inspection. This audit did not load the predictor or generate final explanations/CSV.", "",
             "## Aligned files and ID correspondence", "",
             f"Aligned set: {matching['feature_file_count']} pkl + {matching['video_file_count']} mp4; exact feature-ID→video-stem matches {matching['exact_match_count']}.",
             f"Missing feature/video: {len(matching['missing_feature_ids'])}/{len(matching['missing_video_ids'])}; duplicate feature IDs {matching['duplicate_feature_id_count']}; ambiguous mappings {matching['ambiguous_mapping_count']}; filename/embedded-ID mismatches {matching['feature_filename_id_mismatch_count']}.",
             "All aligned files matched the pre-existing size/SHA256 inventory: " + str(result["inventory_integrity"]["all_match"]) + ".", "",
             "## Feature interface", "",
             f"Every sample has keys {features['common_keys']}; required shapes text/audio/vision/text_bert = [50,768]/[50,74]/[50,35]/[3,50].",
             f"Numeric finite and binary valid-prefix mask checks: {features['all_valid']}; valid length range {features['valid_length_min']}–{features['valid_length_max']}. Padding comes only from text_bert[1]; native zero vectors are kept as data.",
             f"raw_text present in {features['raw_text_present_count']}/{features['sample_count']} samples; no label fields. Actual timestamp/alignment metadata keys: {features['timestamp_or_alignment_keys']}.",
             f"text_bert permits exact recovery of integer token IDs and mask; all samples begin/end with observed IDs {features['first_valid_token_ids']}/{features['last_valid_token_ids']}. The supplied files have no tokenizer vocabulary, token offsets, word timestamps or feature-slot→media table. Slot counts differ from whitespace word counts in {features['slot_count_vs_whitespace_word_count_mismatch']} samples, which rules out a direct one-word-per-slot assumption.", "",
             "## What the videos provide", "",
             f"All {videos['opened_count']} paired MP4s opened in OpenCV. Nominal duration range from reported frame_count/FPS: {videos['nominal_duration_min_seconds']:.3f}–{videos['nominal_duration_max_seconds']:.3f} seconds. Sequential decoding gives {videos['decoded_frame_count_min']}–{videos['decoded_frame_count_max']} frames and last-frame positions {videos['last_decoded_time_min_seconds']:.3f}–{videos['last_decoded_time_max_seconds']:.3f} seconds; {videos['reported_vs_decoded_frame_count_mismatch_count']} files have a reported/decoded frame-count mismatch.",
             "This environment has no ffprobe executable or raw PTS API in OpenCV. Decoder-reported frame positions are not an alignment from feature slots to video/audio seconds.", "",
             "## Grounding decision", "",
             "The pkl gives whole-sample raw_text and BERT token IDs, but no per-slot text span. Audio/vision arrays give no timestamp or source frame index. MP4 timing cannot resolve which feature slot came from which time. We therefore return `unverified` for interval grounding: text_fragment, audio_time_range and video_frame_time stay null. Exact feature ID→MP4 identity is verified separately.",
             "No `duration/50`, character-proportional text slicing or Q1 uniform-bin rule was used.", "",
             "## At most two candidate ways forward (not implemented)", "",
             "1. **Recover authoritative preprocessing provenance.** Obtain the source aligned-feature generation record with per-slot token offsets, word/audio times and visual frame/PTS indices, linked by exact sample ID and compatible with these file hashes. Validate every mapping against raw_text and MP4. If complete, interval grounding could be VERIFIED; missing offsets or undocumented resampling remain errors.",
             "2. **Reconstruct and validate approximate alignment from provided media.** Fix the exact BERT tokenizer/version, align raw_text to the paired audio and video PTS using a documented forced-alignment/decoding pipeline, then compare reconstructed slot indices with feature preprocessing. Word alignment, subword splits, truncation, ASR mismatch, video edits and decoder timing create uncertainty. Report sample-specific error bounds and use APPROXIMATE only after independent spot checks; otherwise remain UNVERIFIED.", "",
             "The second route is a proposal, not an automatically accepted replacement for absent official metadata. No HEAF parameter may be retuned from Attachment4.", ""]
    return "\n".join(lines)


def run() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "METHOD_LOCKED_BEFORE_ATTACHMENT4_CONTENT_AUDIT":
        raise RuntimeError("HEAF method is not locked")
    cfg = yaml.safe_load((ROOT / "configs/final/q3_heaf.yaml").read_text(encoding="utf-8"))
    if cfg["predictor"]["checkpoint_sha256"] != lock["predictor"]["sha256"] or cfg["temporal"]["rho"] != lock["temporal"]["rho"]:
        raise RuntimeError("YAML/JSON method locks disagree")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    feature_entries, video_entries = _aligned_entries(manifest)
    if not feature_entries or not video_entries:
        raise RuntimeError("no aligned Attachment4 files in inventory")
    inventory_checks = []
    for item in (*feature_entries, *video_entries):
        path = Path(item["absolute_path"])
        inventory_checks.append({"name": path.name, "kind": path.suffix.lower(),
                                 "size_match": path.stat().st_size == item["size_bytes"],
                                 "sha256_match": sha256_file(path) == item["sha256"]})
    if not all(row["size_match"] and row["sha256_match"] for row in inventory_checks):
        raise RuntimeError("aligned Attachment4 file differs from inventory")

    dataset = UnlabeledAligned50Dataset([Path(item["absolute_path"]) for item in feature_entries])
    ids = [record["id"] for record in dataset.records]
    feature_counts = Counter(ids)
    video_counts = Counter(Path(item["absolute_path"]).stem for item in video_entries)
    missing_feature = sorted(set(video_counts) - set(feature_counts))
    missing_video = sorted(set(feature_counts) - set(video_counts))
    ambiguous = sorted(key for key in set(feature_counts) | set(video_counts)
                       if feature_counts[key] > 1 or video_counts[key] > 1)
    filename_mismatch = [record["id"] for record in dataset.records
                         if Path(record["source"]).stem != record["id"]]
    videos_by_id = {Path(item["absolute_path"]).stem: Path(item["absolute_path"])
                    for item in video_entries}
    video_rows = {sample_id: inspect_video(videos_by_id[sample_id])
                  for sample_id in sorted(videos_by_id)}
    fields = [set(record["keys"]) for record in dataset.records]
    all_fields = sorted(set.union(*fields))
    metadata_keywords = ("time", "stamp", "interval", "align", "offset", "segment", "frame", "length", "word")
    time_keys = [key for key in all_fields if any(word in key.lower() for word in metadata_keywords)]
    token_first = sorted({int(record["text_bert"][0, 0]) for record in dataset.records})
    token_last = sorted({int(record["text_bert"][0, record["valid_length"] - 1]) for record in dataset.records})
    sample_rows = []
    for record in dataset.records:
        sample_id = record["id"]
        text = record["raw_text"]
        row = {"sample_id": sample_id, "pkl_file": Path(record["source"]).name,
               "video_file": videos_by_id[sample_id].name if sample_id in videos_by_id else None,
               "pkl_keys": record["keys"],
               "timestamp_or_alignment_keys": [key for key in record["keys"] if key in time_keys],
               "feature_shapes": {m: list(record["features"][m].shape) for m in ("text", "audio", "vision")},
               "feature_dtypes": {m: str(record["features"][m].dtype) for m in ("text", "audio", "vision")},
               "text_bert_shape": list(record["text_bert"].shape),
               "valid_length": record["valid_length"], "raw_text_present": bool(text),
               "raw_text_characters": len(text), "raw_text_whitespace_words": len(text.split()),
               "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "first_valid_token_id": int(record["text_bert"][0, 0]),
               "last_valid_token_id": int(record["text_bert"][0, record["valid_length"] - 1]),
               "native_zero_valid_positions": {m: int(record["native_zero"][m].sum()) for m in ("text", "audio", "vision")},
               "padding_suffix_nonzero_values": record["suffix_nonzero"],
               "video_metadata": video_rows.get(sample_id)}
        sample_rows.append(row)
    smoke = []
    for record in dataset.records[:2]:
        for modality in ("text", "audio", "vision"):
            smoke.append({"sample_id": record["id"], "modality": modality,
                          "audit_interval": [0, min(5, record["valid_length"])],
                          "grounding": ground_interval(
                              sample_id=record["id"], modality=modality,
                              start_index=0, end_index=min(5, record["valid_length"]),
                              valid_length=record["valid_length"],
                              media_file=videos_by_id.get(record["id"]))})
    result = {
        "conclusion": "BLOCKED_BY_GROUNDING", "grounding_status": "UNVERIFIED",
        "method_lock": {"json": "outputs/q3/q3_method_lock.json", "source_commit": lock["source_commit"],
                        "checkpoint_sha256": lock["predictor"]["sha256"], "rho": lock["temporal"]["rho"]},
        "scope": {"attachment": 4, "feature_version": "aligned_50", "prediction_run": False,
                  "unaligned_content_read": False, "labels_used": False},
        "inventory_integrity": {"all_match": True, "files_checked": len(inventory_checks), "file_checks": inventory_checks},
        "id_matching": {"feature_file_count": len(feature_entries), "video_file_count": len(video_entries),
                        "unique_feature_ids": len(feature_counts), "unique_video_stems": len(video_counts),
                        "exact_match_count": len(set(feature_counts) & set(video_counts)),
                        "missing_feature_ids": missing_feature, "missing_video_ids": missing_video,
                        "duplicate_feature_id_count": sum(count - 1 for count in feature_counts.values() if count > 1),
                        "duplicate_video_stem_count": sum(count - 1 for count in video_counts.values() if count > 1),
                        "ambiguous_mapping_count": len(ambiguous), "ambiguous_ids": ambiguous,
                        "feature_filename_id_mismatch_count": len(filename_mismatch),
                        "feature_filename_id_mismatches": filename_mismatch},
        "feature_interface": {"sample_count": len(dataset), "all_valid": True,
                              "common_keys": sorted(set.intersection(*fields)), "all_keys": all_fields,
                              "timestamp_or_alignment_keys": time_keys,
                              "label_keys": [key for key in all_fields if "label" in key.lower() or "annotation" in key.lower()],
                              "raw_text_present_count": sum(bool(record["raw_text"]) for record in dataset.records),
                              "valid_length_min": min(record["valid_length"] for record in dataset.records),
                              "valid_length_max": max(record["valid_length"] for record in dataset.records),
                              "first_valid_token_ids": token_first, "last_valid_token_ids": token_last,
                              "slot_count_vs_whitespace_word_count_mismatch": sum(
                                  record["valid_length"] - 2 != len(record["raw_text"].split()) for record in dataset.records),
                              "token_id_recoverable": True, "token_offsets_or_vocabulary_supplied": False,
                              "padding_source": "text_bert[1] binary true prefix, not feature zeros"},
        "video_metadata": {"opened_count": len(video_rows),
                           "nominal_duration_min_seconds": min(row["nominal_duration_seconds_frame_count_over_fps"] for row in video_rows.values()),
                           "nominal_duration_max_seconds": max(row["nominal_duration_seconds_frame_count_over_fps"] for row in video_rows.values()),
                           "decoded_frame_count_min": min(row["decoded_frame_count"] for row in video_rows.values()),
                           "decoded_frame_count_max": max(row["decoded_frame_count"] for row in video_rows.values()),
                           "last_decoded_time_min_seconds": min(row["last_decoded_frame_pos_msec_seconds"] for row in video_rows.values()),
                           "last_decoded_time_max_seconds": max(row["last_decoded_frame_pos_msec_seconds"] for row in video_rows.values()),
                           "reported_vs_decoded_frame_count_mismatch_count": sum(row["decoded_count_differs_from_reported"] for row in video_rows.values()),
                           "raw_pts_api_available": False, "probe_tool": "OpenCV VideoCapture 4.10.0"},
        "grounding_gap": {"exact_id_to_video": not (missing_feature or missing_video or ambiguous or filename_mismatch),
                          "slot_to_text_span": False, "slot_to_audio_seconds": False,
                          "slot_to_video_frame_or_seconds": False,
                          "raw_text_whole_sample_only": True,
                          "unverified_fields_must_be_null": ["text_fragment", "audio_time_range", "video_frame_time"]},
        "samples": sample_rows,
        "smoke_cases": smoke,
        "candidate_count": 2,
        "prohibited_shortcuts_used": []
    }
    write_json(OUTPUT / "q3_grounding_audit.json", result)
    write_json(EXPERIMENT / "metrics.json", result)
    (OUTPUT / "q3_grounding_audit.md").write_text(render_report(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({"conclusion": outcome["conclusion"],
                      "grounding_status": outcome["grounding_status"],
                      "id_matching": outcome["id_matching"]}, ensure_ascii=False), flush=True)
