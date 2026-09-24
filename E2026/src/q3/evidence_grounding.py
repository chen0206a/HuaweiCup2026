"""Evidence grounding with explicit provenance; never infer seconds from slots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.dataset import MODALITIES


def _empty(status: str, media_file: str | Path | None, note: str) -> dict:
    return {"grounding_status": status, "mapping_method": None,
            "media_file": str(media_file) if media_file is not None else None,
            "text_fragment": None, "audio_time_range": None,
            "video_frame_time": None, "mapping_note": note}


def ground_interval(*, sample_id: str, modality: str, start_index: int,
                    end_index: int, valid_length: int,
                    media_file: str | Path | None,
                    mapping: dict[str, Any] | None = None,
                    raw_text: str | None = None,
                    text_bert: Any | None = None,
                    tokenizer: Any | None = None) -> dict:
    """Ground an exact half-open slot interval only with an audited mapping record.

    ``mapping`` is an independent, reviewed record for this *exact* sample,
    modality and interval. It is never created from media duration, character
    proportions, Q1 bins, model attention, or a feature vector's zero pattern.
    """
    if modality not in MODALITIES or not sample_id or not 0 <= start_index < end_index <= valid_length <= 50:
        raise ValueError("invalid sample/modality/valid-prefix interval")
    if mapping is None:
        result = _empty("unverified", media_file,
                        "No reviewed feature-row-to-raw-evidence mapping for this interval; raw evidence withheld.")
        if modality == "text" and raw_text is not None and text_bert is not None:
            from src.q3.text_grounding import ground_text_interval

            token_map = ground_text_interval(
                raw_text, text_bert, start_index, end_index, tokenizer=tokenizer
            )
            result["text_token_mapping"] = token_map
            result["mapping_note"] += (
                " BERT token identity and raw-text span are verified, but the link "
                "from this P2 text feature row to the token slot is not verified."
            )
        return result
    if mapping.get("sample_id") != sample_id or mapping.get("modality") != modality or (
            mapping.get("start_index"), mapping.get("end_index")) != (start_index, end_index):
        raise ValueError("mapping provenance does not match requested interval")
    status = mapping.get("grounding_status")
    if status not in ("verified", "approximate"):
        raise ValueError("mapping status must be verified or approximate")
    method = mapping.get("mapping_method")
    source = mapping.get("source_metadata")
    if not method or not source:
        raise ValueError("grounding requires mapping method and source metadata")
    if status == "approximate" and (not mapping.get("assumptions") or
                                     mapping.get("estimated_uncertainty_seconds") is None):
        raise ValueError("approximate grounding requires assumptions and uncertainty")
    evidence = mapping.get("evidence", {})
    text = evidence.get("text_fragment") if modality == "text" else None
    audio = evidence.get("audio_time_range") if modality == "audio" else None
    frame = evidence.get("video_frame_time") if modality == "vision" else None
    if modality == "text" and (not isinstance(text, str) or not text):
        raise ValueError("text grounding requires a nonempty mapped fragment")
    if modality == "audio" and (not isinstance(audio, dict) or
                                not 0 <= audio.get("start_seconds", -1) < audio.get("end_seconds", -1)):
        raise ValueError("audio grounding requires a valid mapped time range")
    if modality == "vision" and (not isinstance(frame, (int, float)) or frame < 0):
        raise ValueError("vision grounding requires a mapped frame time")
    note = f"Source: {source}."
    if status == "approximate":
        note += (f" Assumptions: {mapping['assumptions']}. "
                 f"Estimated uncertainty: {mapping['estimated_uncertainty_seconds']} seconds.")
    return {"grounding_status": status, "mapping_method": str(method),
            "media_file": str(media_file) if media_file is not None else None,
            "text_fragment": text, "audio_time_range": audio,
            "video_frame_time": frame, "mapping_note": note}
