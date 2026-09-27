"""Resolve Whisper word time intervals without changing raw timestamps.

Use a word interval only when it is finite, positive, within the source
timeline, and monotonic relative to the preceding word. Otherwise fall back to
the owning Whisper segment interval, provided that interval is itself valid.
No interval is interpolated or inferred.
"""
from __future__ import annotations

import math
from typing import Any


def valid_interval(start: Any, end: Any, duration: float) -> bool:
    try:
        start_f, end_f = float(start), float(end)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(start_f) and math.isfinite(end_f)
            and 0.0 <= start_f < end_f <= float(duration))


def resolve_word_interval(
    word: dict[str, Any],
    segment: dict[str, Any],
    duration: float,
    previous: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return source/effective interval while retaining raw word boundaries.

    `previous` is the prior raw word interval in the same sample. A word is
    nonmonotonic if either boundary moves backwards. Overlap alone is allowed
    because adjacent spoken words can overlap in Whisper output. Using raw
    prior intervals prevents a coarse segment fallback from cascading into
    later otherwise-valid word intervals.
    """
    raw_start, raw_end = word.get("start"), word.get("end")
    reason = None
    if not valid_interval(raw_start, raw_end, duration):
        reason = "invalid_word_interval"
    else:
        start, end = float(raw_start), float(raw_end)
        if previous is not None and (start < previous[0] or end < previous[1]):
            reason = "nonmonotonic_word_interval"
    if reason is None:
        return {"timestamp_source": "word", "raw_word_start": raw_start,
                "raw_word_end": raw_end, "effective_start": float(raw_start),
                "effective_end": float(raw_end), "fallback_reason": None}

    segment_start, segment_end = segment.get("start"), segment.get("end")
    if valid_interval(segment_start, segment_end, duration):
        return {"timestamp_source": "segment_fallback", "raw_word_start": raw_start,
                "raw_word_end": raw_end, "effective_start": float(segment_start),
                "effective_end": float(segment_end), "fallback_reason": reason}
    return {"timestamp_source": "unresolved", "raw_word_start": raw_start,
            "raw_word_end": raw_end, "effective_start": None,
            "effective_end": None, "fallback_reason": reason + "; invalid_segment_interval"}
