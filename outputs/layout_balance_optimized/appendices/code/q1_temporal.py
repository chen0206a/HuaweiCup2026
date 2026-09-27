"""Q1 temporal feature contract and interval-overlap alignment.

All times are seconds relative to the MP4 presentation timeline. Encoder output
must retain source intervals; a shape-only resize is not accepted as alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class TimedFeature:
    modality: str
    start: float
    end: float
    vector: np.ndarray
    source: dict[str, Any]

    def validate(self, duration: float) -> None:
        if self.modality not in {"text", "audio", "vision"}:
            raise ValueError("Unknown modality")
        if not 0 <= self.start < self.end <= duration + 1e-6:
            raise ValueError("Feature interval lies outside source video")
        if self.vector.ndim != 1 or not np.isfinite(self.vector).all():
            raise ValueError("Feature vector must be finite and one-dimensional")
        required = {"text": {"transcript_span", "timestamp_source"}, "audio": {"wav_sample_start", "wav_sample_end"}, "vision": {"frame_indices"}}[self.modality]
        if not required.issubset(self.source):
            raise ValueError(f"Missing provenance: {required - self.source.keys()}")


def align_to_bins(features: list[TimedFeature], bins: list[dict], duration: float) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    """Overlap-weighted mean; returns [T,D], valid mask, and trace per bin.

    Missing bins remain zero with mask=False; zero alone never means padding.
    Each trace retains the original feature index and overlap in seconds.
    """
    if not features:
        raise ValueError("No encoder features supplied")
    dimension = len(features[0].vector)
    for feature in features:
        feature.validate(duration)
        if len(feature.vector) != dimension:
            raise ValueError("Inconsistent feature dimensions")
    values = np.zeros((len(bins), dimension), dtype=np.float32)
    valid = np.zeros(len(bins), dtype=bool)
    traces: list[list[dict]] = []
    for i, bin_info in enumerate(bins):
        start, end = float(bin_info["start"]), float(bin_info["end"])
        if bin_info["index"] != i or not 0 <= start < end <= duration + 1e-6:
            raise ValueError("Invalid unified time bin")
        overlaps = [(j, max(0.0, min(end, f.end) - max(start, f.start))) for j, f in enumerate(features)]
        overlaps = [(j, weight) for j, weight in overlaps if weight > 0]
        traces.append([{"feature_index": j, "overlap_seconds": weight, "source_interval": [features[j].start, features[j].end], "source": features[j].source} for j, weight in overlaps])
        if overlaps:
            total = sum(weight for _, weight in overlaps)
            values[i] = sum(features[j].vector * (weight / total) for j, weight in overlaps)
            valid[i] = True
    return values, valid, traces


def align_to_bins_hard(features: list[TimedFeature], bins: list[dict], duration: float) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    """Choose exactly one maximum-overlap native feature per covered time bin.

    Ties prefer the feature whose interval center is nearest the bin center,
    then the earlier native feature index. This is timestamp-overlap hard
    assignment; no interpolation, resizing, or feature blending occurs.
    """
    if not features:
        raise ValueError("No encoder features supplied")
    dimension = len(features[0].vector)
    for feature in features:
        feature.validate(duration)
        if len(feature.vector) != dimension:
            raise ValueError("Inconsistent feature dimensions")
    values = np.zeros((len(bins), dimension), dtype=np.float32)
    valid = np.zeros(len(bins), dtype=bool)
    traces: list[list[dict]] = []
    for i, bin_info in enumerate(bins):
        start, end = float(bin_info["start"]), float(bin_info["end"])
        if bin_info["index"] != i or not 0 <= start < end <= duration + 1e-6:
            raise ValueError("Invalid unified time bin")
        center = (start + end) / 2
        overlaps = [(j, max(0.0, min(end, f.end) - max(start, f.start))) for j, f in enumerate(features)]
        overlaps = [(j, weight) for j, weight in overlaps if weight > 0]
        if not overlaps:
            traces.append([])
            continue
        chosen, weight = min(overlaps, key=lambda pair: (-pair[1], abs((features[pair[0]].start + features[pair[0]].end) / 2 - center), pair[0]))
        values[i] = features[chosen].vector
        valid[i] = True
        traces.append([{"feature_index": chosen, "overlap_seconds": weight,
                        "source_interval": [features[chosen].start, features[chosen].end],
                        "source": features[chosen].source, "assignment": "maximum_overlap_hard"}])
    return values, valid, traces


ALIGNMENT_METHODS = ("maximum_overlap_hard", "nearest_center", "overlap_weighted_mean")


def align_to_bins_method(
    method: str, features: list[TimedFeature], bins: list[dict], duration: float,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Timestamp-overlap alignment variants with a shared observation mask.

    A bin is valid only if at least one native feature has positive temporal
    overlap with it. Nearest-center chooses the globally nearest native center
    only for such observed bins; it never fills a no-observation bin. Every
    trace item carries native indices, overlap durations and weights.
    """
    if method not in ALIGNMENT_METHODS:
        raise ValueError(f"Unknown alignment method: {method}")
    if not features:
        raise ValueError("No encoder features supplied")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    dimension = len(features[0].vector)
    for feature in features:
        feature.validate(duration)
        if len(feature.vector) != dimension:
            raise ValueError("Inconsistent feature dimensions")
    values = np.zeros((len(bins), dimension), dtype=np.float32)
    valid = np.zeros(len(bins), dtype=bool)
    traces: list[dict[str, Any]] = []
    for i, bin_info in enumerate(bins):
        start, end = float(bin_info["start"]), float(bin_info["end"])
        if bin_info["index"] != i or not 0 <= start < end <= duration + 1e-6:
            raise ValueError("Invalid unified time bin")
        center = (start + end) / 2
        observed = [(j, max(0.0, min(end, f.end) - max(start, f.start)))
                    for j, f in enumerate(features)]
        observed = [(j, seconds) for j, seconds in observed if seconds > 0]
        if not observed:
            traces.append({"source_feature_indices": [], "overlap_seconds": [],
                           "normalized_weights": [], "source_intervals": [],
                           "sources": [], "assignment": method})
            continue
        if method == "maximum_overlap_hard":
            chosen = min(observed, key=lambda pair: (-pair[1],
                         abs((features[pair[0]].start + features[pair[0]].end) / 2 - center), pair[0]))
            selected = [chosen]
        elif method == "nearest_center":
            chosen_index = min(range(len(features)), key=lambda j: (
                abs((features[j].start + features[j].end) / 2 - center), j))
            chosen_overlap = max(0.0, min(end, features[chosen_index].end) - max(start, features[chosen_index].start))
            selected = [(chosen_index, chosen_overlap)]
        else:
            selected = observed
        if method in {"maximum_overlap_hard", "nearest_center"}:
            weights = [1.0]
        else:
            total_overlap = sum(seconds for _, seconds in selected)
            weights = [seconds / (total_overlap + epsilon) for _, seconds in selected]
        values[i] = sum(features[j].vector * weight for (j, _), weight in zip(selected, weights))
        valid[i] = True
        traces.append({
            "source_feature_indices": [j for j, _ in selected],
            "overlap_seconds": [seconds for _, seconds in selected],
            "normalized_weights": weights,
            "source_intervals": [[features[j].start, features[j].end] for j, _ in selected],
            "sources": [features[j].source for j, _ in selected],
            "assignment": method,
        })
    return values, valid, traces


def validate_timeline(timeline: dict) -> None:
    bins = timeline["bins"]
    duration = timeline["duration"]
    if timeline["timeline"] != {"start": 0.0, "end": duration, "time_base": "seconds_from_mp4_presentation_start"}:
        raise ValueError("Continuous source timeline is absent or inconsistent")
    if timeline["num_bins"] != len(bins) or timeline["num_bins"] != 50:
        raise ValueError("Wrong bin count")
    if abs(bins[0]["start"]) > 1e-8 or abs(bins[-1]["end"] - duration) > 1e-7:
        raise ValueError("Bins do not span full video")
    for i, item in enumerate(bins):
        if item["index"] != i or item["start"] >= item["end"]:
            raise ValueError("Invalid bin")
        if i and abs(bins[i - 1]["end"] - item["start"]) > 1e-7:
            raise ValueError("Bin gap or overlap")
    count = len(timeline["frame_times"])
    if timeline["frame_count"] != count:
        raise ValueError("Frame count differs from decoded frame PTS count")
    mapped = [frame for item in bins for frame in item["video_frame_indices"]]
    if sorted(mapped) != list(range(count)):
        raise ValueError("Frame-to-bin mapping is incomplete or duplicated")
    for item in bins:
        for frame in item["video_frame_indices"]:
            pts = timeline["frame_times"][frame]
            if not (item["start"] <= pts < item["end"] or (item["index"] == len(bins) - 1 and pts == duration)):
                raise ValueError("Frame PTS assigned to wrong bin")
    sample_count = timeline["wav"]["sample_count"]
    if bins[0]["audio_sample_start"] != 0 or bins[-1]["audio_sample_end"] > sample_count:
        raise ValueError("WAV sample mapping lies outside audio")
    for i, item in enumerate(bins):
        expected_start = min(sample_count, round(item["start"] * 16000))
        expected_end = min(sample_count, round(item["end"] * 16000))
        if (item["audio_sample_start"], item["audio_sample_end"]) != (expected_start, expected_end):
            raise ValueError("WAV sample indices inconsistent with bin time")
        if i and bins[i - 1]["audio_sample_end"] != item["audio_sample_start"]:
            raise ValueError("WAV sample gap or overlap")
