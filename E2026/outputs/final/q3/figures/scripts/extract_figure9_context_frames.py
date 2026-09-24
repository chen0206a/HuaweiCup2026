"""Extract fixed-position Attachment4 scene frames for Figure 9 v2.

These images are video context only. No HEAF value is read for frame selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs" / "final" / "q3" / "figures"
DATA = OUT / "data"
EXPLANATIONS = ROOT / "outputs" / "q3" / "final" / "attachment4_explanations.jsonl"
FRAME_PLAN = {
    "14": [(0.50, "sample14_context_frame.png")],
    "02": [
        (0.25, "sample02_context_25.png"),
        (0.50, "sample02_context_50.png"),
        (0.75, "sample02_context_75.png"),
    ],
}


def read_cases() -> dict[str, dict]:
    rows = (json.loads(line) for line in EXPLANATIONS.read_text(encoding="utf-8").splitlines())
    cases = {row["sample_id"]: row for row in rows if row["sample_id"] in FRAME_PLAN}
    if set(cases) != set(FRAME_PLAN):
        raise RuntimeError("Sample 02 and 14 must both be present in final JSONL")
    return cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_frames(video: Path) -> tuple[float, int, list[float]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to decode {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or reported_count <= 0:
        raise RuntimeError(f"Invalid video FPS/frame count: {video}")
    positions = []
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        positions.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    cap.release()
    if not positions:
        raise RuntimeError(f"No decoded frames: {video}")
    return fps, reported_count, positions


def write_selected(video: Path, selections: dict[int, Path]) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to reopen {video}")
    written = set()
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in selections:
            target = selections[index]
            encoded_ok, encoded_png = cv2.imencode(".png", frame)
            if not encoded_ok:
                raise RuntimeError(f"Cannot write frame {target}")
            target.write_bytes(encoded_png.tobytes())
            written.add(index)
        index += 1
    cap.release()
    if written != set(selections):
        raise RuntimeError(f"Selected frame indices not decoded: {set(selections) - written}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    cases = read_cases()
    records = []
    for sample_id, plan in FRAME_PLAN.items():
        media_ref = cases[sample_id]["raw_evidence"]["media_file"]
        video = (ROOT / media_ref).resolve()
        if video.name != f"{sample_id}.mp4" or not video.is_file():
            raise RuntimeError(f"Video identity mismatch for sample {sample_id}: {video}")
        fps, reported_count, pts = inspect_frames(video)
        nominal_duration = reported_count / fps
        actual_count = len(pts)
        reliable_pts = (
            np.all(np.isfinite(pts))
            and all(b >= a for a, b in zip(pts, pts[1:]))
            and pts[-1] > 0
            and abs(pts[-1] - (actual_count - 1) / fps) <= 1.0 / fps + 0.01
        )
        selected: dict[int, Path] = {}
        for relative_position, filename in plan:
            requested_time = relative_position * nominal_duration
            if reliable_pts:
                chosen = int(np.argmin(np.abs(np.asarray(pts) - requested_time)))
                selection_method = "nearest_decoded_frame_to_fixed_video_duration_fraction_by_decoder_pts"
            else:
                # OpenCV can report a nominal count larger than the sequence it
                # actually decodes. Keep the requested *duration* fraction and
                # choose the nearest available frame; do not rescale to the
                # shortened decoded sequence.
                chosen = min(int(round(requested_time * fps)), actual_count - 1)
                selection_method = "nearest_available_frame_to_fixed_reported_duration_fraction_by_index_fps"
            selected[chosen] = DATA / filename
            records.append({
                "sample_id": sample_id,
                "mp4_path": str(video),
                "mp4_sha256": sha256(video),
                "output_frame": str(Path("data") / filename),
                "requested_relative_position": relative_position,
                "requested_time_seconds": requested_time,
                "actual_decoded_frame_index_zero_based": chosen,
                "actual_decoded_time_seconds": pts[chosen] if reliable_pts else chosen / fps,
                "actual_time_source": "OpenCV CAP_PROP_POS_MSEC" if reliable_pts else "decoded_frame_index / reported_fps",
                "absolute_timing_error_seconds": abs((pts[chosen] if reliable_pts else chosen / fps) - requested_time),
                "selection_method": selection_method,
                "reported_fps": fps,
                "reported_frame_count": reported_count,
                "decoded_frame_count": actual_count,
                "reported_duration_seconds": nominal_duration,
                "purpose": "context_only",
                "grounding_status": "not_keyframe_mapping",
            })
        if len(selected) != len(plan):
            raise RuntimeError(f"Duplicate selected frame index for sample {sample_id}")
        write_selected(video, selected)

    manifest = {
        "figure": "figure9_q3_case_studies_v2",
        "frame_selection_independent_of_heaf": True,
        "scene_frame_grounding_note": "Frames show original video context only; no feature slot to video time mapping is inferred.",
        "frames": records,
    }
    target = DATA / "figure9_video_frame_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in records:
        print(row["sample_id"], row["output_frame"], row["actual_decoded_frame_index_zero_based"], row["actual_decoded_time_seconds"])


if __name__ == "__main__":
    main()
