"""Q1 CPU preparation. Reads only Attachment 1; never imports Attachment 2 features."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "attachment_1"
VIDEOS = SOURCE / "videos"
OUT = ROOT / "outputs" / "q1_cpu"
FFBIN = Path(os.environ.get("FFMPEG_BIN", r"C:\Users\xuli2\anaconda3\envs\dvfb\Library\bin"))
FFPROBE = FFBIN / "ffprobe.exe"
FFMPEG = FFBIN / "ffmpeg.exe"
NUM_BINS = 50


def run(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        raise RuntimeError(f"{' '.join(args)}\n{p.stderr[-3000:]}")
    return p.stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def label_rows() -> list[dict]:
    sheet = load_workbook(SOURCE / "label-100.xlsx", read_only=True, data_only=True)["label"]
    rows = list(sheet.values)
    assert rows[0][:2] == ("video_id", "clip_id")
    return [dict(zip(rows[0], row)) for row in rows[1:] if row[0] is not None]


def probe(path: Path) -> dict:
    raw = json.loads(run([str(FFPROBE), "-v", "error", "-count_frames", "-show_format", "-show_streams", "-of", "json", str(path)]))
    video = next(s for s in raw["streams"] if s["codec_type"] == "video")
    audio = [s for s in raw["streams"] if s["codec_type"] == "audio"]
    container_duration = float(raw["format"].get("duration") or video.get("duration"))
    fps = str(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    frame_raw = run([str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)])
    frame_times = [float(line.split(",")[0]) for line in frame_raw.splitlines() if line.strip() and line.split(",")[0] != "N/A"]
    fps_float = float(Fraction(fps)) if fps != "0/0" else None
    duration = max(container_duration, max(frame_times, default=0) + (1 / fps_float if fps_float else 0))
    return {
        "duration": duration,
        "ffprobe_format_duration": container_duration,
        "fps": fps,
        "fps_float": fps_float,
        "frame_count": int(video.get("nb_read_frames") or len(frame_times)),
        "resolution": {"width": video["width"], "height": video["height"]},
        "video_codec": video.get("codec_name"),
        "video_start_time": float(video.get("start_time") or 0),
        "audio_streams": [{"index": a["index"], "codec": a.get("codec_name"), "sample_rate": int(a.get("sample_rate") or 0), "channels": a.get("channels"), "channel_layout": a.get("channel_layout"), "start_time": float(a.get("start_time") or 0), "duration": float(a.get("duration") or duration)} for a in audio],
        "frame_times": frame_times,
    }


def wav_info(path: Path) -> dict:
    import wave
    with wave.open(str(path), "rb") as w:
        return {"sample_rate": w.getframerate(), "channels": w.getnchannels(), "sample_count": w.getnframes(), "duration": w.getnframes() / w.getframerate()}


def prepare_one(path: Path, row: dict) -> dict:
    sample_id = f"{path.parent.name}__{path.stem}"
    item_dir = OUT / "samples" / sample_id
    item_dir.mkdir(parents=True, exist_ok=True)
    meta = probe(path)
    wav = OUT / "wav_16k_mono" / path.parent.name / f"{path.stem}.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    if not wav.exists():
        run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(wav)])
    wave_meta = wav_info(wav)
    assert wave_meta["sample_rate"] == 16000 and wave_meta["channels"] == 1
    duration = meta["duration"]
    bins = []
    for i in range(NUM_BINS):
        start, end = duration * i / NUM_BINS, duration * (i + 1) / NUM_BINS
        frame_indices = [j for j, t in enumerate(meta["frame_times"]) if start <= t < end or (i == NUM_BINS - 1 and t == duration)]
        bins.append({"index": i, "start": start, "end": end, "audio_sample_start": min(wave_meta["sample_count"], round(start * 16000)), "audio_sample_end": min(wave_meta["sample_count"], round(end * 16000)), "video_frame_indices": frame_indices})
    result = {
        "schema_version": "q1-cpu-1.0", "sample_id": sample_id,
        "video_id": path.parent.name, "clip_id": path.stem,
        "source_mp4": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(path),
        "wav_16k_mono": str(wav.relative_to(ROOT)).replace("\\", "/"),
        "wav_sha256": sha256(wav), "wav": wave_meta,
        "duration": duration, "ffprobe_format_duration": meta["ffprobe_format_duration"], "fps": meta["fps"], "fps_float": meta["fps_float"],
        "frame_count": meta["frame_count"], "resolution": meta["resolution"],
        "video_codec": meta["video_codec"], "video_start_time": meta["video_start_time"],
        "audio_streams": meta["audio_streams"], "frame_times": meta["frame_times"],
        "timeline": {"start": 0.0, "end": duration, "time_base": "seconds_from_mp4_presentation_start"},
        "binning_policy": "optional_50_equal_duration_bins_for_downstream_and_paper_not_contest_requirement",
        "num_bins": NUM_BINS, "bins": bins,
        "reference_transcript_from_label_not_asr": row.get("text"),
        "q1_feature_status": {"text": "pending_gpu_asr_timestamps", "audio": "pending_gpu_encoder", "vision": "pending_gpu_encoder", "aligned": "pending_feature_extraction"},
        "valid_length": None, "padding_mask": None,
    }
    (item_dir / "timeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if not FFMPEG.exists() or not FFPROBE.exists():
        raise FileNotFoundError(f"ffmpeg/ffprobe unavailable in {FFBIN}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = label_rows()
    labelled = [(str(r["video_id"]), str(r["clip_id"])) for r in rows]
    paths = list(VIDEOS.rglob("*.mp4"))
    found = [(p.parent.name, p.stem) for p in paths]
    duplicate_labels = sorted({x for x in labelled if labelled.count(x) > 1})
    duplicate_videos = sorted({x for x in found if found.count(x) > 1})
    audit = {"label_rows": len(rows), "video_files": len(paths), "matched": len(set(labelled) & set(found)), "missing_videos": sorted(set(labelled) - set(found)), "unlabelled_videos": sorted(set(found) - set(labelled)), "duplicate_labels": duplicate_labels, "duplicate_videos": duplicate_videos}
    (OUT / "id_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(rows) != 100 or audit["missing_videos"] or audit["unlabelled_videos"] or duplicate_labels or duplicate_videos:
        raise ValueError(f"Expected exact 100-to-100 mapping; see {OUT / 'id_audit.json'}")
    manifest = []
    log_path = OUT / "processing_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for index, row in enumerate(rows, 1):
            path = VIDEOS / str(row["video_id"]) / f"{row['clip_id']}.mp4"
            try:
                result = prepare_one(path, row)
                manifest.append({k: result[k] for k in ("sample_id", "video_id", "clip_id", "source_mp4", "source_sha256", "wav_16k_mono", "wav_sha256", "duration", "fps", "frame_count", "resolution", "num_bins")})
                status = "ok"
            except Exception as exc:
                status = "error"
                result = {"sample_id": f"{row['video_id']}__{row['clip_id']}", "error": repr(exc)}
            log_record = {"at_utc": datetime.now(timezone.utc).isoformat(), "index": index, "status": status, "sample_id": result["sample_id"]}
            if status == "error":
                log_record["error"] = result["error"]
            log.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            log.flush()
            print(index, status, result["sample_id"], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(manifest) != 100:
        raise RuntimeError(f"Only {len(manifest)} of 100 succeeded; inspect {log_path}")


if __name__ == "__main__":
    main()
