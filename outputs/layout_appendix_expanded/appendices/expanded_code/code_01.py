NUM_BINS = 50

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
        "binning_policy": ("optional_50_equal_duration_bins_for_downstream_"
                           "and_paper_not_contest_requirement"),
        "num_bins": NUM_BINS, "bins": bins,
        "reference_transcript_from_label_not_asr": row.get("text"),
        "q1_feature_status": {"text": "pending_gpu_asr_timestamps", "audio": "pending_gpu_encoder", "vision": "pending_gpu_encoder", "aligned": "pending_feature_extraction"},
        "valid_length": None, "padding_mask": None,
    }
    (item_dir / "timeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
