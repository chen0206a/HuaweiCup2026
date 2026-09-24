"""Verify and index the frozen Q2 assets; never train or rewrite input assets.

Run from the repository root with ``python E2026/scripts/lock_q2_assets.py``.
Attachment 3 is only enumerated and hashed as opaque files.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
METRICS = PROJECT / "outputs/metrics"
CHECKPOINTS = PROJECT / "outputs/checkpoints"
BENCHMARK = METRICS / "b2_benchmark_definition.json"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
EXPECTED_COUNTS = {"train": 3395, "valid": 728, "test": 727}
EXPECTED_DIMS = {"text": 768, "audio": 74, "vision": 35}
CLASS_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}
NOW = datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path, *, hash_file: bool = True) -> dict:
    path = path.resolve()
    result = {
        "absolute_path": path.as_posix(),
        "relative_path": path.relative_to(PROJECT).as_posix()
        if path.is_relative_to(PROJECT)
        else Path(__import__("os").path.relpath(path, PROJECT)).as_posix(),
        "exists_locally": path.is_file(),
    }
    if path.is_file():
        result.update(size_bytes=path.stat().st_size,
                      sha256=sha256(path) if hash_file else None,
                      file_type=path.suffix.lower() or "extensionless")
    return result


def json_file(name: str) -> dict:
    with (METRICS / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def text_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def source_commit(path: Path) -> str | None:
    relative = path.relative_to(REPO).as_posix()
    command = ["git", "log", "-1", "--format=%H", "--", relative]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout.strip() or None


def resolve_raw(name: str) -> Path:
    # Preserve the user's current local layout. Do not move or copy the raw file.
    candidates = [PROJECT / "data/raw/attachment2" / name, PROJECT / "data/raw" / name]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one local {name}, found: {found}")
    return found[0]


def clip_id(video: object, clip: object) -> str:
    if isinstance(clip, (int, np.integer)) or isinstance(clip, float) and clip.is_integer():
        clip = int(clip)
    return f"{video}$_${clip}"


def audit_attachment2(pkl_path: Path, xlsx_path: Path) -> tuple[dict, list[str]]:
    with pkl_path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or set(data) != set(EXPECTED_COUNTS):
        raise RuntimeError(f"Unexpected aligned pickle top-level keys: {list(data)}")
    sheet = pd.read_excel(xlsx_path, sheet_name="label")
    expected_columns = {"video_id", "clip_id", "label", "annotation", "mode"}
    if not expected_columns.issubset(sheet.columns):
        raise RuntimeError(f"Unexpected label columns: {list(sheet.columns)}")
    sheet_ids = [clip_id(v, c) for v, c in zip(sheet.video_id, sheet.clip_id)]
    if len(sheet) != 4850 or len(sheet_ids) != len(set(sheet_ids)):
        raise RuntimeError("Unexpected label.xlsx row count or duplicate ID")
    lookup = sheet.set_index(pd.Index(sheet_ids))

    split_details: dict[str, dict] = {}
    all_ids: dict[str, set[str]] = {}
    valid_ids: list[str] = []
    for split, expected_count in EXPECTED_COUNTS.items():
        entry = data[split]
        ids = [str(value) for value in entry["id"]]
        if len(ids) != expected_count or len(ids) != len(set(ids)):
            raise RuntimeError(f"Unexpected count or duplicate clip ID in {split}")
        if set(ids) != set(lookup.index[lookup["mode"] == split]):
            raise RuntimeError(f"XLSX/PKL IDs disagree for {split}")
        all_ids[split] = set(ids)
        if split == "valid":
            valid_ids = ids
        shapes: dict[str, list[int]] = {}
        dtypes: dict[str, str] = {}
        zero_summary: dict[str, dict] = {}
        token_mask = np.asarray(entry["text_bert"])[:, 1, :]
        if token_mask.shape != (expected_count, 50) or not np.isin(token_mask, (0, 1)).all():
            raise RuntimeError(f"Invalid text_bert attention mask in {split}")
        pad = token_mask.astype(bool)
        if not pad[:, 0].all() or np.any(np.diff(token_mask, axis=1) > 0):
            raise RuntimeError(f"Non-prefix attention mask in {split}")
        for modality, dim in EXPECTED_DIMS.items():
            values = np.asarray(entry[modality])
            if values.shape != (expected_count, 50, dim):
                raise RuntimeError(f"Unexpected {split}/{modality} shape: {values.shape}")
            shapes[modality] = list(values.shape)
            dtypes[modality] = str(values.dtype)
            if modality in ("audio", "vision") and np.any(values[~pad] != 0):
                raise RuntimeError(f"Nonzero {modality} outside valid text positions in {split}")
            native_zero = np.all(values == 0, axis=-1)
            zero_summary[modality] = {
                "zero_valid_timesteps": int((native_zero & pad).sum()),
                "valid_timesteps": int(pad.sum()),
                "all_zero_across_valid_region_samples": int(np.all(native_zero | ~pad, axis=1).sum()),
                "nonzero_timesteps_outside_padding_mask": int((~native_zero & ~pad).sum()),
            }
        for field in ("classification_labels", "regression_labels"):
            values = np.asarray(entry[field])
            if values.shape != (expected_count,):
                raise RuntimeError(f"Unexpected {split}/{field} shape: {values.shape}")
            dtypes[field] = str(values.dtype)
        classes = np.asarray(entry["classification_labels"])
        regression = np.asarray(entry["regression_labels"])
        if not np.isin(classes, (0, 1, 2)).all():
            raise RuntimeError(f"Unexpected class code in {split}")
        row_order = lookup.loc[ids]
        if not np.allclose(row_order["label"].to_numpy(dtype=float), regression, atol=1e-5):
            raise RuntimeError(f"Regression label mismatch in {split}")
        annotations = row_order["annotation"].astype(str).str.lower().to_numpy()
        expected_names = np.asarray([CLASS_MAPPING[int(value)].lower() for value in classes])
        if not np.array_equal(annotations, expected_names):
            raise RuntimeError(f"Class/annotation mismatch in {split}")
        split_details[split] = {
            "sample_count": expected_count,
            "modality_shapes": shapes,
            "dtypes": dtypes,
            "class_counts": {CLASS_MAPPING[c]: int((classes == c).sum()) for c in CLASS_MAPPING},
            "native_effective_zero": zero_summary,
            "padding_valid_timestep_count": int(pad.sum()),
        }
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if all_ids[a] & all_ids[b]:
            raise RuntimeError(f"Clip ID overlap: {a}/{b}")
    result = {
        "status": "VERIFIED",
        "audited_at_utc": NOW,
        "raw_files": {"aligned_50.pkl": path_info(pkl_path), "label.xlsx": path_info(xlsx_path)},
        "expected_historical_server_paths": {
            "aligned_50.pkl": "/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl",
            "label.xlsx": "/root/workspace/E2026/data/raw/attachment2/label.xlsx",
        },
        "local_path_note": "Current local files are in data/raw, not data/raw/attachment2; no file was moved.",
        "top_level_keys": list(data),
        "splits": split_details,
        "split_source": "Existing train/valid/test top-level keys in aligned_50.pkl; XLSX mode agrees; no new split generated.",
        "sequence_length": 50,
        "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
        "label_columns": ["label", "annotation", "mode"],
        "id_columns": ["video_id", "clip_id"],
        "id_rule": "video_id + '$_$' + clip_id",
        "xlsx_row_count": len(sheet),
        "padding_mask_source": "text_bert[:, 1, :] == 1, verified binary valid-prefix mask",
        "native_zero_rule": "all original feature dimensions are zero within padding_mask=True; data-quality only, never missing/padding by itself",
        "duplicate_clip_id_audit": "PASS: unique within each split and XLSX; no overlap across splits",
        "xlsx_id_and_label_crosscheck": "PASS: one-to-one IDs, mode, regression label, classification annotation",
        "test_usage_this_stage": "structural and label integrity verification only; no prediction or model selection",
    }
    return result, valid_ids


def verify_benchmark(valid_ids: list[str]) -> dict:
    digest = sha256(BENCHMARK)
    if digest != BENCHMARK_SHA256:
        raise RuntimeError(f"Frozen benchmark hash mismatch: {digest}")
    definition = json_file("b2_benchmark_definition.json")
    scenarios = definition["scenarios"]
    single = [item for item in scenarios if "+" not in item["modalities"]]
    double = [item for item in scenarios if "+" in item["modalities"]]
    if definition["seed"] != 20260923 or len(scenarios) != 54 or len(single) != 45 or len(double) != 9:
        raise RuntimeError("Frozen benchmark seed or scenario count changed")
    if len({item["scenario_id"] for item in scenarios}) != 54:
        raise RuntimeError("Duplicate benchmark scenario ID")
    sample_id_digest = hashlib.sha256(json.dumps(valid_ids, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "status": "FROZEN_VERIFIED",
        "definition_file": path_info(BENCHMARK),
        "benchmark_seed": 20260923,
        "scenario_counts": {"clean": 1, "single_modality_missing": 45,
                            "double_modality_missing": 9},
        "scenarios": scenarios,
        "validation_sample_ids_in_order": valid_ids,
        "validation_sample_ids_json_sha256": sample_id_digest,
        "sample_id_source": "aligned_50.pkl top-level valid/id; order preserved",
        "mask_rule": definition["length_rule"],
        "locations": definition["locations"],
        "padding_rule": definition["padding"],
        "attachment2_test_used_for_selection": False,
        "benchmark_regenerated": False,
    }


def verify_checkpoints(summary: dict) -> tuple[dict, dict[str, Path]]:
    names = {
        42: ("b0_weighted_ce_best_selection_score.pt", "b5_pooling_p2_best_robust_score.pt"),
        43: ("b21_b0_seed_43_best_selection_score.pt", "b5_p2_multiseed_seed43_best_robust_score.pt"),
        44: ("b21_b0_seed_44_best_selection_score.pt", "b5_p2_multiseed_seed44_best_robust_score.pt"),
    }
    entries = []
    key_paths = {}
    for seed, (b0_name, p2_name) in names.items():
        b0_path, p2_path = CHECKPOINTS / b0_name, CHECKPOINTS / p2_name
        if not b0_path.is_file() or not p2_path.is_file():
            raise RuntimeError(f"Missing critical seed {seed} B0/P2 checkpoint")
        b0 = torch.load(b0_path, map_location="cpu", weights_only=False)
        p2 = torch.load(p2_path, map_location="cpu", weights_only=False)
        b0_state, p2_state = b0["model_state_dict"], p2["model_state_dict"]
        if not all(torch.equal(value, p2_state[key]) for key, value in b0_state.items()):
            raise RuntimeError(f"Frozen B0 tensor mismatch inside seed {seed} P2 checkpoint")
        if sum(value.numel() for value in b0_state.values()) != 163460:
            raise RuntimeError(f"Unexpected B0 parameter count at seed {seed}")
        if sum(value.numel() for value in p2_state.values()) != 164343:
            raise RuntimeError(f"Unexpected P2 parameter count at seed {seed}")
        per_seed = summary["per_seed"][str(seed)]
        if b0["best_epoch"] != per_seed["B0_best_epoch"] or p2["epoch"] != per_seed["P2_best_robust_epoch"]:
            raise RuntimeError(f"Checkpoint epoch/result mismatch at seed {seed}")
        if abs(b0["selection_score"] - per_seed["B0"]["clean"]["selection_score"]) > 1e-10:
            raise RuntimeError(f"B0 clean score mismatch at seed {seed}")
        if abs(p2["robust_score"] - per_seed["P2"]["robust_score"]) > 1e-10:
            raise RuntimeError(f"P2 robust score mismatch at seed {seed}")
        if p2["benchmark_sha256"] != BENCHMARK_SHA256:
            raise RuntimeError(f"P2 checkpoint benchmark hash mismatch at seed {seed}")
        if seed in (43, 44):
            expected = summary["seed43_44_details"][str(seed)]["baseline_checkpoint_sha256"]
            if sha256(b0_path) != expected:
                raise RuntimeError(f"Seed {seed} B0 hash differs from experiment record")
        for model, path, epoch, criterion, metrics in (
            ("B0-WCE", b0_path, b0["best_epoch"], "best_validation_selection_score", per_seed["B0"]),
            ("B5-P2", p2_path, p2["epoch"], "best_validation_robust_score", per_seed["P2"]),
        ):
            entry = {
                "model": model, "seed": seed, "architecture": model,
                "checkpoint": path_info(path), "best_epoch": epoch,
                "selection_criterion": criterion,
                "validation_metrics": {k: metrics[k] for k in ("clean", "mean_missing", "robust_score")},
                "parameter_count_total": 163460 if model == "B0-WCE" else 164343,
                "parameter_count_trainable_during_original_training": 163460 if model == "B0-WCE" else 883,
                "parent_b0_checkpoint": path_info(b0_path) if model == "B5-P2" else None,
                "is_final_main_checkpoint": model == "B5-P2",
            }
            entries.append(entry)
            key_paths[f"{model}:{seed}"] = path
    best_clean = {}
    for seed, name in ((42, "b5_pooling_p2_best_clean_score.pt"),
                       (43, "b5_p2_multiseed_seed43_best_clean_score.pt"),
                       (44, "b5_p2_multiseed_seed44_best_clean_score.pt")):
        best_clean[str(seed)] = path_info(CHECKPOINTS / name)
    return {"status": "VERIFIED", "generated_at_utc": NOW,
            "checkpoints": entries, "p2_best_clean_checkpoints_retained_reference": best_clean,
            "parent_b0_tensors_equal": True}, key_paths


def local_inventory() -> dict:
    # Generated manifests are excluded to avoid self-referential hashes.
    files = []
    for directory in (PROJECT / "data/raw", PROJECT / "data/processed"):
        if directory.is_dir():
            for path in sorted(item for item in directory.rglob("*") if item.is_file()):
                record = path_info(path)
                record["category"] = "raw" if directory.name == "raw" and path.suffix in (".pkl", ".xlsx") else "derived"
                files.append(record)
    return {"generated_at_utc": NOW, "inventory_scope": "recursive data/raw and data/processed; generated data/manifests excluded to avoid self-reference",
            "files": files, "attachment3_status": "SEALED"}


def external_inventory() -> tuple[dict, dict]:
    contest = REPO / "第二十三届中国研究生数学建模竞赛 - 中文题目/中文题目/E题/E题数据/E题数据"
    attachment3 = contest / "附件3-模态缺失特征样本"
    attachment4 = contest / "附件4-可解释专项视频样本与特征文件"
    sealed_files = [path_info(path) for path in sorted(attachment3.rglob("*")) if path.is_file()] if attachment3.is_dir() else []
    attachment3_record = {"attachment3_status": "SEALED", "directory": attachment3.as_posix(),
                          "relative_directory": Path(__import__("os").path.relpath(attachment3, PROJECT)).as_posix(),
                          "file_metadata_only": sealed_files,
                          "content_parsed_or_deserialized": False}
    attachment4_files = [path_info(path) for path in sorted(attachment4.rglob("*")) if path.is_file()] if attachment4.is_dir() else []
    attachment4_dirs = [Path(__import__("os").path.relpath(path, PROJECT)).as_posix()
                        for path in sorted(attachment4.rglob("*")) if path.is_dir()] if attachment4.is_dir() else []
    attachment4_record = {"status": "FILENAME_SIZE_HASH_ONLY", "directory": attachment4.as_posix(),
                          "relative_directory": Path(__import__("os").path.relpath(attachment4, PROJECT)).as_posix(),
                          "subdirectories_relative_to_project": attachment4_dirs,
                          "files": attachment4_files, "content_interpreted": False}
    return attachment3_record, attachment4_record


def experiment_index(summary: dict) -> dict:
    b21, b3 = json_file("b21_multiseed_summary.json"), json_file("b3_summary.json")
    b31, b32 = json_file("b31_alpha_sweep.json"), json_file("b32_multiseed_summary.json")
    b4, pool = json_file("b4_reliability_metrics.json"), json_file("b5_pooling_metrics.json")
    fusion, l1 = json_file("b5_fusion_metrics.json"), json_file("b5_l1_lambda_metrics.json")
    n1 = json_file("b5_n1_text_zscore_metrics.json")
    entries = []

    def add(name: str, seeds: list[int], hypothesis: str, component: str,
            result: object, conclusion: str, result_file: str,
            checkpoint_names: list[str], role: str | None = None) -> None:
        path = METRICS / result_file
        entry = {
            "experiment_name": name, "role": role,
            "source_commit": source_commit(path), "training_seeds": seeds,
            "hypothesis": hypothesis, "changed_component": component,
            "robust_result": result, "conclusion": conclusion,
            "output_paths": [path_info(path)],
            "checkpoint_paths": [path_info(CHECKPOINTS / filename) for filename in checkpoint_names],
        }
        entries.append(entry)

    add("B0-WCE", [42, 43, 44], "Reference for Q2", "masked mean B0, weighted CE",
        {"three_seed_mean": summary["models"]["B0-WCE"]["robust_score"]["mean"],
         "sample_std": summary["models"]["B0-WCE"]["robust_score"]["std"]},
        "KEEP", "b5_p2_multiseed_summary.json",
        ["b0_weighted_ce_best_selection_score.pt", "b21_b0_seed_43_best_selection_score.pt",
         "b21_b0_seed_44_best_selection_score.pt"], "BASELINE")
    add("B1 Transformer", [42, 43, 44], "Test unimodal temporal encoding", "2-layer temporal Transformer per modality",
        {"three_seed_mean": b21["models"]["B1-WCE"]["robust_score"]["mean"],
         "sample_std": b21["models"]["B1-WCE"]["robust_score"]["std"]}, "STOP", "b21_multiseed_summary.json",
        ["b1_weighted_ce_best_selection_score.pt", "b21_b1_seed_43_best_selection_score.pt",
         "b21_b1_seed_44_best_selection_score.pt"])
    add("B2 BlockMask", [42, 43, 44], "Test missing augmentation", "continuous block missing during training",
        {"B2_B0_three_seed_mean": b21["models"]["B2-B0-BlockMask"]["robust_score"]["mean"],
         "B2_B0_sample_std": b21["models"]["B2-B0-BlockMask"]["robust_score"]["std"],
         "B2_B1_three_seed_mean": b21["models"]["B2-B1-BlockMask"]["robust_score"]["mean"],
         "B2_B1_sample_std": b21["models"]["B2-B1-BlockMask"]["robust_score"]["std"]},
        "STOP", "b21_multiseed_summary.json",
        ["b2_b0_best_robust_score.pt", "b2_b1_best_robust_score.pt",
         "b21_b2_b0_seed_43_best_robust_score.pt", "b21_b2_b0_seed_44_best_robust_score.pt",
         "b21_b2_b1_seed_43_best_robust_score.pt", "b21_b2_b1_seed_44_best_robust_score.pt"])
    add("B3 reconstruction", [42], "Recover missing latent features", "joint cross-modal reconstruction",
        {"seed42": b3["comparison"]["B3-B0-Reconstruction"]["robust_score"]}, "STOP", "b3_summary.json",
        ["b3_b0_best_clean_score.pt", "b3_b0_best_robust_score.pt"])
    add("B3.1 frozen reconstruction", [42], "Avoid backbone drift", "frozen B0, global alpha blend",
        {"seed42_by_alpha": {a: v["robust_score"] for a, v in b31["alphas"].items()}},
        "DIAGNOSTIC", "b31_alpha_sweep.json", [Path(b31["checkpoint"]).name])
    add("B3.2 reconstruction stability", [42, 43, 44], "Check global alpha across seeds", "frozen reconstruction multi-seed",
        {"three_seed_mean_by_alpha": {a: v["robust_score"]["robust_score"]["mean"]
                                       for a, v in b32["summary_mean_sample_std"].items()},
         "all_seed_gain_consistent": b32["robust_gain_consistent_across_all_seeds"]},
        "STOP", "b32_multiseed_summary.json",
        ["b31_frozen_reconstructors_seed42.pt", "b32_frozen_reconstructors_seed_43.pt",
         "b32_frozen_reconstructors_seed_44.pt"])
    add("B4' dynamic gate", [42], "Test reliability-aware dynamic fusion", "reliability gate",
        {"seed42": b4["validation"]["robust_score"]}, "DIAGNOSTIC", "b4_reliability_metrics.json",
        ["b4_reliability_best_clean_score.pt", "b4_reliability_best_robust_score.pt"])
    add("B5-P1 Mean+Max", [42], "Test pooling bottleneck", "mean plus max residual",
        {"seed42": pool["P1"]["validation"]["robust_score"]}, "DIAGNOSTIC", "b5_pooling_metrics.json",
        ["b5_pooling_p1_best_clean_score.pt", "b5_pooling_p1_best_robust_score.pt"])
    add("B5-P2 Attention", [42, 43, 44], "Test pooling bottleneck", "mean plus lightweight attention residual",
        {"three_seed_mean": summary["models"]["B5-P2"]["robust_score"]["mean"],
         "sample_std": summary["models"]["B5-P2"]["robust_score"]["std"],
         "paired_delta_mean": summary["paired_deltas"]["robust_score"]["mean"],
         "paired_delta_sample_std": summary["paired_deltas"]["robust_score"]["sample_std"]},
        "KEEP", "b5_p2_multiseed_summary.json",
        ["b5_pooling_p2_best_robust_score.pt", "b5_p2_multiseed_seed43_best_robust_score.pt",
         "b5_p2_multiseed_seed44_best_robust_score.pt", "b5_pooling_p2_best_clean_score.pt",
         "b5_p2_multiseed_seed43_best_clean_score.pt", "b5_p2_multiseed_seed44_best_clean_score.pt"], "FINAL")
    add("B5-F low-rank fusion", [42], "Test pairwise multiplicative interaction", "low-rank fusion residual",
        {"seed42": fusion["F1"]["validation"]["robust_score"]}, "STOP", "b5_fusion_metrics.json",
        ["b5_fusion_f1_best_clean_score.pt", "b5_fusion_f1_best_robust_score.pt"])
    add("B5-L1 lambda balance", [42], "Test classification/regression loss balance", "lambda_reg grid only",
        {name: value["best_robust_validation"]["robust_score"] for name, value in l1["candidates"].items()},
        "STOP", "b5_l1_lambda_metrics.json",
        [Path(value["checkpoints"][choice]).name for value in l1["candidates"].values()
         for choice in ("best_clean", "best_robust")])
    add("B5-H0 head diagnostic", [42], "Measure classification/regression head complementarity", "no training; predictions only",
        None, "DIAGNOSTIC", "b5_h0_head_diagnostic.json", ["b0_weighted_ce_best_selection_score.pt"])
    add("B5-N1 text z-score", [42], "Test train-only text normalization", "text featurewise z-score",
        {"seed42": n1["N1_best_robust"]["metrics"]["robust_score"]}, "STOP",
        "b5_n1_text_zscore_metrics.json",
        ["b5_n1_text_zscore_best_clean_score.pt", "b5_n1_text_zscore_best_robust_score.pt"])
    return {"generated_at_utc": NOW, "metric_scope": "attachment2 validation clean + frozen 54 missing scenarios",
            "experiment_entries": entries, "final_choice": "B5-P2", "baseline": "B0-WCE",
            "note": "Missing local optional checkpoints are recorded explicitly; no metric was inferred from a missing checkpoint."}


def make_config(model: str, raw_path: Path, checkpoint_manifest: dict) -> dict:
    base = {
        "run_name": "Q2-final-B0-WCE" if model == "b0" else "Q2-final-B5-P2",
        "model_status": "LOCKED", "project_root": PROJECT.as_posix(),
        "data": {"pkl_path": raw_path.relative_to(PROJECT).as_posix(),
                 "pkl_path_absolute": raw_path.resolve().as_posix(),
                 "train_split": "train", "valid_split": "valid", "test_split": "test",
                 "batch_size": 128, "num_workers": 0, "sequence_length": 50,
                 "feature_dimensions": EXPECTED_DIMS,
                 "padding_mask_source": "text_bert[:,1,:] == 1"},
        "preprocessing": {"normalization": "none"},
        "model": {"hidden_dim": 128, "fusion_dim": 128, "dropout": 0.1,
                  "classification_classes": 3, "regression_outputs": 1},
        "training": {"seeds": [42, 43, 44], "epochs": 80, "patience": 12,
                     "learning_rate": 0.001, "weight_decay": 0.0001,
                     "optimizer": "AdamW", "classification_loss": "balanced_train_weighted_cross_entropy",
                     "regression_loss": "SmoothL1", "lambda_reg": 1.0,
                     "class_weighting": "balanced_train", "augmentation": "none",
                     "evaluate_test": False},
        "benchmark": {"definition": BENCHMARK.relative_to(PROJECT).as_posix(),
                      "seed": 20260923, "sha256": BENCHMARK_SHA256,
                      "selection_score": "0.25*Accuracy + 0.25*MacroF1 + 0.25*(1-MAE/6) + 0.25*(Pearson+1)/2",
                      "robust_score": "0.5*clean_selection_score + 0.5*mean_missing_selection_score"},
    }
    if model == "b0":
        base["model"].update(architecture="B0Baseline", pooling="masked_mean",
                             modality_projection="Linear -> LayerNorm -> GELU -> Dropout",
                             fusion="concat -> Linear(384,128) -> LayerNorm -> GELU -> Dropout",
                             heads=["Linear(128,3)", "Linear(128,1)"])
        base["training"]["checkpoint_selection"] = "best_validation_selection_score"
    else:
        base["model"].update(architecture="B5PoolingResidual", mode="mean_attention",
                             attention_scorer="per-modality Linear(input_dim,1)",
                             attention_padding="masked before softmax",
                             residual="project(masked_mean(x)) + gamma*project(attention_pool(x))",
                             gamma_initial_value=0)
        base["training"].update(freeze_b0_parameters=True, frozen_b0_eval_mode=True,
                                trainable_components=["attention_scorer", "gamma"],
                                checkpoint_selection="best_validation_robust_score",
                                initialization_protocol="corresponding_seed_b0_wce_best_selection_score")
    base["checkpoints"] = {
        str(item["seed"]): item["checkpoint"]["relative_path"]
        for item in checkpoint_manifest["checkpoints"]
        if item["model"] == ("B0-WCE" if model == "b0" else "B5-P2")
    }
    return base


def main() -> None:
    pkl_path, xlsx_path = resolve_raw("aligned_50.pkl"), resolve_raw("label.xlsx")
    original_hashes = {path.as_posix(): sha256(path) for path in (pkl_path, xlsx_path, BENCHMARK)}
    attachment2, valid_ids = audit_attachment2(pkl_path, xlsx_path)
    benchmark = verify_benchmark(valid_ids)
    summary = json_file("b5_p2_multiseed_summary.json")
    if summary["benchmark_sha256"] != BENCHMARK_SHA256 or summary["training_seeds"] != [42, 43, 44]:
        raise RuntimeError("Multi-seed result does not match frozen protocol")
    checkpoints, key_paths = verify_checkpoints(summary)
    original_hashes.update({path.as_posix(): sha256(path) for path in key_paths.values()})
    inventory = local_inventory()
    attachment3, attachment4 = external_inventory()
    experiments = experiment_index(summary)
    baseline = summary["models"]["B0-WCE"]["robust_score"]
    final = summary["models"]["B5-P2"]["robust_score"]
    paired = summary["paired_deltas"]["robust_score"]
    lock = {
        "model_status": "LOCKED", "model_name": "B5-P2 Attention Residual Pooling",
        "model_version": "q2_final_2026-09-24", "generated_at_utc": NOW,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "git_commit_role": "source tree before this generated lock commit",
        "baseline": "B0-WCE", "architecture": "B0 masked mean plus per-modality scalar attention residual; frozen B0",
        "input_dimensions": {"sequence_length": 50, **EXPECTED_DIMS},
        "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
        "loss": {"classification": "balanced WeightedCE", "regression": "SmoothL1", "lambda_reg": 1.0},
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001},
        "training_protocol": {"batch_size": 128, "max_epochs": 80, "patience": 12,
                              "normalization": "none", "augmentation": "none",
                              "P2_frozen_B0_eval_mode": True, "P2_trainable": "attention scorers + 3 gamma scalars"},
        "benchmark": {"sha256": BENCHMARK_SHA256, "seed": 20260923,
                      "definition": path_info(BENCHMARK), "clean_condition_count": 1,
                      "missing_scenario_count": 54},
        "training_seeds": [42, 43, 44],
        "checkpoint_manifest": {
            "absolute_path": (PROJECT / "outputs/final/q2/q2_checkpoint_manifest.json").as_posix(),
            "relative_path": "outputs/final/q2/q2_checkpoint_manifest.json",
        },
        "checkpoint_paths": {key: path_info(path) for key, path in key_paths.items()},
        "parameter_counts": {"B0_total": 163460, "B0_trainable": 163460,
                             "P2_total": 164343, "P2_trainable": 883},
        "validation_summary": {"B0_clean": summary["models"]["B0-WCE"]["clean"],
                               "B0_mean_missing": summary["models"]["B0-WCE"]["mean_missing"],
                               "P2_clean": summary["models"]["B5-P2"]["clean"],
                               "P2_mean_missing": summary["models"]["B5-P2"]["mean_missing"],
                               "B0_robust_mean": baseline["mean"], "B0_robust_sample_std": baseline["std"],
                               "P2_robust_mean": final["mean"], "P2_robust_sample_std": final["std"],
                               "paired_delta_mean": paired["mean"], "paired_delta_sample_std": paired["sample_std"],
                               "paired_delta_by_seed": {str(seed): summary["per_seed"][str(seed)]["delta_robust_P2_minus_B0"] for seed in (42, 43, 44)},
                               "interpretation": "2/3 positive; seed44 approximately tied; seed-sensitive, not universal improvement"},
        "attachment3_status": "SEALED", "attachment2_test_used_for_selection_this_stage": False,
        "source_result": path_info(METRICS / "b5_p2_multiseed_summary.json"),
    }
    # All critical checks above complete before writing any generated metadata.
    manifests = PROJECT / "data/manifests"
    final_dir = PROJECT / "outputs/final/q2"
    json_write(manifests / "data_inventory.json", inventory)
    json_write(manifests / "attachment2_manifest.json", attachment2)
    json_write(manifests / "q2_missing_benchmark_manifest.json", benchmark)
    json_write(manifests / "attachment3_sealed_inventory.json", attachment3)
    json_write(manifests / "q3/attachment4_inventory.json", attachment4)
    (PROJECT / "outputs/q3").mkdir(parents=True, exist_ok=True)
    (PROJECT / "experiments/q3").mkdir(parents=True, exist_ok=True)
    for path in (PROJECT / "outputs/q3/.gitkeep", PROJECT / "experiments/q3/.gitkeep"):
        path.touch(exist_ok=True)
    json_write(final_dir / "q2_checkpoint_manifest.json", checkpoints)
    json_write(final_dir / "q2_experiment_index.json", experiments)
    config_dir = PROJECT / "configs/final"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, model in (("q2_b0_wce.yaml", "b0"), ("q2_b5_p2.yaml", "p2")):
        (config_dir / name).write_text(yaml.safe_dump(make_config(model, pkl_path, checkpoints),
                                 allow_unicode=True, sort_keys=False), encoding="utf-8")
    text_write(manifests / "attachment2_manifest.md", f"""
# Attachment 2 aligned-50 manifest

- Status: **VERIFIED**; train/valid/test = **3395/728/727**.
- Raw pickle: `{attachment2['raw_files']['aligned_50.pkl']['relative_path']}`; SHA256 `{attachment2['raw_files']['aligned_50.pkl']['sha256']}`.
- Label table: `{attachment2['raw_files']['label.xlsx']['relative_path']}`; SHA256 `{attachment2['raw_files']['label.xlsx']['sha256']}`.
- Text/audio/vision: `(N,50,768)` float32, `(N,50,74)` float64, `(N,50,35)` float64 in the raw pickle. The model data interface casts audio/vision to float32.
- Class mapping: 0 Negative, 1 Neutral, 2 Positive. XLSX uses `video_id`, `clip_id`, `label`, `annotation`, `mode`.
- Padding source: `text_bert[:,1,:] == 1`; text embeddings can be nonzero outside this mask. Native all-zero valid timesteps remain data, not missing markers.
- IDs: unique within splits, disjoint across splits, one-to-one with XLSX; labels agree.
- Local path difference: files were copied to `data/raw/`, while historical server configs name `data/raw/attachment2/`. No raw file was moved.
- Attachment 2 test was checked for structure and label integrity only; it did not enter model selection.
""")
    lines = ["# Q2 experiment index", "", "All reported robust scores come from existing result JSON; no experiment was run during locking.", "", "| Experiment | Seeds | Robust result | Conclusion | Role | Source commit |", "|---|---|---|---|---|---|"]
    for item in experiments["experiment_entries"]:
        result = item["robust_result"]
        if isinstance(result, dict) and "three_seed_mean" in result:
            display = f"{result['three_seed_mean']:.6f} ± {result['sample_std']:.6f}"
        elif isinstance(result, dict) and "seed42" in result:
            display = f"{result['seed42']:.6f}"
        elif result is None:
            display = "diagnostic only"
        else:
            display = "see JSON"
        lines.append(f"| {item['experiment_name']} | {','.join(map(str,item['training_seeds']))} | {display} | {item['conclusion']} | {item['role'] or '—'} | {(item['source_commit'] or 'unknown')[:10]} |")
    lines += ["", "`B0-WCE` is the baseline; `B5-P2` is the final model. P2 improved in two seeds and approximately tied in seed 44.", "",
              "See `q2_experiment_index.json` for source files, hypotheses, changed components and checkpoint availability."]
    text_write(final_dir / "q2_experiment_index.md", "\n".join(lines))
    text_write(final_dir / "q2_model_lock.md", f"""
# Q2 final model lock

**Status: LOCKED.** Baseline: B0-WCE. Final: B5-P2 attention residual pooling.

The input is the existing aligned-50 T/A/V features (768/74/35 dimensions), with the audited `text_bert[:,1,:]` padding mask. B0 applies masked mean, modality projection, concat fusion, and 3-class plus regression heads. P2 adds a per-modality scalar attention pool and zero-initialized gamma residual. Seed-specific B0 weights are frozen and held in eval mode; only 883 P2 parameters train. No normalization or BlockMask training augmentation is used.

| Model | 3-seed validation robust score (mean ± sample SD) |
|---|---:|
| B0-WCE | {baseline['mean']:.6f} ± {baseline['std']:.6f} |
| B5-P2 | {final['mean']:.6f} ± {final['std']:.6f} |

Paired P2−B0 robust delta: seed42 {summary['per_seed']['42']['delta_robust_P2_minus_B0']:+.6f}, seed43 {summary['per_seed']['43']['delta_robust_P2_minus_B0']:+.6f}, seed44 {summary['per_seed']['44']['delta_robust_P2_minus_B0']:+.6f}; mean {paired['mean']:+.6f} ± {paired['sample_std']:.6f}. Two seeds improved; seed44 was approximately tied. The result is seed-sensitive.

Benchmark SHA256: `{BENCHMARK_SHA256}`. Source commit before this lock: `{lock['git_commit']}`. Attachment 3 remains **SEALED**. Attachment 2 test was not used for model selection in this stage.

The actual local raw path is `data/raw/`; historical server configs use `data/raw/attachment2/`. The manifest records both without moving the raw files. The six main B0/P2 checkpoints are present and hash-indexed. P2 seed43/44 best-clean checkpoint files are referenced by existing results but are not presently in this local checkpoint directory; no existing file was deleted.
""")
    for filename, expected in original_hashes.items():
        if sha256(Path(filename)) != expected:
            raise RuntimeError(f"Input asset changed during indexing: {filename}")
    lock["safety_checks"] = {
        "raw_files_sha256_unchanged_during_lock": True,
        "original_split_unchanged": True,
        "benchmark_file_sha256_unchanged_and_not_regenerated": True,
        "six_main_checkpoint_sha256_unchanged_and_not_overwritten": True,
        "attachment3_content_not_parsed_or_deserialized": True,
        "attachment2_test_not_used_for_model_selection": True,
    }
    json_write(final_dir / "q2_model_lock.json", lock)
    print(json.dumps({"status": "LOCKED", "raw_path": pkl_path.as_posix(),
                      "benchmark_sha256": BENCHMARK_SHA256,
                      "main_checkpoint_count": len(checkpoints["checkpoints"]),
                      "attachment3_status": "SEALED"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
