"""Run the locked HEAF protocol once on Attachment4 aligned-50 (unlabeled)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import MODALITIES  # noqa: E402
from src.q3.coalitions import explain_batch  # noqa: E402
from src.q3.data_adapter import UnlabeledAligned50Dataset  # noqa: E402
from src.q3.evidence_grounding import ground_interval  # noqa: E402
from src.q3.faithfulness import deletion_curve  # noqa: E402
from src.q3.frozen_predictor import FrozenP2Predictor, sha256_file  # noqa: E402
from src.q3.text_grounding import load_pinned_tokenizer  # noqa: E402
from src.q3.temporal_occlusion import evaluate_windows  # noqa: E402


METHOD_YAML = ROOT / "configs/final/q3_heaf.yaml"
METHOD_JSON = ROOT / "outputs/q3/q3_method_lock.json"
SCHEMA_PATH = ROOT / "outputs/q3/q3_explanation_schema.json"
ATTACHMENT4_MANIFEST = ROOT / "data/manifests/q3/attachment4_inventory.json"
CHECKPOINT_MANIFEST = ROOT / "outputs/final/q2/q2_checkpoint_manifest.json"
TEXT_AUDIT = ROOT / "outputs/q3/q3_text_row_identity.json"
GROUNDING_AUDIT = ROOT / "outputs/q3/q3_av_grounding_audit.json"
CHECKPOINT_PATH = ROOT / "outputs/checkpoints/b5_pooling_p2_best_robust_score.pt"
OUTPUT_DIR = ROOT / "outputs/q3/final"
EXPERIMENT_DIR = ROOT / "experiments/q3/exp_003_attachment4_final_heaf"
EXPECTED_CHECKPOINT_SHA256 = "cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff"
NA = "NA"
CLASS_NAMES = ("Negative", "Neutral", "Positive")
PAIR_KEYS = ("text_audio", "text_vision", "audio_vision")
CSV_FIELDS = [
    "sample_id", "paired_video_file", "predicted_class", "predicted_class_name",
    "predicted_intensity", "confidence", "class_probabilities",
    "shapley_text", "shapley_audio", "shapley_vision",
    "shapley_regression_text", "shapley_regression_audio", "shapley_regression_vision",
    "primary_modality_classification", "primary_modality_regression",
    "primary_modality_agreement", "interaction_TA", "interaction_TV", "interaction_AV",
    "interaction_TA_regression", "interaction_TV_regression", "interaction_AV_regression",
    "key_interval_modality", "key_start_index", "key_end_index", "key_importance",
    "regression_shift", "temporal_importance_curve", "deletion_curve",
    "grounding_status", "text_fragment", "text_token_ids", "text_tokens",
    "text_char_start", "text_char_end", "audio_time_start", "audio_time_end",
    "video_frame_time",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def finite_or_null(values: np.ndarray, valid_length: int) -> list[float | None]:
    return [float(values[i]) if i < valid_length and math.isfinite(float(values[i])) else None
            for i in range(50)]


def load_inventory_entries() -> tuple[list[dict], list[dict]]:
    manifest = json.loads(ATTACHMENT4_MANIFEST.read_text(encoding="utf-8"))
    features, videos = [], []
    for row in manifest["files"]:
        path = Path(row["absolute_path"])
        if path.parent.name == "对齐版本" and path.suffix.lower() == ".pkl":
            features.append(row)
        elif path.parent.name == "videos" and path.parent.parent.name == "对齐版本" and path.suffix.lower() == ".mp4":
            videos.append(row)
    return sorted(features, key=lambda row: Path(row["absolute_path"]).stem), sorted(
        videos, key=lambda row: Path(row["absolute_path"]).stem)


def validate_preconditions() -> tuple[dict, list[dict], list[dict], UnlabeledAligned50Dataset, str, str]:
    cfg = yaml.safe_load(METHOD_YAML.read_text(encoding="utf-8"))
    lock = json.loads(METHOD_JSON.read_text(encoding="utf-8"))
    schema_raw = SCHEMA_PATH.read_bytes()
    schema_sha = hashlib.sha256(schema_raw).hexdigest()
    schema = json.loads(schema_raw)
    if cfg["predictor"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("locked YAML checkpoint hash mismatch")
    if lock["predictor"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("locked JSON checkpoint hash mismatch")
    if cfg["temporal"]["rho"] != 0.30 or cfg["temporal"]["stride"] != 1:
        raise RuntimeError("locked temporal parameters changed")
    if cfg["schema"]["version"] != schema["properties"]["schema_version"]["const"]:
        raise RuntimeError("schema version differs from method lock")
    if schema_sha != cfg["schema"]["sha256"]:
        raise RuntimeError("explanation schema hash differs from method lock")
    if lock["predictor"]["seed"] != 42 or lock["predictor"]["mode"] != "mean_attention":
        raise RuntimeError("predictor identity differs from method lock")
    if lock["source_validation"]["status"] != "HEAF_VALIDATION_PASSED":
        raise RuntimeError("HEAF validation is not passed")

    checkpoint_manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    entries = [row for row in checkpoint_manifest["checkpoints"]
               if row["model"] == "B5-P2" and row["seed"] == 42 and row["is_final_main_checkpoint"]]
    if len(entries) != 1 or entries[0]["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Q2 checkpoint manifest does not uniquely confirm seed42 P2")
    checkpoint_before = sha256_file(CHECKPOINT_PATH)
    if checkpoint_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA256 mismatch before inference: {checkpoint_before}")

    text_audit = json.loads(TEXT_AUDIT.read_text(encoding="utf-8"))
    grounding_audit = json.loads(GROUNDING_AUDIT.read_text(encoding="utf-8"))
    if text_audit["conclusion"] != "TEXT_FEATURE_ROW = VERIFIED" or text_audit["grounding_status"] != "PARTIAL_GROUNDING_READY":
        raise RuntimeError("verified text row grounding status is not present")
    if (grounding_audit["text"], grounding_audit["audio"], grounding_audit["vision"]) != (
            "VERIFIED", "UNVERIFIED", "UNVERIFIED"):
        raise RuntimeError("current grounding audit does not match expected per-modality statuses")

    feature_entries, video_entries = load_inventory_entries()
    if len(feature_entries) != 20 or len(video_entries) != 20:
        raise RuntimeError(f"expected 20 aligned pkls/videos, got {len(feature_entries)}/{len(video_entries)}")
    video_by_id = {Path(row["absolute_path"]).stem: row for row in video_entries}
    if len(video_by_id) != 20:
        raise RuntimeError("paired MP4 stems are not unique")
    for row in (*feature_entries, *video_entries):
        path = Path(row["absolute_path"])
        if not path.is_file() or path.stat().st_size != row["size_bytes"]:
            raise RuntimeError(f"Attachment4 file missing or size changed: {path}")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Attachment4 inventory SHA256 mismatch: {path}")

    dataset = UnlabeledAligned50Dataset([Path(row["absolute_path"]) for row in feature_entries])
    if len(dataset) != 20 or len({row["id"] for row in dataset.records}) != 20:
        raise RuntimeError("sample IDs are not unique across all 20 records")
    if {row["id"] for row in dataset.records} != set(video_by_id):
        raise RuntimeError("feature IDs do not exactly match paired MP4 stems")
    label_or_annotation_keys = []
    for record in dataset.records:
        if Path(record["source"]).stem != record["id"]:
            raise RuntimeError(f"feature filename and embedded ID differ for {record['id']}")
        if record["valid_length"] < 1 or record["valid_length"] > 50:
            raise RuntimeError(f"invalid valid-prefix length for {record['id']}")
        if any(record["features"][m].shape != (50, dim) for m, dim in
               (("text", 768), ("audio", 74), ("vision", 35))):
            raise RuntimeError(f"feature shape mismatch for {record['id']}")
        if record["text_bert"].shape != (3, 50):
            raise RuntimeError(f"text_bert shape mismatch for {record['id']}")
        if not all(np.isfinite(record["features"][m]).all() for m in MODALITIES):
            raise RuntimeError(f"nonfinite features for {record['id']}")
        label_or_annotation_keys.extend(k for k in record["keys"]
                                        if "label" in k.lower() or "annotation" in k.lower())
    if label_or_annotation_keys:
        raise RuntimeError(f"Attachment4 unexpectedly contains label/annotation fields: {sorted(set(label_or_annotation_keys))}")
    return cfg, feature_entries, video_entries, dataset, checkpoint_before, schema_sha


def make_batch(dataset: UnlabeledAligned50Dataset) -> dict[str, torch.Tensor]:
    items = [dataset[i] for i in range(len(dataset))]
    batch = {modality: torch.stack([item[modality] for item in items]) for modality in MODALITIES}
    batch["padding_mask"] = torch.stack([item["padding_mask"] for item in items])
    batch["native_zero_mask"] = torch.stack([item["native_zero_mask"] for item in items])
    return batch


def one_sample(batch: dict, index: int) -> dict:
    return {key: value[index:index + 1].clone() for key, value in batch.items()}


def distribution(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError("summary distribution is empty or nonfinite")
    return {"n": int(len(arr)), "mean": float(arr.mean()), "sd": float(arr.std(ddof=0)),
            "min": float(arr.min()), "q10": float(np.quantile(arr, 0.10)),
            "q25": float(np.quantile(arr, 0.25)), "median": float(np.median(arr)),
            "q75": float(np.quantile(arr, 0.75)), "q90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()), "positive_fraction": float(np.mean(arr > 0)),
            "negative_fraction": float(np.mean(arr < 0))}


def card_drop(values: dict[str, float]) -> dict:
    return {"class_logit": float(values["class_logit"]),
            "class_log_odds": float(values["class_margin"]),
            "confidence": float(values["confidence"]),
            "regression_signed": float(values["regression"]),
            "regression_absolute": float(abs(values["regression"]))}


def build_card(record: dict, expl: dict, row: int, window: dict,
               deletion: list[dict], video_relpath: str,
               tokenizer: Any) -> tuple[dict, dict]:
    sample_id = record["id"]
    fixed_class = int(expl["fixed_class"][row])
    class_name = CLASS_NAMES[fixed_class]
    probs_logits = np.asarray(expl["logits"][row, -1], dtype=np.float64)
    shifted = probs_logits - probs_logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    intensity = float(expl["regression"][row, -1])
    phi_class = {name: float(expl["phi_class"][row, i]) for i, name in enumerate(MODALITIES)}
    phi_reg = {name: float(expl["phi_reg"][row, i]) for i, name in enumerate(MODALITIES)}
    primary_class_i = int(expl["primary_class"][row])
    primary_reg_i = int(expl["primary_reg"][row])
    primary_class = MODALITIES[primary_class_i]
    primary_reg = MODALITIES[primary_reg_i]
    supports_class = bool(expl["supports_class"][row])
    margin_full = float(expl["class_margin"][row, -1])
    class_logit = float(probs_logits[fixed_class])

    if primary_class == "text":
        evidence = ground_interval(
            sample_id=sample_id, modality="text", start_index=window["top_start"],
            end_index=window["top_end"], valid_length=record["valid_length"],
            media_file=video_relpath, raw_text=record["raw_text"],
            text_bert=record["text_bert"], tokenizer=tokenizer)
    else:
        evidence = ground_interval(
            sample_id=sample_id, modality=primary_class,
            start_index=window["top_start"], end_index=window["top_end"],
            valid_length=record["valid_length"], media_file=video_relpath)

    raw_evidence = {
        "grounding_status": evidence["grounding_status"],
        "mapping_method": evidence["mapping_method"],
        "media_file": evidence["media_file"],
        "text_fragment": evidence["text_fragment"],
        "audio_time_range": evidence["audio_time_range"],
        "video_frame_time": evidence["video_frame_time"],
        "mapping_note": evidence["mapping_note"],
    }
    text_detail = {"token_ids": None, "tokens": None, "character_span": None}
    if primary_class == "text" and evidence["grounding_status"] == "verified":
        token_mapping = evidence["text_token_mapping"]
        special_slots = {entry["slot_index"] for entry in token_mapping["excluded_special_tokens"]}
        slots = range(window["top_start"], window["top_end"])
        kept = [j for j, slot in enumerate(slots) if slot not in special_slots
                and token_mapping["tokens"][j] not in ("[CLS]", "[SEP]", "[PAD]")]
        text_detail = {
            "token_ids": [token_mapping["token_ids"][j] for j in kept],
            "tokens": [token_mapping["tokens"][j] for j in kept],
            "character_span": token_mapping["raw_text_span"],
        }
        raw_evidence["mapping_note"] = (
            f"Verified feature row→text_bert token→pinned tokenizer offsets; "
            f"token_ids={text_detail['token_ids']}; tokens={text_detail['tokens']}; "
            f"raw_text_character_span={text_detail['character_span']} (half-open). "
            "[CLS]/[SEP] omitted from text evidence."
        )

    pair_class = {pair: {"value": float(expl["interaction_class"][pair]["value"][row]),
                         "delta_without_third": float(expl["interaction_class"][pair]["delta_without_third"][row]),
                         "delta_with_third": float(expl["interaction_class"][pair]["delta_with_third"][row])}
                  for pair in PAIR_KEYS}
    pair_reg = {pair: {"value": float(expl["interaction_reg"][pair]["value"][row]),
                       "delta_without_third": float(expl["interaction_reg"][pair]["delta_without_third"][row]),
                       "delta_with_third": float(expl["interaction_reg"][pair]["delta_with_third"][row])}
                for pair in PAIR_KEYS}

    top = window["top"]
    key = {
        "modality": primary_class,
        "start_index": int(window["top_start"]),
        "end_index": int(window["top_end"]),
        "valid_length": int(record["valid_length"]),
        "window_length": int(window["window_length"]),
        "importance": float(top["class_margin"]),
        "class_logit_drop": float(top["class_logit"]),
        "confidence_drop": float(top["confidence"]),
        "regression_shift": float(top["regression"]),
        "supports_predicted_class": bool(window["supports_predicted_class"]),
        "temporal_curve": finite_or_null(window["curve"], record["valid_length"]),
        "regression_temporal_curve": finite_or_null(window["regression_curve"], record["valid_length"]),
        "temporal_windows": [
            {"start_index": int(start), "end_index": int(start + window["window_length"]),
             "class_logit_drop": float(window["drops"]["class_logit"][i]),
             "class_log_odds_drop": float(window["drops"]["class_margin"][i]),
             "confidence_drop": float(window["drops"]["confidence"][i]),
             "regression_shift": float(window["drops"]["regression"][i])}
            for i, start in enumerate(window["starts"])
        ],
    }
    faithfulness = {
        "random_seed": 20260924,
        "random_draws": 32,
        "top_deletion_drop": card_drop(window["top"]),
        "random_deletion_drop": card_drop(window["random_mean"]),
        "deletion_curve": [
            {"fraction": float(point["fraction"]),
             "deleted_positions": int(point["deleted_positions"]),
             "top": card_drop(point["top"]),
             "random_mean": card_drop(point["random_mean"])}
            for point in deletion
        ],
    }
    card = {
        "schema_version": "0.2.0",
        "sample_id": sample_id,
        "predictor": {"model": "B5-P2", "seed": 42,
                      "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                      "feature_version": "aligned_50",
                      "baseline_kind": "all_valid_features_zero_padding_unchanged"},
        "prediction": {"predicted_class": fixed_class,
                       "predicted_class_name": class_name,
                       "predicted_intensity": intensity,
                       "confidence": float(probs[fixed_class]),
                       "class_logit": class_logit,
                       "class_log_odds": margin_full,
                       "class_probabilities": [float(value) for value in probs]},
        "modality_contribution": {
            "classification_log_odds": phi_class,
            "regression": phi_reg,
            "classification_empty_baseline": float(expl["class_margin"][row, 0]),
            "regression_empty_baseline": float(expl["regression"][row, 0]),
        },
        "primary_modality_classification": {
            "name": primary_class,
            "selection_basis": "largest_positive_class_shapley" if supports_class else "absolute_negative_fallback",
            "supports_predicted_class": supports_class,
        },
        "primary_modality_regression": {
            "name": primary_reg,
            "selection_basis": "largest_absolute_regression_shapley",
            "direction": "increases_output" if phi_reg[primary_reg] > 0 else (
                "decreases_output" if phi_reg[primary_reg] < 0 else "zero_effect"),
        },
        "primary_modality_agreement": primary_class == primary_reg,
        "pairwise_interaction": {"classification_log_odds": pair_class, "regression": pair_reg},
        "key_interval": key,
        "faithfulness": faithfulness,
        "raw_evidence": raw_evidence,
    }
    return card, text_detail


def summarize(cards: list[dict], text_details: dict[str, dict], input_check: dict,
              max_efficiency: dict[str, float], predictor: FrozenP2Predictor,
              checkpoint_before: str, schema_sha: str) -> dict:
    predictions = [row["prediction"] for row in cards]
    primary_class = [row["primary_modality_classification"]["name"] for row in cards]
    primary_reg = [row["primary_modality_regression"]["name"] for row in cards]
    class_counts = Counter(row["predicted_class_name"] for row in predictions)
    class_dist = {name: {"count": class_counts.get(name, 0),
                         "proportion": class_counts.get(name, 0) / len(cards)}
                  for name in CLASS_NAMES}

    shapley = {}
    for target, key in (("classification_log_odds", "classification_log_odds"), ("regression", "regression")):
        shapley[target] = {modality: distribution([
            row["modality_contribution"][key][modality] for row in cards])
            for modality in MODALITIES}
    interactions = {}
    for target in ("classification_log_odds", "regression"):
        interactions[target] = {pair: distribution([
            row["pairwise_interaction"][target][pair]["value"] for row in cards])
            for pair in PAIR_KEYS}
    primary_agreement = [row["primary_modality_agreement"] for row in cards]
    grounding_counts = Counter(row["raw_evidence"]["grounding_status"] for row in cards)
    text_verified = sum(row["raw_evidence"]["grounding_status"] == "verified"
                        and row["primary_modality_classification"]["name"] == "text"
                        and bool(row["raw_evidence"]["text_fragment"]) for row in cards)
    av_unverified = sum(row["primary_modality_classification"]["name"] in ("audio", "vision")
                        and row["raw_evidence"]["grounding_status"] == "unverified" for row in cards)
    faith_summary = {}
    for point_index, fraction in enumerate((0.10, 0.20, 0.30, 0.40)):
        deltas = [row["faithfulness"]["deletion_curve"][point_index]["top"]["class_log_odds"] -
                  row["faithfulness"]["deletion_curve"][point_index]["random_mean"]["class_log_odds"]
                  for row in cards]
        faith_summary[f"{fraction:.2f}"] = distribution(deltas)

    rng_candidates = []
    for row in cards:
        d10 = row["faithfulness"]["deletion_curve"][0]
        rng_candidates.append({"sample_id": row["sample_id"],
                               "top_minus_random_margin_drop": d10["top"]["class_log_odds"] - d10["random_mean"]["class_log_odds"]})
    by_id = {row["sample_id"]: row for row in cards}
    candidate_rules = {}
    text_rows = [row for row in cards if row["primary_modality_classification"]["name"] == "text"]
    if text_rows:
        pick = max(text_rows, key=lambda row: (
            row["faithfulness"]["deletion_curve"][0]["top"]["class_log_odds"] -
            row["faithfulness"]["deletion_curve"][0]["random_mean"]["class_log_odds"],
            tuple(-ord(char) for char in row["sample_id"])))
        candidate_rules["text_primary_high_faithfulness"] = {
            "sample_id": pick["sample_id"],
            "criterion": "maximum q=10% top-minus-random class-margin drop among classification text-primary samples; ID tie-break",
            "value": next(row["top_minus_random_margin_drop"] for row in rng_candidates if row["sample_id"] == pick["sample_id"]),
        }
    else:
        candidate_rules["text_primary_high_faithfulness"] = None

    for modality in ("vision", "audio"):
        eligible = [row for row in cards if row["primary_modality_classification"]["name"] == modality]
        if eligible:
            chosen = max(eligible, key=lambda row: (
                abs(row["modality_contribution"]["classification_log_odds"][modality]),
                tuple(-ord(char) for char in row["sample_id"])))
            candidate_rules[f"{modality}_primary"] = {
                "sample_id": chosen["sample_id"],
                "criterion": f"maximum absolute classification Shapley for {modality} among {modality}-primary samples; ID tie-break",
                "phi": chosen["modality_contribution"]["classification_log_odds"][modality],
            }
        else:
            candidate_rules[f"{modality}_primary"] = None

    interaction_pool = []
    for row in cards:
        for pair in PAIR_KEYS:
            interaction_pool.append((abs(row["pairwise_interaction"]["classification_log_odds"][pair]["value"]),
                                     row["sample_id"], pair,
                                     row["pairwise_interaction"]["classification_log_odds"][pair]["value"]))
    maximum = max(entry[0] for entry in interaction_pool)
    tied = sorted((entry for entry in interaction_pool if entry[0] == maximum), key=lambda item: (item[1], PAIR_KEYS.index(item[2])))
    chosen_interaction = tied[0]
    candidate_rules["strong_interaction"] = {
        "sample_id": chosen_interaction[1], "pair": chosen_interaction[2],
        "criterion": "maximum absolute classification pair-interaction value; sample ID then fixed pair order tie-break",
        "value": chosen_interaction[3],
    }
    weak = min(rng_candidates, key=lambda row: (row["top_minus_random_margin_drop"], row["sample_id"]))
    candidate_rules["weak_or_failure_boundary"] = {
        **weak,
        "primary_modality_classification": by_id[weak["sample_id"]]["primary_modality_classification"]["name"],
        "criterion": "minimum q=10% top-minus-random class-margin drop; sample ID tie-break",
    }

    checkpoint_after = sha256_file(predictor.checkpoint)
    return {
        "status": "Q3_FINAL_INFERENCE_COMPLETE",
        "scope": {"attachment": 4, "version": "aligned_50", "n_samples": len(cards),
                  "labels_present_or_used": False, "attachment2_test_used": False,
                  "attachment3_used": False, "training": False, "tuning": False,
                  "heaf_or_checkpoint_changed": False,
                  "interpretation": "model-explanation faithfulness, not prediction correctness"},
        "predictor": {"model": "B5-P2", "seed": 42, "mode": "mean_attention",
                      "checkpoint_sha256_expected": EXPECTED_CHECKPOINT_SHA256,
                      "checkpoint_sha256_before": checkpoint_before,
                      "checkpoint_sha256_after": checkpoint_after,
                      "checkpoint_unchanged": checkpoint_before == checkpoint_after,
                      "feature_normalization": "none", "input_dtype": "float32"},
        "method_lock": {"yaml": "configs/final/q3_heaf.yaml",
                        "json": "outputs/q3/q3_method_lock.json",
                        "schema_version": "0.2.0", "schema_sha256": schema_sha,
                        "rho": 0.30, "stride": 1, "coalitions": 8,
                        "random_draws": 32, "random_seed": 20260924,
                        "deletion_fractions": [0.10, 0.20, 0.30, 0.40]},
        "input_integrity": input_check,
        "qa": {"coverage": len(cards), "unique_ids": len({row["sample_id"] for row in cards}) == len(cards),
               "all_predictions_and_explanations_finite": True,
               "shapley_efficiency_max_abs_error": max_efficiency,
               "schema_validation": "all explanation cards validated against locked schema v0.2.0",
               "text_fragment_membership": "validated against exact raw_text character span for every verified text-primary explanation",
               "av_raw_media_fields_null_when_unverified": all(
                   row["raw_evidence"]["grounding_status"] != "unverified" or
                   (row["raw_evidence"]["audio_time_range"] is None and row["raw_evidence"]["video_frame_time"] is None)
                   for row in cards),
               "no_label_metrics_or_correctness_fields": True,
               "checkpoint_unchanged": checkpoint_before == checkpoint_after},
        "unlabeled_summary": {
            "predicted_class_counts_and_proportions": class_dist,
            "predicted_intensity_distribution": distribution([row["predicted_intensity"] for row in predictions]),
            "confidence_distribution": distribution([row["confidence"] for row in predictions]),
            "classification_primary_modality_counts": dict(Counter(primary_class)),
            "regression_primary_modality_counts": dict(Counter(primary_reg)),
            "classification_regression_primary_agreement": {
                "count": int(sum(primary_agreement)), "n": len(cards),
                "proportion": float(np.mean(primary_agreement))},
            "shapley_distribution": shapley,
            "pair_interaction_distribution": interactions,
            "grounding_status_counts": dict(grounding_counts),
            "verified_text_explanation_count": int(text_verified),
            "unverified_av_primary_count": int(av_unverified),
            "faithfulness_top_minus_random_class_margin_drop_by_deletion_fraction": faith_summary,
            "typical_explanation_candidates": candidate_rules,
            "summary_exclusion_note": "No accuracy, F1, MAE, Pearson, correct/incorrect, or other labeled performance metric is reported.",
        },
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg, feature_entries, video_entries, dataset, checkpoint_before, schema_sha = validate_preconditions()
    print("All 20 Attachment4 records, paired video IDs, source hashes, method/schema locks, and checkpoint hash passed preflight.", flush=True)

    device = torch.device("cpu")
    predictor = FrozenP2Predictor(CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256, seed=42, device=device)
    if predictor.model.training or any(parameter.requires_grad for parameter in predictor.model.parameters()):
        raise RuntimeError("predictor is not frozen/eval")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    tokenizer = load_pinned_tokenizer()

    dataset.records.sort(key=lambda row: row["id"])
    feature_by_id = {row["id"]: row for row in dataset.records}
    video_by_id = {Path(row["absolute_path"]).stem: row for row in video_entries}
    batch = make_batch(dataset)
    source_snapshot = {key: value.clone() for key, value in batch.items()}
    explained = explain_batch(predictor, batch)
    eff_class = float(np.max(np.abs(explained["efficiency_error_class"])))
    eff_reg = float(np.max(np.abs(explained["efficiency_error_reg"])))
    max_efficiency = {"classification": eff_class, "regression": eff_reg}
    if eff_class > 1e-5 or eff_reg > 1e-5:
        raise RuntimeError(f"Shapley efficiency failed: {max_efficiency}")

    cards: list[dict] = []
    text_details: dict[str, dict] = {}
    for index, sample_id in enumerate(sorted(feature_by_id)):
        record = feature_by_id[sample_id]
        full_logits = np.asarray(explained["logits"][index, -1], dtype=np.float64)
        full_regression = float(explained["regression"][index, -1])
        fixed_class = int(explained["fixed_class"][index])
        sample = one_sample(batch, index)
        window = evaluate_windows(
            predictor, sample, explained_modality := MODALITIES[int(explained["primary_class"][index])],
            0.30, fixed_class, full_logits, full_regression, sample_id,
            random_draws=32)
        deletion = deletion_curve(predictor, sample, window, fixed_class,
                                  full_logits, full_regression, sample_id,
                                  random_draws=32)
        video_relpath = video_by_id[sample_id]["relative_path"]
        card, text_detail = build_card(record, explained, index, window, deletion,
                                       video_relpath, tokenizer)
        errors = sorted(validator.iter_errors(card), key=lambda err: list(err.absolute_path))
        if errors:
            raise RuntimeError(f"schema validation failed for {sample_id}: {errors[0].message} at {list(errors[0].absolute_path)}")
        if not all(math.isfinite(float(value)) for value in (
                card["prediction"]["predicted_intensity"], card["prediction"]["confidence"],
                *card["prediction"]["class_probabilities"],
                *card["modality_contribution"]["classification_log_odds"].values(),
                *card["modality_contribution"]["regression"].values())):
            raise RuntimeError(f"nonfinite final card values for {sample_id}")
        if card["raw_evidence"]["grounding_status"] == "verified":
            span = text_detail["character_span"]
            if not span or record["raw_text"][span["start_char"]:span["end_char"]] != card["raw_evidence"]["text_fragment"]:
                raise RuntimeError(f"grounded text fragment does not match raw_text span for {sample_id}")
            if card["key_interval"]["modality"] != "text":
                raise RuntimeError("text evidence attached to non-text classification primary")
        else:
            if card["raw_evidence"]["audio_time_range"] is not None or card["raw_evidence"]["video_frame_time"] is not None:
                raise RuntimeError(f"unverified A/V mapping is non-null for {sample_id}")
        cards.append(card)
        text_details[sample_id] = text_detail
        print(f"HEAF complete: {sample_id}; class-primary={explained_modality}; interval=[{window['top_start']},{window['top_end']})", flush=True)

    if any(not torch.equal(batch[key], original) for key, original in source_snapshot.items()):
        raise RuntimeError("final inference mutated an input tensor")
    if len(cards) != 20 or len({row["sample_id"] for row in cards}) != 20:
        raise RuntimeError("final explanation coverage or IDs are incomplete")
    checkpoint_after = sha256_file(CHECKPOINT_PATH)
    if checkpoint_after != checkpoint_before or checkpoint_after != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash changed during inference")

    input_check = {
        "sample_count": len(dataset),
        "unique_sample_ids": len({row["id"] for row in dataset.records}),
        "paired_mp4_count": len(video_entries),
        "exact_id_pairing_count": 20,
        "all_inventory_size_sha256_match": True,
        "required_shapes": {"text": [50, 768], "audio": [50, 74], "vision": [50, 35], "text_bert": [3, 50]},
        "all_feature_values_finite": True,
        "padding_source": "text_bert[1] binary valid prefix; audio/vision suffixes validated zero by adapter",
        "label_or_annotation_fields_present": False,
        "records": [{"sample_id": row["id"], "paired_video_file": video_by_id[row["id"]]["relative_path"],
                     "valid_length": row["valid_length"], "pkl_sha256": next(
                         entry["sha256"] for entry in feature_entries if Path(entry["absolute_path"]).stem == row["id"]),
                     "mp4_sha256": video_by_id[row["id"]]["sha256"]}
                    for row in dataset.records],
    }
    summary = summarize(cards, text_details, input_check, max_efficiency,
                        predictor, checkpoint_before, schema_sha)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "attachment4_explanations.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for card in cards:
            stream.write(json.dumps(card, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = OUTPUT_DIR / "attachment4_predictions_explanations.csv"
    by_id = {row["sample_id"]: row for row in cards}
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for sample_id in sorted(by_id):
            card = by_id[sample_id]
            pred = card["prediction"]
            phi = card["modality_contribution"]["classification_log_odds"]
            phi_reg = card["modality_contribution"]["regression"]
            pairs = card["pairwise_interaction"]["classification_log_odds"]
            pairs_reg = card["pairwise_interaction"]["regression"]
            key = card["key_interval"]
            raw = card["raw_evidence"]
            detail = text_details[sample_id]
            text_verified = raw["grounding_status"] == "verified"
            if text_verified:
                start = detail["character_span"]["start_char"]
                end = detail["character_span"]["end_char"]
            else:
                # Keep explicit NA tokens in CSV; JSONL uses null for the same fields.
                start = end = None
            writer.writerow({
                "sample_id": sample_id,
                "paired_video_file": raw["media_file"] or NA,
                "predicted_class": pred["predicted_class"],
                "predicted_class_name": pred["predicted_class_name"],
                "predicted_intensity": pred["predicted_intensity"],
                "confidence": pred["confidence"],
                "class_probabilities": json.dumps(pred["class_probabilities"], separators=(",", ":")),
                "shapley_text": phi["text"], "shapley_audio": phi["audio"], "shapley_vision": phi["vision"],
                "shapley_regression_text": phi_reg["text"], "shapley_regression_audio": phi_reg["audio"], "shapley_regression_vision": phi_reg["vision"],
                "primary_modality_classification": card["primary_modality_classification"]["name"],
                "primary_modality_regression": card["primary_modality_regression"]["name"],
                "primary_modality_agreement": card["primary_modality_agreement"],
                "interaction_TA": pairs["text_audio"]["value"], "interaction_TV": pairs["text_vision"]["value"], "interaction_AV": pairs["audio_vision"]["value"],
                "interaction_TA_regression": pairs_reg["text_audio"]["value"], "interaction_TV_regression": pairs_reg["text_vision"]["value"], "interaction_AV_regression": pairs_reg["audio_vision"]["value"],
                "key_interval_modality": key["modality"], "key_start_index": key["start_index"], "key_end_index": key["end_index"], "key_importance": key["importance"],
                "regression_shift": key["regression_shift"],
                "temporal_importance_curve": json.dumps(key["temporal_curve"], ensure_ascii=False, separators=(",", ":")),
                "deletion_curve": json.dumps(card["faithfulness"]["deletion_curve"], ensure_ascii=False, separators=(",", ":")),
                "grounding_status": raw["grounding_status"],
                "text_fragment": raw["text_fragment"] if text_verified else NA,
                "text_token_ids": json.dumps(detail["token_ids"], ensure_ascii=False, separators=(",", ":")) if text_verified else NA,
                "text_tokens": json.dumps(detail["tokens"], ensure_ascii=False, separators=(",", ":")) if text_verified else NA,
                "text_char_start": start if start is not None else NA,
                "text_char_end": end if end is not None else NA,
                "audio_time_start": NA, "audio_time_end": NA, "video_frame_time": NA,
            })

    write_json(OUTPUT_DIR / "attachment4_summary.json", summary)
    qa_lines = [
        "# Attachment4 final HEAF delivery check", "",
        "**Status: `Q3_FINAL_INFERENCE_COMPLETE`.**", "",
        f"Coverage: {summary['qa']['coverage']}/20; unique IDs: {summary['qa']['unique_ids']}; exact feature↔MP4 pairs: {input_check['exact_id_pairing_count']}/20.",
        f"Checkpoint SHA256 unchanged: `{checkpoint_before}`; method schema: 0.2.0 / `{schema_sha}`.",
        f"Input shapes, finite values, masks and file inventory hashes passed. Maximum Shapley efficiency errors: class `{eff_class:.3g}`, regression `{eff_reg:.3g}`.",
        "All 20 explanation cards validated against the locked JSON Schema. Text fragments were rechecked against exact raw_text character spans. A/V grounding remains unverified; A/V time/frame fields are null in JSONL and `NA` in CSV.",
        "Attachment4 contains no labels; no accuracy, F1, MAE, Pearson, correctness, Attachment2 test, or Attachment3 result is included or used.",
        "Faithfulness values describe frozen-model intervention effects only; they do not establish prediction correctness or real-world causal effects.",
        "Typical case nominations use the fixed statistical rules recorded in `attachment4_summary.json`; no manual visual selection was used.",
        "", "## Per-modality and grounding summary", "",
        "```json", json.dumps({
            "predicted_class_counts_and_proportions": summary["unlabeled_summary"]["predicted_class_counts_and_proportions"],
            "classification_primary_modality_counts": summary["unlabeled_summary"]["classification_primary_modality_counts"],
            "regression_primary_modality_counts": summary["unlabeled_summary"]["regression_primary_modality_counts"],
            "primary_agreement": summary["unlabeled_summary"]["classification_regression_primary_agreement"],
            "grounding_status_counts": summary["unlabeled_summary"]["grounding_status_counts"],
            "verified_text_explanation_count": summary["unlabeled_summary"]["verified_text_explanation_count"],
            "unverified_av_primary_count": summary["unlabeled_summary"]["unverified_av_primary_count"],
            "representative_candidates": summary["unlabeled_summary"]["typical_explanation_candidates"],
        }, ensure_ascii=False, indent=2), "```", "",
        "## Files", "",
        "- `attachment4_predictions_explanations.csv` (CSV null rule: explicit `NA`)",
        "- `attachment4_explanations.jsonl` (JSON null rule: `null`; locked schema 0.2.0)",
        "- `attachment4_summary.json` (unlabeled distributions and deterministic representative candidates)",
        "",
    ]
    (OUTPUT_DIR / "attachment4_delivery_check.md").write_text("\n".join(qa_lines), encoding="utf-8")

    # Keep the formal experiment record reproducible and tied to the exact lock.
    write_json(EXPERIMENT_DIR / "metrics.json", summary)
    (EXPERIMENT_DIR / "config.yaml").write_text(yaml.safe_dump({
        "experiment": "Q3-3 Attachment4 final HEAF inference",
        "predictor": {"model": "B5-P2", "seed": 42, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                      "mode": "mean_attention", "dtype": "float32", "normalization": "none"},
        "data": {"source": "Attachment4", "feature_version": "aligned_50", "samples": 20,
                 "labels_used": False},
        "method_lock": "configs/final/q3_heaf.yaml",
        "rho": 0.30, "stride": 1, "random_draws": 32, "random_seed": 20260924,
        "deletion_fractions": [0.10, 0.20, 0.30, 0.40],
        "grounding": {"text": "verified", "audio": "unverified", "vision": "unverified"},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (EXPERIMENT_DIR / "notes.md").write_text(
        "# Q3-3 Attachment4 final HEAF inference\n\n"
        "Ran the frozen seed42 B5-P2 predictor and locked HEAF protocol on all 20 Attachment4 aligned-50 samples. "
        "No training, tuning, checkpoint or method changes, label metrics, Attachment2 test, or Attachment3 access. "
        "Text evidence was grounded with verified tokenizer offsets; A/V evidence remains feature-space only and raw media times are null. "
        "See `outputs/q3/final/attachment4_delivery_check.md` for QA and the other delivery artifacts.\n",
        encoding="utf-8")
    print("Q3-3 outputs written; schema, coverage, grounding, finite-value and checkpoint QA passed.", flush=True)


if __name__ == "__main__":
    main()
