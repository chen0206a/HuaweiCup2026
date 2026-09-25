"""Gate Attachment3 inference with the Q3-verified BERT feature reconstruction.

Only the 20 already-audited, unlabeled Attachment4 aligned examples provide
official text features for this numerical interface regression. No Attachment2
test split or Attachment3 prediction is used to select an adapter parameter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.pooling_residual import B5PoolingResidual
from src.q3.data_adapter import UnlabeledAligned50Dataset
from src.q3.text_feature_reconstruction import (
    MODEL_ID, REVISION, WEIGHTS_SHA256, load_pinned_bert,
    reconstruct_aligned_text,
)
from src.q3.text_grounding import TOKENIZER_JSON_SHA256
from scripts.run_q3_text_row_identity import cosine_matrix, get_aligned_paths


OUT_DIR = ROOT / "outputs/final/q2/attachment3"
CHECKPOINT = ROOT / "outputs/checkpoints/b5_pooling_p2_best_robust_score.pt"
LOCK = ROOT / "outputs/final/q2/q2_model_lock.json"
INPUT_AUDIT = OUT_DIR / "attachment3_input_audit.json"
Q3_REFERENCE = ROOT / "outputs/q3/q3_text_row_identity.json"

# Fixed before reading any reconstruction result. The feature limits are wider
# than the earlier Q3 float32 reconstruction error (3.0041e-5 / 8.3977e-7),
# while the output limit is a conservative float32 inference tolerance.
LIMITS = {
    "min_row_cosine": 0.99999,
    "max_feature_abs_diff": 5e-5,
    "max_feature_rmse": 2e-6,
    "max_model_logit_abs_diff": 1e-4,
    "max_model_regression_abs_diff": 1e-4,
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    torch.set_num_threads(1)
    input_audit = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    files = input_audit["files"]
    attachment3_schema_ok = (
        len(files) == 30 and input_audit["all_manifest_hashes_match"]
        and all(set(f["fields"]) == {"text_bert", "audio", "vision"} for f in files)
        and all(f["fields"]["text_bert"]["shape"] == [1, 3, 50]
                and f["fields"]["audio"]["shape"] == [1, 50, 74]
                and f["fields"]["vision"]["shape"] == [1, 50, 35]
                for f in files)
    )
    if not attachment3_schema_ok:
        raise ValueError("Attachment3 30-file schema/hash gate failed")

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected_checkpoint_sha = lock["checkpoint_paths"]["B5-P2:42"]["sha256"]
    checkpoint_sha_before = sha256(CHECKPOINT)
    if checkpoint_sha_before != expected_checkpoint_sha:
        raise ValueError("Q2 locked checkpoint SHA256 mismatch")
    tokenizer_file = hf_hub_download(
        MODEL_ID, "tokenizer.json", revision=REVISION, local_files_only=True)
    tokenizer_sha = sha256(tokenizer_file)
    if tokenizer_sha != TOKENIZER_JSON_SHA256:
        raise ValueError("Q3 pinned tokenizer.json SHA256 mismatch")
    weight_file = hf_hub_download(
        MODEL_ID, "model.safetensors", revision=REVISION, local_files_only=True)
    if sha256(weight_file) != WEIGHTS_SHA256:
        raise ValueError("Q3 pinned BERT weight SHA256 mismatch")

    bert = load_pinned_bert(local_files_only=True)
    paths = get_aligned_paths()
    known = UnlabeledAligned50Dataset(paths)
    if len(known) != 20:
        raise ValueError(f"Expected 20 Q3-verified known examples, got {len(known)}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("mode") != "mean_attention":
        raise ValueError("Locked checkpoint is not mean_attention")
    predictor = B5PoolingResidual("mean_attention", **checkpoint["config"]["model"])
    predictor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    predictor.eval()

    official_text, reconstructed_text, audio, vision, padding = [], [], [], [], []
    all_cosines, total_sq_error, total_elements, max_feature_abs_diff = [], 0.0, 0, 0.0
    same_index_count, valid_row_count = 0, 0
    per_sample = []
    for record in known.records:
        n = record["valid_length"]
        official = np.asarray(record["features"]["text"], dtype=np.float32)
        rebuilt = reconstruct_aligned_text(bert, record["text_bert"])
        if official.shape != (50, 768) or rebuilt.shape != (50, 768):
            raise ValueError("Known text interface shape changed")
        provided_valid = official[:n]
        rebuilt_valid = rebuilt[:n]
        cosine = cosine_matrix(provided_valid, rebuilt_valid)
        diagonal = np.diag(cosine)
        same_index = int(np.sum(np.argmax(cosine, axis=1) == np.arange(n)))
        diff = provided_valid.astype(np.float64) - rebuilt_valid.astype(np.float64)
        abs_error = float(np.max(np.abs(diff)))
        total_sq_error += float(np.square(diff).sum())
        total_elements += diff.size
        max_feature_abs_diff = max(max_feature_abs_diff, abs_error)
        all_cosines.extend(float(v) for v in diagonal)
        same_index_count += same_index
        valid_row_count += n
        per_sample.append({
            "sample_id": record["id"], "valid_rows": n,
            "min_same_index_cosine": float(diagonal.min()),
            "mean_same_index_cosine": float(diagonal.mean()),
            "same_index_argmax_rows": same_index,
            "max_feature_abs_diff": abs_error,
            "feature_rmse": float(np.sqrt(np.square(diff).mean())),
        })
        official_text.append(official)
        reconstructed_text.append(rebuilt)
        audio.append(np.asarray(record["features"]["audio"], dtype=np.float32))
        vision.append(np.asarray(record["features"]["vision"], dtype=np.float32))
        padding.append(record["padding_mask"])

    common = {
        "audio": torch.from_numpy(np.stack(audio)),
        "vision": torch.from_numpy(np.stack(vision)),
        "padding_mask": torch.from_numpy(np.stack(padding)),
    }
    with torch.no_grad():
        official_output = predictor({**common, "text": torch.from_numpy(np.stack(official_text))})
        rebuilt_output = predictor({**common, "text": torch.from_numpy(np.stack(reconstructed_text))})
    logit_diff = float((official_output["classification_logits"]
                        - rebuilt_output["classification_logits"]).abs().max().item())
    regression_diff = float((official_output["regression"]
                             - rebuilt_output["regression"]).abs().max().item())
    feature_rmse = float(np.sqrt(total_sq_error / total_elements))
    checkpoint_sha_after = sha256(CHECKPOINT)
    reference = json.loads(Q3_REFERENCE.read_text(encoding="utf-8"))
    reference_consistent = (
        reference["counts"]["valid_rows"] == valid_row_count
        and reference["counts"]["same_index_argmax_rows"] == same_index_count
        and abs(reference["aggregate"]["global_max_absolute_error"]
                - max_feature_abs_diff) <= 1e-7
        and abs(reference["aggregate"]["global_rmse_row_weighted"]
                - feature_rmse) <= 1e-8
    )
    passed = all((
        min(all_cosines) >= LIMITS["min_row_cosine"],
        same_index_count == valid_row_count,
        max_feature_abs_diff <= LIMITS["max_feature_abs_diff"],
        feature_rmse <= LIMITS["max_feature_rmse"],
        logit_diff <= LIMITS["max_model_logit_abs_diff"],
        regression_diff <= LIMITS["max_model_regression_abs_diff"],
        reference_consistent,
        checkpoint_sha_before == checkpoint_sha_after == expected_checkpoint_sha,
    ))
    result = {
        "status": "PASS" if passed else "FAIL",
        "failure_status": None if passed else "BLOCKED_BY_TEXT_INTERFACE",
        "attachment3_schema": {
            "count": 30, "top_level_keys": ["test"],
            "fields": {"text_bert": [1, 3, 50], "audio": [1, 50, 74],
                       "vision": [1, 50, 35]},
            "text_absent_count": 30, "label_field_count": 0,
            "all_manifest_hashes_match": True,
        },
        "adapter": {
            "tokenizer": MODEL_ID, "tokenizer_revision": REVISION,
            "tokenizer_json_sha256": tokenizer_sha,
            "model": MODEL_ID, "model_revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256,
            "output": "BertModel.last_hidden_state, valid prefix only, no attention_mask, no pooling",
            "padded_text_rows": "exact zeros; excluded by the unchanged text_bert[1] padding mask",
        },
        "known_data": "20 unlabeled Attachment4 aligned samples with provided text/text_bert; no labels used",
        "known_sample_count": len(known), "valid_row_count": valid_row_count,
        "same_index_argmax_rows": same_index_count,
        "mean_same_index_cosine": float(np.mean(all_cosines)),
        "min_same_index_cosine": float(np.min(all_cosines)),
        "max_feature_abs_diff": max_feature_abs_diff,
        "feature_rmse": feature_rmse,
        "max_model_logit_abs_diff": logit_diff,
        "max_model_regression_abs_diff": regression_diff,
        "q3_published_reference_consistent": reference_consistent,
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "limits_fixed_before_run": LIMITS,
        "per_sample": per_sample,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "attachment3_text_interface_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: result[k] for k in (
        "status", "known_sample_count", "valid_row_count", "same_index_argmax_rows",
        "mean_same_index_cosine", "max_feature_abs_diff", "feature_rmse",
        "max_model_logit_abs_diff", "max_model_regression_abs_diff",
        "q3_published_reference_consistent")}, indent=2))
    if not passed:
        raise RuntimeError("BLOCKED_BY_TEXT_INTERFACE: regression gate failed")


if __name__ == "__main__":
    main()
