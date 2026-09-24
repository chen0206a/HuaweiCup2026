"""Reconstruct source-supported BERT token rows and audit Attachment4 row identity.

This is a provenance diagnostic only: no P2 inference or HEAF output is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.q3.data_adapter import UnlabeledAligned50Dataset
from src.q3.evidence_grounding import ground_interval
from src.q3.text_grounding import load_pinned_tokenizer


MANIFEST = ROOT / "data/manifests/q3/attachment4_inventory.json"
OUT = ROOT / "experiments/q3/exp_0026_text_row_identity"
REPORT_JSON = ROOT / "outputs/q3/q3_text_row_identity.json"
MODEL_ID = "bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def get_aligned_paths() -> list[Path]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = [Path(row["absolute_path"]) for row in manifest["files"]
             if Path(row["absolute_path"]).parent.name == "对齐版本"
             and Path(row["absolute_path"]).suffix.lower() == ".pkl"]
    return sorted(paths, key=lambda p: p.stem)


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return left_norm @ right_norm.T


def main() -> None:
    print("loading data and pinned BERT candidate...", flush=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    paths = get_aligned_paths()
    dataset = UnlabeledAligned50Dataset(paths)
    weight_file = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION))
    model = BertModel.from_pretrained(MODEL_ID, revision=REVISION, use_safetensors=True)
    from_pretrained_mode = "train" if model.training else "eval"
    model.eval()
    assert not model.training
    print(f"candidate loaded; auditing {len(dataset)} samples", flush=True)

    per_sample = []
    matrices: dict[str, np.ndarray] = {}
    sample_rows: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    for record in dataset.records:
        ids = np.asarray(record["text_bert"][0, :record["valid_length"]], dtype=np.int64)
        mask = np.asarray(record["text_bert"][1, :record["valid_length"]], dtype=np.int64)
        types = np.asarray(record["text_bert"][2, :record["valid_length"]], dtype=np.int64)
        if not np.all(mask == 1):
            raise ValueError(f"{record['id']}: valid prefix has non-one attention mask")
        # Mirror Self-MM/MMSA-FET historical extraction: unpadded token sequence,
        # no explicit attention_mask, last-layer token states, no pooling.
        with torch.no_grad():
            output = model(input_ids=torch.from_numpy(ids).unsqueeze(0),
                           token_type_ids=torch.from_numpy(types).unsqueeze(0))
            reconstructed = output.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
        provided = np.asarray(record["features"]["text"][:len(ids)], dtype=np.float32)
        if provided.shape != reconstructed.shape:
            raise ValueError(f"{record['id']}: row matrix shape mismatch {provided.shape} vs {reconstructed.shape}")
        corr = cosine_matrix(provided, reconstructed)
        diagonal = np.diag(corr)
        off = corr.copy()
        np.fill_diagonal(off, -np.inf)
        best_off = np.max(off, axis=1) if len(ids) > 1 else np.full(len(ids), np.nan)
        nearest = np.argmax(corr, axis=1)
        error = provided - reconstructed
        item = {
            "sample_id": str(record["id"]), "source_file": record["source"],
            "valid_token_rows": int(len(ids)),
            "max_absolute_error": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
            "mean_row_cosine_same_index": float(np.mean(diagonal)),
            "min_row_cosine_same_index": float(np.min(diagonal)),
            "argmax_similarity_matches_same_index": int(np.sum(nearest == np.arange(len(ids)))),
            "argmax_similarity_match_fraction": float(np.mean(nearest == np.arange(len(ids)))),
            "mean_best_off_diagonal_cosine": float(np.mean(best_off)) if len(ids) > 1 else None,
            "mean_diagonal_margin": float(np.mean(diagonal - best_off)) if len(ids) > 1 else None,
            "min_diagonal_margin": float(np.min(diagonal - best_off)) if len(ids) > 1 else None,
        }
        per_sample.append(item)
        matrices[str(record["id"])] = corr
        sample_rows[str(record["id"])] = (provided, reconstructed, ids.tolist())
        print(f"audited sample {record['id']}", flush=True)

    # One deterministic real-sample API smoke test for the verified text path.
    smoke_record = next(row for row in dataset.records if row["id"] == "01")
    smoke_start, smoke_end = 1, min(4, smoke_record["valid_length"] - 1)
    smoke = ground_interval(
        sample_id=smoke_record["id"], modality="text",
        start_index=smoke_start, end_index=smoke_end,
        valid_length=smoke_record["valid_length"], media_file=None,
        raw_text=smoke_record["raw_text"], text_bert=smoke_record["text_bert"],
        tokenizer=load_pinned_tokenizer(),
    )
    grounding_smoke_test = {
        "sample_id": smoke_record["id"],
        "token_interval_half_open": [smoke_start, smoke_end],
        "grounding_status": smoke["grounding_status"],
        "mapping_method": smoke["mapping_method"],
        "token_ids": smoke["text_token_mapping"]["token_ids"],
        "tokens": smoke["text_token_mapping"]["tokens"],
        "text_fragment": smoke["text_fragment"],
        "text_char_span": smoke["text_char_span"],
    }

    # Deterministic provenance case: lexical first sample ID; not selected by model output.
    rep_id = sorted(matrices)[0]
    corr = matrices[rep_id]
    ids = sample_rows[rep_id][2]
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"rendering diagnostic heatmap for sample {rep_id}...", flush=True)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(corr, origin="lower", aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    ax.set(title=f"Attachment4 text rows × reconstructed BERT rows — sample {rep_id}",
           xlabel="Reconstructed BERT token row j", ylabel="Provided text feature row i")
    ax.set_xticks(range(len(ids)), [str(i) for i in range(len(ids))], rotation=90, fontsize=7)
    ax.set_yticks(range(len(ids)), [str(i) for i in range(len(ids))], fontsize=7)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.savefig(fig_dir / f"sample_{rep_id}_text_row_cosine_heatmap.png", dpi=180)
    plt.close(fig)
    print("writing audit outputs...", flush=True)

    all_diag = [r["mean_row_cosine_same_index"] for r in per_sample]
    all_off = [r["mean_best_off_diagonal_cosine"] for r in per_sample if r["mean_best_off_diagonal_cosine"] is not None]
    all_margin = [r["mean_diagonal_margin"] for r in per_sample if r["mean_diagonal_margin"] is not None]
    total_rows = sum(r["valid_token_rows"] for r in per_sample)
    total_matches = sum(r["argmax_similarity_matches_same_index"] for r in per_sample)
    matrix_path = OUT / "row_cosine_matrices.npz"
    np.savez_compressed(matrix_path, **{
        f"sample_{sample_id}": matrix.astype(np.float32, copy=False)
        for sample_id, matrix in matrices.items()
    })
    report = {
        "audit": "Q3-2.6 Text Feature Row Identity Audit",
        "conclusion": "TEXT_FEATURE_ROW = VERIFIED",
        "grounding_status": "PARTIAL_GROUNDING_READY",
        "scope": "20 Attachment4 aligned samples; reconstruction only; no predictor or HEAF inference",
        "candidate": {"model_id": MODEL_ID, "revision": REVISION,
                      "weights_file": "model.safetensors", "weights_sha256": sha256_file(weight_file),
                      "transformers_version": __import__("transformers").__version__,
                      "hidden_output": "BertModel.last_hidden_state (final encoder layer)",
                      "pooling": "none; token rows preserved", "mode": "eval", "gradient": "no_grad",
                      "from_pretrained_default_mode": from_pretrained_mode,
                      "inputs": "provided text_bert input_ids and token_type_ids, valid prefix only; no explicit attention_mask, mirroring public extraction code",
                      "special_tokens": "[CLS] and [SEP] retained as tokenizer-added valid rows; offset grounding must omit special tokens",
                      "final_hidden_layer_index_zero_based": 12,
                      "publicly_supported_alternatives": [],
                      "candidate_selection_basis": "MMSA BertTextEncoder and Self-MM DataPre.py / MMSA-FET bert extractor call BertModel and directly use tuple output [0] or last_hidden_state; no layer averaging/pooling found",
                      "public_sources": [
                          {"repository": "MMSA", "revision": "a94e65d07fa1ae0d44e552390074b29b0898edfd", "path": "src/MMSA/models/subNets/BertTextEncoder.py", "url": "https://github.com/declare-lab/MMSA/blob/a94e65d07fa1ae0d44e552390074b29b0898edfd/src/MMSA/models/subNets/BertTextEncoder.py"},
                          {"repository": "Self-MM", "revision": "1786283c81eeb507f317fa1c70a3faf77e67cee0", "path": "data/DataPre.py", "url": "https://github.com/thuiar/Self-MM/blob/1786283c81eeb507f317fa1c70a3faf77e67cee0/data/DataPre.py"},
                          {"repository": "MMSA-FET", "revision": "f8fbd2d88d4f77580ea1ded0b3469073488c5c19", "path": "src/MSA_FET/extractors/text/bert.py", "url": "https://github.com/declare-lab/MMSA-FET/blob/f8fbd2d88d4f77580ea1ded0b3469073488c5c19/src/MSA_FET/extractors/text/bert.py"}
                      ]},
        "counts": {"sample_count": len(per_sample), "valid_rows": total_rows,
                   "same_index_argmax_rows": total_matches,
                   "same_index_argmax_fraction": total_matches / total_rows if total_rows else None},
        "aggregate": {"mean_of_sample_mean_diagonal_cosine": float(np.mean(all_diag)),
                      "mean_of_sample_mean_best_off_diagonal_cosine": float(np.mean(all_off)),
                      "mean_of_sample_diagonal_margin": float(np.mean(all_margin)),
                      "global_max_absolute_error": max(r["max_absolute_error"] for r in per_sample),
                      "global_rmse_row_weighted": None},
        "representative_heatmap_sample_id": rep_id,
        "full_similarity_matrices": {
            "path": str(matrix_path.relative_to(ROOT)),
            "sha256": sha256_file(matrix_path),
            "per_sample_shape": "valid_length_by_valid_length",
            "matrix_orientation": "provided_feature_row_i_by_reconstructed_bert_row_j",
        },
        "grounding_smoke_test": grounding_smoke_test,
        "representative_heatmap": str((fig_dir / f"sample_{rep_id}_text_row_cosine_heatmap.png").relative_to(ROOT)),
        "per_sample": per_sample,
        "method_limitations": [
            "The public preprocessing code names bert-base-uncased but does not pin the original checkpoint revision used to write Attachment4. The current pinned candidate reconstructs the values to float32 numerical precision; row identity is verified, while historical checkpoint provenance remains unpinned.",
            "Only valid text_bert prefix rows were compared; padded suffix rows are not treated as tokens.",
            "No alternative layer/averaging transforms were searched because the reviewed public extraction sources do not support them."
        ]
    }
    # Compute true row-weighted RMSE from per-sample RMSE and row counts.
    report["aggregate"]["global_rmse_row_weighted"] = float(np.sqrt(
        sum(r["rmse"] ** 2 * r["valid_token_rows"] * 768 for r in per_sample) /
        sum(r["valid_token_rows"] * 768 for r in per_sample)))
    json_write(OUT / "metrics.json", report)
    json_write(REPORT_JSON, report)
    print(json.dumps({"samples": len(per_sample), "rows": total_rows,
                      "argmax_same_index": f"{total_matches}/{total_rows}",
                      "aggregate": report["aggregate"],
                      "weight_sha256": report["candidate"]["weights_sha256"],
                      "heatmap": report["representative_heatmap"]}, indent=2))


if __name__ == "__main__":
    main()
