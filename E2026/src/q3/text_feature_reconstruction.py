"""Exact token-row reconstruction used in the Q3 text identity audit.

The historical extractor encodes only the valid (unpadded) token prefix,
passes token types without an explicit attention mask, and returns BERT's
last_hidden_state without pooling. The model revision and weight hash are the
ones already verified by scripts/run_q3_text_row_identity.py.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import BertModel


MODEL_ID = "bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
WEIGHTS_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned_bert(*, local_files_only: bool = True) -> BertModel:
    """Load the previously audited candidate, rejecting a weight mismatch."""
    weight_path = hf_hub_download(
        MODEL_ID, "model.safetensors", revision=REVISION,
        local_files_only=local_files_only,
    )
    actual = _sha256(weight_path)
    if actual != WEIGHTS_SHA256:
        raise ValueError(f"BERT weight SHA256 mismatch: {actual}")
    model = BertModel.from_pretrained(
        MODEL_ID, revision=REVISION, use_safetensors=True,
        local_files_only=local_files_only,
    )
    model.eval()
    return model


def reconstruct_valid_text_rows(
    model: BertModel, text_bert: np.ndarray, valid_length: int | None = None,
) -> np.ndarray:
    """Return float32 [valid_length,768] rows using the Q3-verified path."""
    bert = np.asarray(text_bert)
    if bert.shape != (3, 50) or not np.isfinite(bert).all():
        raise ValueError("text_bert must be finite [3,50]")
    if not np.all(bert == np.rint(bert)):
        raise ValueError("text_bert token/mask/type values must be integral")
    mask = bert[1]
    if not np.isin(mask, (0, 1)).all() or not np.all(np.diff(mask) <= 0):
        raise ValueError("text_bert[1] must be a binary valid prefix")
    observed_length = int(mask.sum())
    if observed_length < 1 or (valid_length is not None and observed_length != valid_length):
        raise ValueError("invalid or mismatched text_bert valid_length")
    length = observed_length
    ids = np.asarray(bert[0, :length], dtype=np.int64)
    types = np.asarray(bert[2, :length], dtype=np.int64)
    # This is the same unpadded forward used in Q3-2.6. Do not pass an
    # attention_mask, average layers, pool, or re-encode from raw_text.
    with torch.no_grad():
        output = model(
            input_ids=torch.from_numpy(ids).unsqueeze(0),
            token_type_ids=torch.from_numpy(types).unsqueeze(0),
        )
        rows = output.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)
    if rows.shape != (length, 768) or not np.isfinite(rows).all():
        raise ValueError("unexpected reconstructed BERT rows")
    return rows


def reconstruct_aligned_text(model: BertModel, text_bert: np.ndarray) -> np.ndarray:
    """Pad verified valid rows with exact zeros for the Q2 [50,768] input."""
    valid = reconstruct_valid_text_rows(model, text_bert)
    full = np.zeros((50, 768), dtype=np.float32)
    full[:len(valid)] = valid
    return full
