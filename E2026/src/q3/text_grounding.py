"""Verify BERT token slots against raw text without asserting feature-row provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


TOKENIZER_REPO = "bert-base-uncased"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
TOKENIZER_JSON_SHA256 = "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"
MAX_LENGTH = 50


def load_pinned_tokenizer(tokenizer_json: str | Path | None = None):
    """Load the pinned fast tokenizer and configure the observed 50-slot policy.

    `tokenizers` and `huggingface_hub` are small runtime dependencies. The
    tokenizer artifact is downloaded only when no local tokenizer.json is given.
    """
    if tokenizer_json is None:
        from huggingface_hub import hf_hub_download

        tokenizer_json = hf_hub_download(
            repo_id=TOKENIZER_REPO,
            filename="tokenizer.json",
            revision=TOKENIZER_REVISION,
        )
    tokenizer_path = Path(tokenizer_json)
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if digest != TOKENIZER_JSON_SHA256:
        raise ValueError(f"tokenizer.json SHA256 mismatch: {digest}")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_LENGTH, direction="right")
    tokenizer.enable_padding(
        length=MAX_LENGTH, pad_id=0, pad_type_id=0, pad_token="[PAD]"
    )
    return tokenizer


def ground_text_interval(
    raw_text: str,
    text_bert: Any,
    start_index: int,
    end_index: int,
    *,
    tokenizer: Any | None = None,
    tokenizer_json: str | Path | None = None,
) -> dict[str, Any]:
    """Map a half-open `text_bert` token interval to raw-text offsets.

    Exact token ID, attention mask, and token-type equality are required before
    returning a verified token-to-text mapping. This does not certify that the
    P2 `text` feature row at the same index was generated from that token; that
    separate provenance check remains explicit in the result.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    table = np.asarray(text_bert)
    if table.shape != (3, MAX_LENGTH) or not np.issubdtype(table.dtype, np.integer):
        raise ValueError("text_bert must be integer [3,50]")
    ids, mask, type_ids = table
    if not np.isin(mask, (0, 1)).all() or not np.all(np.diff(mask) <= 0):
        raise ValueError("text_bert attention mask must be a binary true prefix")
    valid_length = int(mask.sum())
    if valid_length < 2 or not (0 <= start_index < end_index <= valid_length):
        raise ValueError("interval must be nonempty and inside the valid token prefix")

    if tokenizer is None:
        tokenizer = load_pinned_tokenizer(tokenizer_json)
    encoding = tokenizer.encode(raw_text, add_special_tokens=True)
    if len(encoding.ids) != MAX_LENGTH:
        raise ValueError("configured tokenizer did not return exactly 50 IDs")
    if not np.array_equal(np.asarray(encoding.ids), ids):
        mismatch = next(
            i for i, (actual, expected) in enumerate(zip(encoding.ids, ids))
            if int(actual) != int(expected)
        )
        raise ValueError(f"token ID mismatch at slot {mismatch}")
    if not np.array_equal(np.asarray(encoding.attention_mask), mask):
        raise ValueError("tokenizer attention mask differs from text_bert")
    if not np.array_equal(np.asarray(encoding.type_ids), type_ids):
        raise ValueError("tokenizer token_type_ids differ from text_bert")
    if int(ids[0]) != 101 or int(ids[valid_length - 1]) != 102:
        raise ValueError("expected [CLS] and [SEP] at the valid-prefix boundaries")

    special_mask = list(encoding.special_tokens_mask)
    interval_slots = range(start_index, end_index)
    excluded = [
        {"slot_index": i, "token_id": int(encoding.ids[i]), "token": encoding.tokens[i]}
        for i in interval_slots if special_mask[i]
    ]
    text_slots = [i for i in interval_slots if not special_mask[i]]
    text_slots = [i for i in text_slots if encoding.offsets[i][1] > encoding.offsets[i][0]]
    if text_slots:
        char_start = min(int(encoding.offsets[i][0]) for i in text_slots)
        char_end = max(int(encoding.offsets[i][1]) for i in text_slots)
        span = {"start_char": char_start, "end_char": char_end}
        fragment = raw_text[char_start:char_end]
    else:
        span, fragment = None, None

    token_slots = list(interval_slots)
    return {
        "token_ids": [int(encoding.ids[i]) for i in token_slots],
        "tokens": [encoding.tokens[i] for i in token_slots],
        "raw_text_span": span,
        "text_fragment": fragment,
        "excluded_special_tokens": excluded,
        "mapping_status": "TOKEN_MAPPING_VERIFIED",
        "feature_row_mapping_status": "UNVERIFIED",
        "provenance": {
            "tokenizer": TOKENIZER_REPO,
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "settings": {
                "fast_tokenizer": True,
                "lowercase": True,
                "add_special_tokens": True,
                "max_length": MAX_LENGTH,
                "truncation": "right",
                "padding": "right_to_50_with_id_0",
            },
            "verified_arrays": ["input_ids", "attention_mask", "token_type_ids"],
            "scope_note": "Proves text_bert token slot to raw_text span; does not prove P2 text feature row identity.",
        },
    }
