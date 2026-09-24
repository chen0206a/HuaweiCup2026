"""Token-slot grounding tests; feature-row provenance stays a separate claim."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.q3.text_grounding import ground_text_interval
from src.q3.evidence_grounding import ground_interval


class _FakeTokenizer:
    def __init__(self, ids=None):
        self.ids = ids or [101, 17, 18, 19, 102] + [0] * 45
        self.tokens = ["[CLS]", "hello", ",", "world", "[SEP]"] + ["[PAD]"] * 45
        self.offsets = [(0, 0), (0, 5), (5, 6), (7, 12), (0, 0)] + [(0, 0)] * 45
        self.attention_mask = [1] * 5 + [0] * 45
        self.type_ids = [0] * 50
        self.special_tokens_mask = [1, 0, 0, 0, 1] + [1] * 45

    def encode(self, text, add_special_tokens=True):
        if not add_special_tokens:
            raise AssertionError("special tokens are part of the verified contract")
        return SimpleNamespace(
            ids=self.ids, tokens=self.tokens, offsets=self.offsets,
            attention_mask=self.attention_mask, type_ids=self.type_ids,
            special_tokens_mask=self.special_tokens_mask,
        )


def _text_bert():
    return np.asarray([
        [101, 17, 18, 19, 102] + [0] * 45,
        [1] * 5 + [0] * 45,
        [0] * 50,
    ], dtype=np.int64)


def test_maps_tokens_to_raw_character_span_and_preserves_surface_text():
    result = ground_text_interval(
        "hello, world", _text_bert(), 1, 4, tokenizer=_FakeTokenizer()
    )
    assert result["mapping_status"] == "TOKEN_MAPPING_VERIFIED"
    assert result["feature_row_mapping_status"] == "UNVERIFIED"
    assert result["tokens"] == ["hello", ",", "world"]
    assert result["raw_text_span"] == {"start_char": 0, "end_char": 12}
    assert result["text_fragment"] == "hello, world"
    assert result["excluded_special_tokens"] == []


def test_special_tokens_are_excluded_from_surface_evidence():
    result = ground_text_interval(
        "hello, world", _text_bert(), 0, 2, tokenizer=_FakeTokenizer()
    )
    assert result["tokens"] == ["[CLS]", "hello"]
    assert result["excluded_special_tokens"] == [
        {"slot_index": 0, "token_id": 101, "token": "[CLS]"}
    ]
    assert result["text_fragment"] == "hello"


def test_mismatch_with_preprovided_ids_fails_closed():
    wrong = _FakeTokenizer(ids=[101, 999, 18, 19, 102] + [0] * 45)
    with pytest.raises(ValueError, match="token ID mismatch at slot 1"):
        ground_text_interval("hello, world", _text_bert(), 1, 4, tokenizer=wrong)


def test_padding_cannot_be_requested_as_text_evidence():
    with pytest.raises(ValueError, match="inside the valid token prefix"):
        ground_text_interval(
            "hello, world", _text_bert(), 4, 6, tokenizer=_FakeTokenizer()
        )


def test_evidence_api_keeps_p2_fragment_null_without_feature_row_provenance():
    result = ground_interval(
        sample_id="01", modality="text", start_index=1, end_index=4,
        valid_length=5, media_file=None, raw_text="hello, world",
        text_bert=_text_bert(), tokenizer=_FakeTokenizer(),
    )
    assert result["grounding_status"] == "unverified"
    assert result["text_fragment"] is None
    assert result["text_token_mapping"]["mapping_status"] == "TOKEN_MAPPING_VERIFIED"
    assert result["text_token_mapping"]["text_fragment"] == "hello, world"
    assert "not verified" in result["mapping_note"]
