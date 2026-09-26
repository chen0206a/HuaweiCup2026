def ground_text_interval(
    raw_text: str,
    text_bert: Any,
    start_index: int,
    end_index: int,
    *,
    tokenizer: Any | None = None,
    tokenizer_json: str | Path | None = None,
) -> dict[str, Any]:
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
        "feature_row_mapping_status": FEATURE_ROW_MAPPING_STATUS,
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
            "feature_row_identity_audit": FEATURE_ROW_AUDIT,
            "scope_note": "P2 text feature row identity was verified against source-supported bert-base-uncased final-layer token rows; tokenizer IDs, masks, and type IDs are rechecked for each sample before returning offsets.",
        },
    }
