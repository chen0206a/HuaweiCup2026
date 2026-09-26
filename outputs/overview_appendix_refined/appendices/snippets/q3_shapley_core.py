
COALITIONS = ((), ("text",), ("audio",), ("vision",),
              ("text", "audio"), ("text", "vision"), ("audio", "vision"),
              ("text", "audio", "vision"))
INDEX = {coalition: i for i, coalition in enumerate(COALITIONS)}

PAIR_NAMES = ("text_audio", "text_vision", "audio_vision")


def exact_shapley(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("expected [B,8] coalition values")
    result = np.zeros((len(values), 3), dtype=np.float64)
    for m, modality in enumerate(MODALITIES):
        rest = [other for other in MODALITIES if other != modality]
        result[:, m] = (values[:, INDEX[(modality,)]] - values[:,
            INDEX[()]]) / 3
        for other in rest:
            pair = tuple(item for item in MODALITIES if item in (modality,
                other))
            result[:, m] += (values[:, INDEX[pair]] - values[:,
                INDEX[(other,)]]) / 6
        pair_rest = tuple(rest)
        result[:, m] += (values[:, INDEX[COALITIONS[-1]]] - values[:,
            INDEX[pair_rest]]) / 3
    return result

def pair_interactions(values: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    values = np.asarray(values, dtype=np.float64)
    result = {}
    for (i, j), name in zip(combinations(MODALITIES, 2), PAIR_NAMES):
        k = next(modality for modality in MODALITIES if modality not in (i,
            j))
        ij = tuple(item for item in MODALITIES if item in (i, j))
        ik = tuple(item for item in MODALITIES if item in (i, k))
        jk = tuple(item for item in MODALITIES if item in (j, k))
        without = values[:, INDEX[ij]] - values[:, INDEX[(i,)]] - values[:,
            INDEX[(j,)]] + values[:, INDEX[()]]
        with_third = values[:, INDEX[COALITIONS[-1]]] - values[:,
            INDEX[ik]] - values[:, INDEX[jk]] + values[:, INDEX[(k,)]]
        result[name] = {"value": (without + with_third) / 2,
                        "delta_without_third": without, "delta_with_third":
                            with_third}
    return result
