def align_to_bins_hard(features: list[TimedFeature], bins: list[dict],
    duration: float) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    if not features:
        raise ValueError("No encoder features supplied")
    dimension = len(features[0].vector)
    for feature in features:
        feature.validate(duration)
        if len(feature.vector) != dimension:
            raise ValueError("Inconsistent feature dimensions")
    values = np.zeros((len(bins), dimension), dtype=np.float32)
    valid = np.zeros(len(bins), dtype=bool)
    traces: list[list[dict]] = []
    for i, bin_info in enumerate(bins):
        start, end = float(bin_info["start"]), float(bin_info["end"])
        if bin_info["index"] != i or not 0 <= start < end <= duration + 1e-6:
            raise ValueError("Invalid unified time bin")
        center = (start + end) / 2
        overlaps = [(j, max(0.0, min(end, f.end) - max(start, f.start))) for
            j, f in enumerate(features)]
        overlaps = [(j, weight) for j, weight in overlaps if weight > 0]
        if not overlaps:
            traces.append([])
            continue
        chosen, weight = min(overlaps, key=lambda pair: (-pair[1],
            abs((features[pair[0]].start + features[pair[0]].end) / 2 -
            center), pair[0]))
        values[i] = features[chosen].vector
        valid[i] = True
        traces.append([{"feature_index": chosen, "overlap_seconds": weight,
                        "source_interval": [features[chosen].start,
                            features[chosen].end],
                        "source": features[chosen].source, "assignment":
                            "maximum_overlap_hard"}])
    return values, valid, traces
