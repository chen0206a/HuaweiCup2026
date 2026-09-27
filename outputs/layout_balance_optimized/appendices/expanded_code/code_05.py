def block_length(valid_length: int, rho: float) -> int:
    if valid_length < 1 or not 0 < rho <= 1:
        raise ValueError("valid_length must be positive and rho in (0, 1]")
    return min(valid_length, max(1, round(rho * valid_length)))

def block_start(valid_length: int, length: int, location: str) -> int:
    remaining = valid_length - length
    if remaining < 0:
        raise ValueError("block exceeds valid sequence")
    if location == "early":
        return 0
    if location == "middle":
        return remaining // 2
    if location == "late":
        return remaining
    raise ValueError(f"unknown location: {location}")

def apply_blocks(batch: dict, requests: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    pad = batch["padding_mask"]
    if pad.ndim != 2 or pad.dtype != torch.bool:
        raise ValueError("padding_mask must be bool [B,T]")
    b, t = pad.shape
    if not pad.any(dim=1).all() or (pad[:, 1:] & ~pad[:, :-1]).any():
        raise ValueError("padding_mask must be a nonempty true prefix")
    out = dict(batch)
    for modality in MODALITIES:
        if batch[modality].shape[:2] != (b, t):
            raise ValueError(f"{modality} shape does not match padding")
        out[modality] = batch[modality].clone()
    out["availability_mask"] = batch["availability_mask"].clone()
    if out["availability_mask"].shape != (b, len(MODALITIES), t):
        raise ValueError("availability_mask must be [B,3,T]")
    if not out["availability_mask"].all():
        raise ValueError("apply_blocks expects complete original availability")
    seen = set()
    metadata = []
    for request in requests:
        i, modality, rho = int(request["sample_index"]), request["modality"], float(request["rho"])
        if not 0 <= i < b or modality not in MODALITIES or (i, modality) in seen:
            raise ValueError("invalid or duplicate sample/modality request")
        seen.add((i, modality))
        valid_length = int(pad[i].sum().item())
        length = block_length(valid_length, rho)
        if "start" in request:
            start = int(request["start"])
            location = request.get("location", "random")
        else:
            location = request["location"]
            start = block_start(valid_length, length, location)
        end = start + length
        if start < 0 or end > valid_length:
            raise ValueError("missing block would overlap padding")
        out[modality][i, start:end, :] = 0
        out["availability_mask"][i, MODALITIES.index(modality), start:end] = False
        metadata.append({
            "sample_id": str(batch["id"][i]), "modality": modality,
            "valid_length": valid_length, "start": start, "end": end,
            "block_length": length, "missing_ratio": rho,
            "actual_missing_ratio": length / valid_length, "location": location,
        })
    return out, metadata
