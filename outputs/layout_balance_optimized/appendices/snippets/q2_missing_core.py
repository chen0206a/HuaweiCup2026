RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)
LOCATIONS = ("early", "middle", "late")


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

def scenarios() -> tuple[Scenario, ...]:
    single = [Scenario(f"{m}_r{rho:.1f}_{loc}", (m,), rho, loc)
              for m, rho, loc in itertools.product(MODALITIES, RATIOS,
                  LOCATIONS)]
    pairs = (("text", "audio"), ("text", "vision"), ("audio", "vision"))
    double = [Scenario(f"{'+'.join(pair)}_r0.3_{loc}", pair, 0.3, loc)
              for pair, loc in itertools.product(pairs, LOCATIONS)]
    result = tuple(single + double)
    assert len(result) == 54 and len({s.scenario_id for s in result}) == 54
    return result

def mask_scenario(batch: dict, scenario: Scenario) -> tuple[dict, list[dict]]:
    requests = [
        {"sample_index": i, "modality": m, "rho": scenario.rho,
         "location": scenario.location}
        for i in range(batch["padding_mask"].shape[0]) for m in
            scenario.modalities
    ]
    return apply_blocks(batch, requests)
