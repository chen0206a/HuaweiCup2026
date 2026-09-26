"""Frozen 54-scenario validation benchmark; no test or attachment 3 access."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from src.data.block_mask import LOCATIONS, RATIOS, apply_blocks, predictor_inputs
from src.data.dataset import MODALITIES
from src.utils.metrics import compute_metrics, validation_selection_score

BENCHMARK_SEED = 20260923  # Definitions are deterministic; retained as a version identifier.


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    modalities: tuple[str, ...]
    rho: float
    location: str

    def as_dict(self) -> dict:
        return {"scenario_id": self.scenario_id, "modalities": "+".join(self.modalities),
                "rho": self.rho, "location": self.location}


def scenarios() -> tuple[Scenario, ...]:
    single = [Scenario(f"{m}_r{rho:.1f}_{loc}", (m,), rho, loc)
              for m, rho, loc in itertools.product(MODALITIES, RATIOS, LOCATIONS)]
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
        for i in range(batch["padding_mask"].shape[0]) for m in scenario.modalities
    ]
    return apply_blocks(batch, requests)


@torch.no_grad()
def evaluate_benchmark(model, valid_loader, device: torch.device,
                       *, scenario_chunk_size: int = 6) -> list[dict]:
    """Score clean and every frozen scenario on the same ordered valid samples."""
    model.eval()
    benchmark = scenarios()
    keys = ("clean",) + tuple(s.scenario_id for s in benchmark)
    collected = {key: {"cls_true": [], "cls_pred": [], "reg_true": [],
                       "reg_pred": [], "vision_zero": []} for key in keys}
    for batch in valid_loader:
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        b = moved["cls_label"].shape[0]
        truths_cls = moved["cls_label"].cpu().tolist()
        truths_reg = moved["reg_label"].cpu().tolist()
        flags = moved["vision_all_zero"].cpu().tolist()

        def collect(key: str, outputs: dict) -> None:
            target = collected[key]
            target["cls_true"].extend(truths_cls)
            target["cls_pred"].extend(outputs["classification_logits"].argmax(-1).cpu().tolist())
            target["reg_true"].extend(truths_reg)
            target["reg_pred"].extend(outputs["regression"].cpu().tolist())
            target["vision_zero"].extend(flags)

        collect("clean", model(predictor_inputs(moved)))
        for left in range(0, len(benchmark), scenario_chunk_size):
            chunk = benchmark[left:left + scenario_chunk_size]
            masked = [mask_scenario(moved, s)[0] for s in chunk]
            predictor = {key: torch.cat([part[key] for part in masked], dim=0)
                         for key in (*MODALITIES, "padding_mask")}
            outputs = model(predictor)
            for j, s in enumerate(chunk):
                sliced = {key: value[j * b:(j + 1) * b] for key, value in outputs.items()}
                collect(s.scenario_id, sliced)

    rows = []
    for scenario in (None, *benchmark):
        key = "clean" if scenario is None else scenario.scenario_id
        data = collected[key]
        metrics = compute_metrics(data["cls_true"], data["cls_pred"],
                                  data["reg_true"], data["reg_pred"])
        indices = np.flatnonzero(data["vision_zero"])
        subset = compute_metrics(
            np.asarray(data["cls_true"])[indices], np.asarray(data["cls_pred"])[indices],
            np.asarray(data["reg_true"])[indices], np.asarray(data["reg_pred"])[indices],
        ) if indices.size else None
        rows.append({"scenario_id": key,
                     "modalities": "none" if scenario is None else "+".join(scenario.modalities),
                     "rho": 0.0 if scenario is None else scenario.rho,
                     "location": "clean" if scenario is None else scenario.location,
                     "sample_count": len(data["cls_true"]),
                     "selection_score": validation_selection_score(metrics),
                     "vision_all_zero_count": int(indices.size),
                     "vision_all_zero_metrics": subset,
                     **metrics})
    clean = rows[0]
    for row in rows:
        for metric in ("accuracy", "macro_f1", "mae", "pearson", "selection_score"):
            row[f"delta_{metric}"] = row[metric] - clean[metric]
    return rows


def summary(rows: list[dict]) -> dict:
    clean, missing = rows[0], rows[1:]
    if clean["scenario_id"] != "clean" or len(missing) != 54:
        raise ValueError("expected clean + 54 missing scenarios")
    metrics = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")

    def mean(subset: list[dict]) -> dict:
        return {key: float(np.mean([row[key] for row in subset])) for key in metrics}

    return {"clean": {key: clean[key] for key in metrics},
            "mean_missing": mean(missing),
            "robust_score": 0.5 * clean["selection_score"] + 0.5 * mean(missing)["selection_score"],
            "by_ratio": {str(rho): mean([r for r in missing if r["rho"] == rho and "+" not in r["modalities"]])
                         for rho in RATIOS},
            "by_location": {loc: mean([r for r in missing if r["location"] == loc])
                            for loc in LOCATIONS},
            "by_modality": {m: mean([r for r in missing if r["modalities"] == m])
                            for m in MODALITIES},
            "double_stress": {"+".join(pair): mean([r for r in missing if r["modalities"] == "+".join(pair)])
                              for pair in (("text", "audio"), ("text", "vision"), ("audio", "vision"))}}
