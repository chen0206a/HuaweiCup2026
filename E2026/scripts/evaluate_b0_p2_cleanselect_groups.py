"""Read-only grouped 54-scenario evaluation of the six locked CleanSelect weights."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.missing_benchmark import evaluate_benchmark, summary  # noqa: E402
from src.models.baseline import B0Baseline  # noqa: E402
from src.models.pooling_residual import B5PoolingResidual  # noqa: E402
from run_q2_cleanselect_fair_comparison import (  # noqa: E402
    MANIFEST, chosen_checkpoints, load_validation, read_json,
)


def main() -> None:
    torch.set_num_threads(6)
    out = ROOT / "outputs/final/q2/public_baselines_expanded"
    out.mkdir(parents=True, exist_ok=True)
    valid = load_validation()
    loader = DataLoader(valid, batch_size=128, shuffle=False, num_workers=0)
    expected_path = ROOT / "outputs/final/q2/checkpoint_selection_fairness/q2_cleanselect_fair_comparison_seedwise.csv"
    with expected_path.open(encoding="utf-8-sig", newline="") as f:
        expected = {(r["model"], int(r["seed"])): r for r in csv.DictReader(f)}
    rows_out = []
    for item in chosen_checkpoints(read_json(MANIFEST)):
        state = torch.load(item["path"], map_location="cpu", weights_only=False)
        cfg = state["config"]
        model = (B0Baseline(**cfg["model"]) if item["model"] == "B0"
                 else B5PoolingResidual("mean_attention", **cfg["model"]))
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.float().eval()
        measured = evaluate_benchmark(model, loader, torch.device("cpu"), scenario_chunk_size=6)
        agg = summary(measured)
        previous = expected[(item["model"], item["seed"])]
        for condition, label in (("clean", "clean"), ("mean_missing", "mean_missing")):
            for metric in ("accuracy", "macro_f1", "mae", "pearson", "selection_score"):
                delta = abs(agg[condition][metric] - float(previous[f"{label}_{metric}"]))
                if delta > 1e-5:
                    raise RuntimeError(f"Frozen score changed: {item['model']} {item['seed']} {condition} {metric}: {delta}")
        for modality in ("text", "audio", "vision"):
            group = agg["by_modality"][modality]
            rows_out.append({"model": item["model"], "seed": item["seed"], "modality": modality,
                             "clean_accuracy": agg["clean"]["accuracy"],
                             "missing_accuracy": group["accuracy"],
                             "clean_macro_f1": agg["clean"]["macro_f1"],
                             "missing_macro_f1": group["macro_f1"],
                             "clean_mae": agg["clean"]["mae"], "missing_mae": group["mae"],
                             "clean_pearson": agg["clean"]["pearson"], "missing_pearson": group["pearson"]})
        print(f"GROUPS {item['model']} seed={item['seed']} PASS", flush=True)
    if len(rows_out) != 18:
        raise RuntimeError("Expected 2 models × 3 seeds × 3 modalities")
    path = out / "b0_p2_cleanselect_missing_groups.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    (out / "b0_p2_cleanselect_missing_groups.json").write_text(json.dumps(rows_out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
