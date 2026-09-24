"""Evaluate both saved B5-P checkpoint rules on the unchanged validation benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, summary
from src.models.baseline import B0Baseline
from src.models.pooling_residual import B5PoolingResidual
from src.training.run_b5_pooling import BENCHMARK_SHA256, CHECKPOINTS, METRICS, verify_config, write_json


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loaders, _ = build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]), num_workers=0,
        seed=42, include_test=False)
    output = {"benchmark_sha256": BENCHMARK_SHA256, "training_seed": 42,
              "checkpoint_rules": {}}
    source = torch.load(cfg["training"]["init_checkpoint"], map_location="cpu", weights_only=False)
    p0 = B0Baseline(**cfg["model"]).to(device).eval()
    p0.load_state_dict(source["model_state_dict"])
    output["checkpoint_rules"]["P0"] = {
        "best_clean": {"epoch": int(source["best_epoch"]),
                       "validation": summary(evaluate_benchmark(p0, loaders["valid"], device))},
        "best_robust": None,
    }
    for label, mode in (("P1", "mean_max"), ("P2", "mean_attention")):
        output["checkpoint_rules"][label] = {}
        for rule in ("best_clean", "best_robust"):
            path = CHECKPOINTS / f"{cfg['outputs']['checkpoint_prefix']}_{label.lower()}_{rule}_score.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint["benchmark_sha256"] != BENCHMARK_SHA256 or checkpoint["mode"] != mode:
                raise RuntimeError(f"{label} checkpoint metadata mismatch")
            model = B5PoolingResidual(mode, **cfg["model"]).to(device).eval()
            model.load_state_dict(checkpoint["model_state_dict"])
            output["checkpoint_rules"][label][rule] = {
                "epoch": int(checkpoint["epoch"]),
                "validation": summary(evaluate_benchmark(model, loaders["valid"], device)),
            }
    write_json(METRICS / "b5_pooling_checkpoint_comparison.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/b5_pooling.yaml"))
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({label: {rule: (None if value is None else value["validation"]["robust_score"])
                              for rule, value in rules.items()}
                      for label, rules in result["checkpoint_rules"].items()}))


if __name__ == "__main__":
    main()
