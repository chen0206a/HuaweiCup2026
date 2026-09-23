"""Repeat frozen B0/B1/B2 training for one seed, then score fixed valid scenes.

Only training.seed and output paths differ from the four frozen base configs.
Test and attachment3 are never constructed or read. Resume skips completed
model+seed results after verifying the checkpoint's recorded training seed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src.evaluation.missing_benchmark import BENCHMARK_SEED, evaluate_benchmark, scenarios, summary
from src.training.run_b2 import data_loaders, device_auto, model_for

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
DEFINITION = METRICS / "b2_benchmark_definition.json"
DEFINITION_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
MODELS = {
    "B0-WCE": ("b0", "configs/b0_weighted_ce_score_selection.yaml", "b0"),
    "B1-WCE": ("b1", "configs/b1_weighted_ce.yaml", "b1"),
    "B2-B0-BlockMask": ("b0", "configs/b2_b0_blockmask.yaml", "b2_b0"),
    "B2-B1-BlockMask": ("b1", "configs/b2_b1_blockmask.yaml", "b2_b1"),
}


def assert_frozen_benchmark() -> None:
    digest = hashlib.sha256(DEFINITION.read_bytes()).hexdigest()
    if digest != DEFINITION_SHA256:
        raise RuntimeError(f"frozen benchmark changed: {digest}")
    definition = json.loads(DEFINITION.read_text(encoding="utf-8"))
    if (definition["seed"] != BENCHMARK_SEED or
            definition["scenarios"] != [scene.as_dict() for scene in scenarios()]):
        raise RuntimeError("unexpected frozen benchmark seed or scenario count")


def config_for(name: str, seed: int) -> tuple[Path, dict, Path, Path]:
    _, relative_base, short = MODELS[name]
    base = yaml.safe_load((ROOT / relative_base).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    cfg["training"]["seed"] = seed
    tag = f"b21_{short}_seed_{seed}"
    if short == "b0":
        cfg["training"]["evaluate_test"] = False
        cfg["outputs"]["metrics_filename"] = f"{tag}_training_metrics.json"
        cfg["outputs"]["history_filename"] = f"{tag}_history.json"
        cfg["outputs"]["checkpoint_name"] = f"{tag}_best_valid_loss.pt"
        cfg["outputs"]["score_checkpoint_name"] = f"{tag}_best_selection_score.pt"
        selected = CHECKPOINTS / cfg["outputs"]["score_checkpoint_name"]
    elif short == "b1":
        cfg["outputs"]["metrics_filename"] = f"{tag}_training_metrics.json"
        cfg["outputs"]["history_filename"] = f"{tag}_history.json"
        cfg["outputs"]["best_loss_checkpoint"] = f"{tag}_best_valid_loss.pt"
        cfg["outputs"]["best_score_checkpoint"] = f"{tag}_best_selection_score.pt"
        selected = CHECKPOINTS / cfg["outputs"]["best_score_checkpoint"]
    else:
        cfg["outputs"]["metrics_filename"] = f"{tag}_training_metrics.json"
        cfg["outputs"]["history_filename"] = f"{tag}_history.json"
        cfg["outputs"]["best_clean_checkpoint"] = f"{tag}_best_clean_score.pt"
        cfg["outputs"]["best_robust_checkpoint"] = f"{tag}_best_robust_score.pt"
        selected = CHECKPOINTS / cfg["outputs"]["best_robust_checkpoint"]
    # Compare every non-seed, non-output item with the frozen config.
    reconstructed = copy.deepcopy(cfg)
    reconstructed["training"]["seed"] = base["training"]["seed"]
    reconstructed["outputs"] = base["outputs"]
    if reconstructed != base:
        raise RuntimeError(f"unapproved config change in {name}")
    config_dir = ROOT / "configs" / "b21"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{tag}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path, cfg, selected, METRICS / cfg["outputs"]["metrics_filename"]


def train_and_evaluate(name: str, seed: int) -> dict:
    assert_frozen_benchmark()
    backbone, _, short = MODELS[name]
    config_path, cfg, checkpoint_path, training_metrics_path = config_for(name, seed)
    result_path = METRICS / f"b21_{short}_seed_{seed}_validation.json"
    if result_path.exists() and checkpoint_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if cached["seed"] == seed and int(state["config"]["training"]["seed"]) == seed:
            print(f"reusing completed {name} seed {seed}", flush=True)
            return cached
    module = "src.training.run_b2" if name.startswith("B2-") else (
        "src.training.train" if name == "B0-WCE" else "src.training.train_b1")
    command = [sys.executable, "-m", module]
    if name.startswith("B2-"):
        command.append("train")
    command += ["--config", str(config_path)]
    log_path = METRICS / f"b21_{short}_seed_{seed}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "4"},
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert_frozen_benchmark()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(state["config"]["training"]["seed"]) != seed:
        raise RuntimeError("checkpoint training seed mismatch")
    _, loaders, _ = data_loaders(cfg["data"]["pkl_path"],
                                 int(cfg["data"]["batch_size"]), seed)
    device = device_auto()
    model = model_for(backbone, cfg["model"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    rows = evaluate_benchmark(model, loaders["valid"], device)
    metrics = summary(rows)
    training = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    epoch = (int(state["epoch"]) if name.startswith("B2-")
             else int(state["best_epoch"] if name == "B0-WCE" else state["best_score_epoch"]))
    result = {"model": name, "seed": seed, "benchmark_seed": BENCHMARK_SEED,
              "benchmark_sha256": DEFINITION_SHA256, "checkpoint": str(checkpoint_path),
              "best_epoch": epoch, "validation": metrics,
              "training_seconds": training["training_seconds"]}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": name, "seed": seed, "best_epoch": epoch,
                      "clean_score": metrics["clean"]["selection_score"],
                      "missing_score": metrics["mean_missing"]["selection_score"],
                      "robust_score": metrics["robust_score"]}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", choices=tuple(MODELS), help="run one model; default all four")
    args = parser.parse_args()
    for name in ((args.model,) if args.model else MODELS):
        train_and_evaluate(name, args.seed)


if __name__ == "__main__":
    main()
