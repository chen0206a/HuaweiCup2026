"""B3.2: frozen B0 reconstruction stability runs for training seeds 43/44."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from src.training import run_b31

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BASE_CONFIG = ROOT / "configs" / "b31_frozen_reconstruction.yaml"


def make_seed_config(seed: int) -> Path:
    if seed not in (43, 44):
        raise ValueError("B3.2 only trains missing seeds 43 and 44; seed 42 is reused")
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    cfg["run_name"] = f"B3.2-Frozen-B0-Conservative-Reconstruction-seed{seed}"
    cfg["training"]["seed"] = seed
    cfg["training"]["init_checkpoint"] = (
        f"/root/workspace/E2026/outputs/checkpoints/"
        f"b21_b0_seed_{seed}_best_selection_score.pt"
    )
    cfg["outputs"].update({
        "checkpoint": f"b32_frozen_reconstructors_seed_{seed}.pt",
        "history": f"b32_training_history_seed_{seed}.json",
        "training_metrics": f"b32_training_metrics_seed_{seed}.json",
        "alpha0_equivalence": f"b32_alpha0_equivalence_seed_{seed}.json",
        "alpha_sweep": f"b32_alpha_sweep_seed_{seed}.json",
        "alpha_scenarios": f"b32_alpha_scenarios_seed_{seed}.csv",
    })

    # Only run name, requested training seed, corresponding B0 checkpoint, and
    # output names differ from B3.1. Model, optimizer, losses, BlockMask,
    # normalization, benchmark, and the fixed alpha grid stay unchanged.
    expected = copy.deepcopy(base)
    expected["run_name"] = cfg["run_name"]
    expected["training"]["seed"] = seed
    expected["training"]["init_checkpoint"] = cfg["training"]["init_checkpoint"]
    expected["outputs"].update(cfg["outputs"])
    if cfg != expected:
        raise RuntimeError("unexpected B3.2 config change outside the allowed seed paths")

    config_dir = ROOT / "configs" / "b32"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"b32_seed_{seed}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run_seed(seed: int) -> dict:
    path = make_seed_config(seed)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    run_b31.verify_benchmark(cfg)
    source = Path(cfg["training"]["init_checkpoint"])
    if not source.is_file():
        raise FileNotFoundError(f"seed-{seed} B0-WCE checkpoint missing: {source}")
    sweep_path = METRICS / cfg["outputs"]["alpha_sweep"]
    checkpoint_path = CHECKPOINTS / cfg["outputs"]["checkpoint"]
    if sweep_path.is_file() and checkpoint_path.is_file():
        cached = json.loads(sweep_path.read_text(encoding="utf-8"))
        ckpt = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
        if (cached["training_seed"] == seed and
                cached["benchmark_sha256"] == run_b31.BENCHMARK_SHA256 and
                ckpt["training_seed"] == seed):
            return {"seed": seed, "reused": True,
                    "robust_scores": {a: v["robust_score"] for a, v in cached["alphas"].items()}}
    training = run_b31.train(path)
    sweep = run_b31.sweep(path)
    return {"seed": seed, "reused": False, "training": training,
            "best_alpha_by_robust_score": sweep["best_alpha_by_robust_score"],
            "robust_scores": {a: v["robust_score"] for a, v in sweep["alphas"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=(43, 44), choices=(43, 44))
    args = parser.parse_args()
    results = [run_seed(seed) for seed in args.seeds]
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
