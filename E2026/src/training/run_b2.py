"""B2: frozen validation benchmark, then block-only augmentation training.

Invoke `inherent` before `train`. Neither path constructs a test Dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.block_mask import augment_train_batch, predictor_inputs
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import BENCHMARK_SEED, evaluate_benchmark, scenarios, summary
from src.models.baseline import B0Baseline, multitask_loss
from src.models.temporal import B1TemporalEncoder
from src.training.train_b1 import seed_everything

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "outputs"
METRICS = OUTPUT / "metrics"
CHECKPOINTS = OUTPUT / "checkpoints"
FLAT_METRICS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
BASELINES = {
    "B0-WCE": ("b0", "b0_weighted_ce_best_selection_score.pt"),
    "B1-WCE": ("b1", "b1_weighted_ce_best_selection_score.pt"),
}


def model_for(backbone: str, model_cfg: dict) -> torch.nn.Module:
    if backbone == "b0":
        return B0Baseline(**model_cfg)
    if backbone == "b1":
        return B1TemporalEncoder(**model_cfg)
    raise ValueError(f"unsupported backbone: {backbone}")


def data_loaders(pkl_path: str, batch_size: int, seed: int):
    return build_datasets_and_loaders(
        pkl_path, normalization="none", batch_size=batch_size,
        num_workers=0, seed=seed, include_test=False,
    )


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flat_rows(model_name: str, rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        item = {"model": model_name, "scenario_id": row["scenario_id"],
                "modalities": row["modalities"], "rho": row["rho"],
                "location": row["location"], "sample_count": row["sample_count"]}
        item.update({m: row[m] for m in FLAT_METRICS})
        item.update({f"delta_{m}": row[f"delta_{m}"] for m in FLAT_METRICS})
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_inherent(pkl_path: str) -> None:
    """First phase: inference only, existing frozen best-selection checkpoints."""
    device = device_auto()
    definitions = {"version": "b2_block_v1", "seed": BENCHMARK_SEED,
                   "length_rule": "max(1, round(rho * valid_length)); Python half-to-even",
                   "locations": {"early": "0", "middle": "floor((L-ell)/2)", "late": "L-ell"},
                   "padding": "text_bert[:,1,:] audited valid prefix; never masked",
                   "availability_as_predictor_input": False,
                   "scenarios": [s.as_dict() for s in scenarios()]}
    write_json(METRICS / "b2_benchmark_definition.json", definitions)
    all_rows = []
    summaries = {}
    for name, (backbone, checkpoint_name) in BASELINES.items():
        state = torch.load(CHECKPOINTS / checkpoint_name, map_location="cpu", weights_only=False)
        cfg = state["config"]
        seed_everything(int(cfg["training"]["seed"]))
        datasets, loaders, _ = data_loaders(pkl_path, int(cfg["data"]["batch_size"]),
                                            int(cfg["training"]["seed"]))
        model = model_for(backbone, cfg["model"]).to(device)
        model.load_state_dict(state["model_state_dict"])
        rows = evaluate_benchmark(model, loaders["valid"], device)
        all_rows.extend(flat_rows(name, rows))
        summaries[name] = {"summary": summary(rows), "checkpoint": checkpoint_name,
                           "valid_count": len(datasets["valid"])}
        write_json(METRICS / f"b2_{backbone}_inherent_detail.json", rows)
        print(json.dumps({"phase": "inherent", "model": name, **summaries[name]["summary"]},
                         ensure_ascii=False), flush=True)
    write_csv(METRICS / "b2_inherent_robustness.csv", all_rows)
    write_json(METRICS / "b2_inherent_summary.json", summaries)


def run_train(config_path: Path) -> None:
    if not (METRICS / "b2_inherent_robustness.csv").exists():
        raise RuntimeError("Run inherent checkpoint benchmark before any B2 training")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["preprocessing"]["normalization"] != "none":
        raise ValueError("B2 comparison requires normalization=none")
    training = cfg["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    rng = random.Random(seed)
    device = device_auto()
    datasets, loaders, _ = data_loaders(cfg["data"]["pkl_path"],
                                        int(cfg["data"]["batch_size"]), seed)
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0):
        raise ValueError("class weights undefined for empty train class")
    weights = torch.as_tensor(len(datasets["train"]) / (3.0 * counts),
                              dtype=torch.float32, device=device)
    model = model_for(cfg["backbone"], cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    best_clean = -float("inf")
    best_robust = -float("inf")
    best_clean_epoch = best_robust_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    clean_path = CHECKPOINTS / cfg["outputs"]["best_clean_checkpoint"]
    robust_path = CHECKPOINTS / cfg["outputs"]["best_robust_checkpoint"]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        blocks = 0
        for batch in loaders["train"]:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            masked, metadata = augment_train_batch(batch, rng)
            blocks += len(metadata)
            optimizer.zero_grad(set_to_none=True)
            output = model(predictor_inputs(masked))
            loss = multitask_loss(output, masked,
                                  lambda_reg=float(training["lambda_reg"]),
                                  class_weights=weights)["total"]
            loss.backward()
            optimizer.step()
            n = batch["cls_label"].shape[0]
            total_loss += float(loss.item()) * n
            seen += n
        rows = evaluate_benchmark(model, loaders["valid"], device)
        result = summary(rows)
        clean_score = result["clean"]["selection_score"]
        robust_score = result["robust_score"]
        row = {"epoch": epoch, "train_loss": total_loss / seen,
               "augmentation_blocks": blocks,
               "clean": result["clean"], "mean_missing": result["mean_missing"],
               "robust_score": robust_score}
        history.append(row)
        checkpoint = {"model_state_dict": model.state_dict(), "config": cfg,
                      "train_class_counts": counts.tolist(), "class_weights": weights.cpu(),
                      "benchmark_seed": BENCHMARK_SEED, "epoch": epoch,
                      "clean_score": clean_score, "robust_score": robust_score}
        if clean_score > best_clean:
            best_clean, best_clean_epoch = clean_score, epoch
            torch.save(checkpoint, clean_path)
        if robust_score > best_robust:
            best_robust, best_robust_epoch, stale = robust_score, epoch, 0
            torch.save(checkpoint, robust_path)
        else:
            stale += 1
        print(json.dumps({"model": cfg["run_name"], **row}, ensure_ascii=False), flush=True)
        if stale >= int(training["patience"]):
            break
    seconds = time.perf_counter() - started
    write_json(METRICS / cfg["outputs"]["history_filename"], history)
    result = {"model": cfg["run_name"], "architecture": cfg["backbone"],
              "normalization": "none", "benchmark_seed": BENCHMARK_SEED,
              "best_clean_epoch": best_clean_epoch, "best_robust_epoch": best_robust_epoch,
              "best_clean_score": best_clean, "best_robust_score": best_robust,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": seconds, "train_class_counts": counts.tolist(),
              "class_weights": weights.cpu().tolist(),
              "train_count": len(datasets["train"]), "valid_count": len(datasets["valid"]),
              "test_role": "never constructed/evaluated"}
    write_json(METRICS / cfg["outputs"]["metrics_filename"], result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run_compare(pkl_path: str) -> None:
    """Evaluate four selected checkpoints on identical frozen validation scenes."""
    device = device_auto()
    models = {**BASELINES,
              "B2-B0-BlockMask": ("b0", "b2_b0_best_robust_score.pt"),
              "B2-B1-BlockMask": ("b1", "b2_b1_best_robust_score.pt")}
    all_flat, all_summary = [], {}
    for name, (backbone, checkpoint_name) in models.items():
        state = torch.load(CHECKPOINTS / checkpoint_name, map_location="cpu", weights_only=False)
        cfg = state["config"]
        seed_everything(int(cfg["training"]["seed"]))
        _, loaders, _ = data_loaders(pkl_path, int(cfg["data"]["batch_size"]),
                                     int(cfg["training"]["seed"]))
        model = model_for(backbone, cfg["model"]).to(device)
        model.load_state_dict(state["model_state_dict"])
        rows = evaluate_benchmark(model, loaders["valid"], device)
        all_flat.extend(flat_rows(name, rows))
        all_summary[name] = {"checkpoint": checkpoint_name, "summary": summary(rows),
                             "clean_vision_all_zero": rows[0]["vision_all_zero_metrics"]}
        if name.startswith("B2-"):
            write_json(METRICS / f"{name.lower().replace('-', '_')}_detail.json", rows)
        print(json.dumps({"phase": "comparison", "model": name,
                          "summary": all_summary[name]["summary"]}, ensure_ascii=False), flush=True)
    write_csv(METRICS / "b2_missing_scenarios.csv", all_flat)
    write_json(METRICS / "b2_summary.json", all_summary)
    make_report(all_summary, all_flat)


def make_report(result: dict, flat: list[dict]) -> None:
    lines = ["# Q2 B2 fixed validation block benchmark", "",
             "Project-internal score only; no attachment2 test or attachment3 used.",
             "Fixed benchmark: `b2_benchmark_definition.json` (54 missing + clean).", "",
             "| Model | Clean Acc | Clean F1 | Clean MAE | Clean r | Missing Acc | Missing F1 | Missing MAE | Missing r | Robust score |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data in result.items():
        s = data["summary"]
        c, m = s["clean"], s["mean_missing"]
        values = [c[x] for x in ("accuracy", "macro_f1", "mae", "pearson")] + [m[x] for x in ("accuracy", "macro_f1", "mae", "pearson")] + [s["robust_score"]]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines.extend(["", "Ratio and modality aggregates use the 45 single-modality scenes.",
                  "Location aggregates use all 54 missing scenes. Double stress uses the 9 two-modality scenes.",
                  "Averages are scenario means on the same 728 validation samples; no test split contributes."])
    for section in ("by_ratio", "by_location", "by_modality", "double_stress"):
        lines.extend(["", f"## {section}", "", "| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for name, data in result.items():
            for group, values in data["summary"][section].items():
                lines.append(f"| {name} | {group} | " + " | ".join(f"{values[x]:.4f}" for x in FLAT_METRICS) + " |")
    (METRICS / "b2_model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = list(result)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for modality in ("text", "audio", "vision"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for ax, metric in zip(axes.flat, FLAT_METRICS[:4]):
            for name, color in zip(names, colors):
                y = [np.mean([r[metric] for r in flat if r["model"] == name and r["modalities"] == modality and r["rho"] == rho])
                     for rho in (0.1, 0.2, 0.3, 0.4, 0.5)]
                ax.plot((0.1, 0.2, 0.3, 0.4, 0.5), y, marker="o", label=name, color=color)
            ax.set(xlabel="Missing ratio over valid length", ylabel=metric, title=f"{modality} missing: {metric}")
            ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=8)
        fig.savefig(figures / f"b2_missing_ratio_{modality}.png", dpi=160)
        plt.close(fig)
    for name in names:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, metric in zip(axes.flat, FLAT_METRICS[:4]):
            ax.bar(("early", "middle", "late"), [result[name]["summary"]["by_location"][loc][metric]
                                                    for loc in ("early", "middle", "late")])
            ax.set(title=metric, ylabel=metric)
        fig.suptitle(name + " - all missing scenarios by location")
        fig.savefig(figures / f"b2_missing_location_{name.lower().replace('-', '_')}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(3)
    for i, (name, color) in enumerate(zip(names, colors)):
        ax.bar(x + (i - 1.5) * .19, [result[name]["summary"]["by_modality"][m]["selection_score"]
                                    for m in ("text", "audio", "vision")], .18, label=name, color=color)
    ax.set_xticks(x, ("text", "audio", "vision"))
    ax.set(ylabel="Mean selection score", title="Single-modality missing sensitivity")
    ax.legend(fontsize=8)
    fig.savefig(figures / "b2_modality_sensitivity.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("inherent", "train", "compare"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pkl-path", default="/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl")
    args = parser.parse_args()
    if args.phase == "inherent":
        run_inherent(args.pkl_path)
    elif args.phase == "train":
        if args.config is None:
            parser.error("train requires --config")
        run_train(args.config)
    else:
        run_compare(args.pkl_path)


if __name__ == "__main__":
    main()
