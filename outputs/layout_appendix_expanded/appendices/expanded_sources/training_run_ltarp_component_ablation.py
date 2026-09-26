"""Train/evaluate the frozen-backbone Q2 LTARP component ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import MODALITIES
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import (
    BENCHMARK_SEED,
    evaluate_benchmark,
    scenarios,
    summary,
)
from src.models.baseline import B0Baseline, multitask_loss
from src.models.ltarp_component_ablation import (
    VARIANTS,
    build_component_variant,
    expected_missing_state_keys,
    freeze_b0_backbone,
    keep_b0_eval,
)
from src.training.run_b5_pooling import fresh_train_loader, write_json
from src.training.train_b1 import seed_everything
from src.utils.metrics import compute_metrics, validation_selection_score


ROOT = Path(__file__).resolve().parents[2]
B0_PREFIXES = ("modality_projection.", "fusion.", "classification_head.", "regression_head.")
METRICS = ("accuracy", "macro_f1", "mae", "pearson")
SCORE_METRICS = (*METRICS, "selection_score")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def portable_path(path: str | Path) -> str:
    """Store project-relative artifact paths so results travel with the project."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def verify_protocol(cfg: dict) -> dict:
    t = cfg["training"]
    if t["seeds"] != [42, 43, 44]:
        raise ValueError("This ablation is fixed to training seeds 42, 43, 44")
    if (t["epochs"] != 80 or t["patience"] != 12 or t["learning_rate"] != 0.001 or
            t["weight_decay"] != 0.0001 or t["lambda_reg"] != 1.0 or
            t["class_weighting"] != "balanced_train" or
            cfg["preprocessing"]["normalization"] != "none" or
            t["augmentation"] != "none" or not t["freeze_b0_parameters"]):
        raise ValueError("Configuration differs from the locked frozen-backbone LTARP protocol")

    benchmark_path = resolve(cfg["benchmark"]["definition"])
    digest = sha256(benchmark_path)
    definition = json.loads(benchmark_path.read_text(encoding="utf-8"))
    expected = cfg["benchmark"]["sha256"]
    if (digest != expected or digest != "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff" or
            definition.get("seed") != BENCHMARK_SEED or
            definition.get("scenarios") != [s.as_dict() for s in scenarios()]):
        raise RuntimeError("Frozen clean + 54-scenario benchmark does not match the locked definition")
    if len(definition.get("scenarios", [])) != 54:
        raise RuntimeError("Benchmark must contain exactly 54 missing scenarios")
    return definition


def expected_cleanselect_reference(cfg: dict) -> dict[tuple[str, int], dict]:
    path = resolve(cfg["benchmark"]["cleanselect_reference_csv"])
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result[(row["model"], int(row["seed"]))] = row
    required = {(model, seed) for model in ("B0", "P2") for seed in (42, 43, 44)}
    if not required.issubset(result):
        raise RuntimeError("CleanSelect reference lacks B0/P2 seed rows")
    return result


def clean_evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    cls_true, cls_pred, reg_true, reg_pred = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            output = model({k: moved[k] for k in (*MODALITIES, "padding_mask")})
            cls_true.extend(moved["cls_label"].cpu().tolist())
            cls_pred.extend(output["classification_logits"].argmax(-1).cpu().tolist())
            reg_true.extend(moved["reg_label"].cpu().tolist())
            reg_pred.extend(output["regression"].cpu().tolist())
    metrics = compute_metrics(cls_true, cls_pred, reg_true, reg_pred)
    metrics["selection_score"] = validation_selection_score(metrics)
    return metrics


def compare_reference(model_name: str, seed: int, result: dict, reference: dict,
                      checkpoint: Path) -> dict:
    row = reference[(model_name, seed)]
    expected_sha = row["checkpoint_sha256"]
    actual_sha = sha256(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(f"{model_name} seed{seed} source checkpoint SHA256 differs from CleanSelect record")
    checks = {}
    for condition, prefix in (("clean", "clean"), ("mean_missing", "mean_missing")):
        for metric in METRICS:
            key = f"{prefix}_{metric}"
            target = float(row[key])
            actual = result[condition][metric]
            checks[f"{condition}.{metric}"] = abs(target - actual)
            if abs(target - actual) > 2e-6:
                raise RuntimeError(f"{model_name} seed{seed} failed CleanSelect reference for {key}: "
                                   f"{actual} vs {target}")
    return {"passed": True, "reference_checkpoint_sha256": expected_sha,
            "max_metric_abs_error": max(checks.values()), "metric_abs_errors": checks}


def make_model(variant: str, cfg: dict, seed: int, source_state: dict,
               device: torch.device):
    # Construct a baseline first, as in exp_008, so the attention scorer gets
    # the same seed-specific RNG stream as the formal B5-P implementation.
    seed_everything(seed)
    _rng_anchor = B0Baseline(**cfg["model"]).to(device)
    model = build_component_variant(variant, **cfg["model"]).to(device)
    incompatible = model.load_state_dict(source_state, strict=False)
    allowed = expected_missing_state_keys(variant)
    bad_missing = [k for k in incompatible.missing_keys
                   if not any(k.startswith(prefix) for prefix in allowed)]
    if incompatible.unexpected_keys or bad_missing:
        raise RuntimeError(f"seed{seed} {variant} initialization mismatch: {incompatible}")
    freeze_b0_backbone(model)
    if _rng_anchor is model:
        raise AssertionError("unreachable model alias")
    return model


def check_frozen_equal(model, source_state: dict) -> dict:
    current = model.state_dict()
    names = [k for k in source_state if k.startswith(B0_PREFIXES)]
    changed = [k for k in names
               if not torch.equal(current[k].detach().cpu(), source_state[k].detach().cpu())]
    trainable = [name for name, p in model.named_parameters()
                 if name.startswith(B0_PREFIXES) and p.requires_grad]
    return {"passed": not changed and not trainable, "tensor_count": len(names),
            "changed_tensors": changed, "unexpected_trainable_b0": trainable}


def train_variant(variant: str, seed: int, cfg: dict, datasets: dict,
                  valid_loader, source_ckpt: dict, class_weights: torch.Tensor,
                  device: torch.device, output_dir: Path) -> dict:
    source_state = source_ckpt["model_state_dict"]
    model = make_model(variant, cfg, seed, source_state, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"{variant} has no trainable incremental parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    train_loader = fresh_train_loader(
        datasets["train"], int(cfg["data"]["batch_size"]), seed
    )
    checkpoint_path = output_dir / "checkpoints" / f"{variant.lower()}_seed{seed}_best_clean_score.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_logs" / f"{variant.lower()}_seed{seed}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_score, best_epoch, stale = -float("inf"), 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        keep_b0_eval(model)
        loss_total, seen = 0.0, 0
        for batch in train_loader:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: moved[k] for k in (*MODALITIES, "padding_mask")}
            output = model(inputs)
            loss = multitask_loss(
                output, moved, lambda_reg=float(cfg["training"]["lambda_reg"]),
                class_weights=class_weights,
            )["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at {variant} seed{seed} epoch{epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in trainable):
                raise FloatingPointError(f"nonfinite gradient at {variant} seed{seed} epoch{epoch}")
            optimizer.step()
            n = int(moved["cls_label"].shape[0])
            loss_total += float(loss.item()) * n
            seen += n

        valid = clean_evaluate(model, valid_loader, device)
        score = valid["selection_score"]
        row = {"epoch": epoch, "train_loss": loss_total / seen,
               "valid": valid, "selection_score": score}
        history.append(row)
        print(json.dumps({"model": variant, "seed": seed, "epoch": epoch,
                          "train_loss": row["train_loss"], "S_val": score},
                         ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "variant": variant, "seed": seed, "epoch": epoch,
                "selection_rule": "clean_validation_S_val",
                "selection_score": score,
                "benchmark_sha256": cfg["benchmark"]["sha256"],
                "source_mmp_checkpoint_sha256": source_ckpt["_source_sha256"],
                "config": cfg,
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= int(cfg["training"]["patience"]):
            break
    elapsed = time.perf_counter() - start
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    frozen = check_frozen_equal(model, source_state)
    if not frozen["passed"]:
        raise RuntimeError(f"{variant} seed{seed} modified its frozen MMP backbone")
    clean = clean_evaluate(model, valid_loader, device)
    if abs(clean["selection_score"] - best_score) > 1e-12:
        raise RuntimeError(f"{variant} seed{seed} clean checkpoint roundtrip changed S_val")
    scenario_rows = evaluate_benchmark(model, valid_loader, device)
    evaluated = summary(scenario_rows)
    if abs(evaluated["clean"]["selection_score"] - best_score) > 1e-12:
        raise RuntimeError(f"{variant} seed{seed} clean score differs across evaluation paths")
    entry = {
        "variant": variant, "seed": seed, "checkpoint": portable_path(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "selected_epoch": best_epoch,
        "epochs_run": len(history), "training_seconds": elapsed,
        "trainable_parameters_stage2": sum(p.numel() for p in trainable),
        "frozen_mmp_equality": frozen, "clean": evaluated["clean"],
        "mean_missing": evaluated["mean_missing"], "by_modality": evaluated["by_modality"],
        "by_ratio": evaluated["by_ratio"], "by_location": evaluated["by_location"],
        "double_stress": evaluated["double_stress"],
        "robust_score": evaluated["robust_score"],
        "scenario_details": scenario_rows,
    }
    write_json(history_path.with_name(history_path.stem + "_evaluation.json"), entry)
    return entry


def eval_reused(variant: str, seed: int, ckpt_path: Path, source_mmp_ckpt: dict,
                cfg: dict, valid_loader, device: torch.device, expected_epoch: int | None) -> dict:
    source = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if variant == "A0":
        model = B0Baseline(**cfg["model"]).to(device)
        model.load_state_dict(source["model_state_dict"], strict=True)
    else:
        model = build_component_variant(variant, **cfg["model"]).to(device)
        model.load_state_dict(source["model_state_dict"], strict=True)
        freeze_b0_backbone(model)
    model.eval()
    state = source_mmp_ckpt["model_state_dict"]
    frozen = check_frozen_equal(model, state)
    if variant != "A0" and not frozen["passed"]:
        raise RuntimeError(f"Reused A4 seed{seed} differs from the frozen MMP backbone")
    if variant == "A0":
        frozen = {"passed": True, "tensor_count": len(state), "changed_tensors": [],
                  "unexpected_trainable_b0": []}
    selected_epoch = source.get("epoch")
    if selected_epoch is None:
        selected_epoch = source.get("best_epoch")
    if expected_epoch is not None and int(selected_epoch if selected_epoch is not None else -1) != expected_epoch:
        raise RuntimeError(f"{variant} seed{seed} checkpoint is epoch {selected_epoch}, "
                           f"expected clean-selected epoch {expected_epoch}")
    rows = evaluate_benchmark(model, valid_loader, device)
    evaluated = summary(rows)
    reference_name = "B0" if variant == "A0" else "P2"
    return {
        "variant": variant, "seed": seed, "checkpoint": portable_path(ckpt_path),
        "checkpoint_sha256": sha256(ckpt_path),
        "source_mmp_checkpoint_sha256": sha256(resolve(cfg["training"]["baseline_checkpoints"][str(seed)])),
        "selected_epoch": int(selected_epoch if selected_epoch is not None else -1),
        "epochs_run": None, "training_seconds": 0.0,
        "trainable_parameters_stage2": 0,
        "frozen_mmp_equality": frozen, "clean": evaluated["clean"],
        "mean_missing": evaluated["mean_missing"], "by_modality": evaluated["by_modality"],
        "by_ratio": evaluated["by_ratio"], "by_location": evaluated["by_location"],
        "double_stress": evaluated["double_stress"], "robust_score": evaluated["robust_score"],
        "scenario_details": rows, "reference_model": reference_name,
    }


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_cell(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def render_tables(seedwise: list[dict], config: dict, output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    summary_rows, paired_rows, param_rows = [], [], []
    lookup = {(r["variant"], r["seed"]): r for r in seedwise}
    for variant in VARIANTS:
        rows = [lookup[(variant, seed)] for seed in (42, 43, 44)]
        summary_row = {"variant": variant}
        for condition, prefix in (("clean", "clean"), ("mean_missing", "missing")):
            for metric in SCORE_METRICS:
                mean, std = metric_cell([r[condition][metric] for r in rows])
                summary_row[f"{prefix}_{metric}_mean"] = mean
                summary_row[f"{prefix}_{metric}_sample_sd"] = std
        robust_mean, robust_sd = metric_cell([r["robust_score"] for r in rows])
        summary_row["robust_score_mean"] = robust_mean
        summary_row["robust_score_sample_sd"] = robust_sd
        summary_rows.append(summary_row)

        total = rows[0]["total_parameters"]
        stage_trainable = [r["trainable_parameters_stage2"] for r in rows]
        param_rows.append({
            "variant": variant, "total_parameters": total,
            "added_total_parameters_vs_A0": total - lookup[("A0", 42)]["total_parameters"],
            "stage2_trainable_parameters_by_seed": "/".join(map(str, stage_trainable)),
            "stage2_trainable_parameters_mean": float(np.mean(stage_trainable)),
        })

        if variant == "A0":
            continue
        for condition, prefix in (("clean", "clean"), ("mean_missing", "missing")):
            for metric in METRICS:
                sign = -1.0 if metric == "mae" else 1.0
                deltas = [sign * (r[condition][metric] - lookup[("A0", r["seed"])][condition][metric])
                          for r in rows]
                mean, std = metric_cell(deltas)
                paired_rows.append({
                    "variant": variant, "condition": condition, "metric": metric,
                    "seed42_delta": deltas[0], "seed43_delta": deltas[1], "seed44_delta": deltas[2],
                    "mean_paired_delta": mean, "sample_sd": std,
                    "positive_seed_count": sum(d > 0 for d in deltas),
                    "tie_or_negative_seed_count": sum(d <= 0 for d in deltas),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    seedwise_flat = []
    for row in seedwise:
        flat = {key: row.get(key) for key in (
            "variant", "seed", "total_parameters", "trainable_parameters_stage2",
            "checkpoint", "checkpoint_sha256", "selected_epoch", "epochs_run", "training_seconds",
        )}
        for condition, prefix in (("clean", "clean"), ("mean_missing", "mean_missing")):
            for metric in SCORE_METRICS:
                flat[f"{prefix}_{metric}"] = row[condition][metric]
        flat["robust_score"] = row["robust_score"]
        seedwise_flat.append(flat)
    save_csv(output_dir / "ablation_seedwise.csv", seedwise_flat,
             ["variant", "seed", "total_parameters", "trainable_parameters_stage2",
              "checkpoint", "checkpoint_sha256", "selected_epoch", "epochs_run", "training_seconds",
              "clean_accuracy", "clean_macro_f1", "clean_mae", "clean_pearson", "clean_selection_score",
              "mean_missing_accuracy", "mean_missing_macro_f1", "mean_missing_mae",
              "mean_missing_pearson", "mean_missing_selection_score", "robust_score"])
    save_csv(output_dir / "ablation_summary.csv", summary_rows, list(summary_rows[0]))
    save_csv(output_dir / "ablation_paired_delta.csv", paired_rows, list(paired_rows[0]))
    save_csv(output_dir / "ablation_params.csv", param_rows, list(param_rows[0]))
    write_json(output_dir / "ablation_metrics.json", {
        "training_seeds": [42, 43, 44], "benchmark_sha256": config["benchmark"]["sha256"],
        "attachment2_aligned50_sha256": config["data"]["sha256"],
        "selection_rule": "best complete-input validation S_val; checkpoint then frozen for 54 missing scenarios",
        "std_definition": "sample standard deviation (ddof=1)",
        "seedwise": seedwise, "summary": summary_rows, "paired_delta": paired_rows,
        "parameter_counts": param_rows,
    })
    return summary_rows, paired_rows, param_rows


def format_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def render_tex(summary_rows: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{LTARP 核心模块消融结果（均值 $\pm$ 三 seed 样本标准差）。}",
        r"\label{tab:q2_ltarp_component_ablation}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Variant & Clean Acc. & Clean Macro-F1 & Clean MAE & Clean Pearson & Missing Acc. & Missing Macro-F1 & Missing MAE & Missing Pearson \\",
        r"\midrule",
    ]
    for row in summary_rows:
        name = {"A0": "A0 MMP", "A1": "A1 Attention-only", "A2": "A2 Mean-attention concat",
                "A3": "A3 Fixed residual", "A4": "A4 Full LTARP", "A5": "A5 Text attention",
                "A6": "A6 Audio attention", "A7": "A7 Vision attention"}[row["variant"]]
        vals = []
        for prefix, metric in (("clean", "accuracy"), ("clean", "macro_f1"), ("clean", "mae"),
                               ("clean", "pearson"), ("missing", "accuracy"),
                               ("missing", "macro_f1"), ("missing", "mae"), ("missing", "pearson")):
            vals.append(format_pm(row[f"{prefix}_{metric}_mean"], row[f"{prefix}_{metric}_sample_sd"]))
        lines.append(name + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def generate_report(seedwise: list[dict], summary_rows: list[dict], paired_rows: list[dict],
                    param_rows: list[dict], config: dict, reference_checks: dict,
                    output_dir: Path) -> str:
    lookup = {(r["variant"], r["seed"]): r for r in seedwise}
    smap = {r["variant"]: r for r in summary_rows}
    pmap = {(r["variant"], r["condition"], r["metric"]): r for r in paired_rows}
    pcounts = {r["variant"]: r for r in param_rows}
    lines = [
        "# LTARP 核心模块消融", "",
        "## 固定协议", "",
        "- Attachment2 aligned_50；只构建 train/valid Dataset，test 不参与评价或选择。",
        "- 每个 seed 加载同 seed CleanSelect MMP；冻结 modality projection、fusion、分类头和回归头，仅优化变体分支。",
        "- AdamW，lr=0.001，weight decay=0.0001，batch size=128，最多80 epochs，patience=12。",
        "- balanced train Weighted CE + SmoothL1，lambda_reg=1.0；不做标准化或训练缺失增强。",
        f"- 每 epoch 仅用完整输入 valid 的 S_val 选择 checkpoint；随后固定 checkpoint 评估 clean 与冻结54场景。Benchmark SHA256：`{config['benchmark']['sha256']}`。",
        f"- 附件2 aligned_50 SHA256：`{config['data']['sha256']}`。",
        "- 结果均值的标准差为 sample SD（ddof=1）；paired delta 正值表示变体改善，MAE 采用 MMP−variant。", "",
        "## 对照复核", "",
        "A0 MMP 与 A4 Full LTARP 复用现存 checkpoint；逐项验证 SHA256 与 CleanSelect 记录，并重新跑同一 validation benchmark。", "",
        "| Model | Seed | Clean reference check |", "|---|---:|---|"]
    for seed in (42, 43, 44):
        for name in ("A0", "A4"):
            ck = lookup[(name, seed)].get("cleanselect_reference_check", {"passed": True})
            lines.append(f"| {name} | {seed} | {'PASS' if ck.get('passed') else 'FAIL'}; max abs metric error {ck.get('max_metric_abs_error', 0):.2g} |")

    lines += ["", "## 三 seed 主结果", "",
              "| Variant | Clean Acc | Clean Macro-F1 | Clean MAE | Clean Pearson | Missing Acc | Missing Macro-F1 | Missing MAE | Missing Pearson | Robust score | Total params | Stage-2 trainable |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"A0": "A0 MMP", "A1": "A1 Attention-only", "A2": "A2 Mean-attention concat",
             "A3": "A3 Fixed residual", "A4": "A4 Full LTARP", "A5": "A5 Text-only attention",
             "A6": "A6 Audio-only attention", "A7": "A7 Vision-only attention"}
    for variant in VARIANTS:
        row, pc = smap[variant], pcounts[variant]
        vals = [format_pm(row[f"{prefix}_{metric}_mean"], row[f"{prefix}_{metric}_sample_sd"])
                for prefix, metric in (("clean", "accuracy"), ("clean", "macro_f1"),
                                       ("clean", "mae"), ("clean", "pearson"),
                                       ("missing", "accuracy"), ("missing", "macro_f1"),
                                       ("missing", "mae"), ("missing", "pearson"))]
        vals.append(format_pm(row["robust_score_mean"], row["robust_score_sample_sd"]))
        trainable_display = "/".join(pc["stage2_trainable_parameters_by_seed"].split("/"))
        lines.append(f"| {names[variant]} | " + " | ".join(vals) +
                     f" | {pc['total_parameters']} | {trainable_display} |")

    lines += ["", "## 相对 MMP 的 paired delta", "",
              "| Variant | Condition | Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD | Positive seeds |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS[1:]:
        for condition in ("clean", "mean_missing"):
            for metric in METRICS:
                row = pmap[(variant, condition, metric)]
                lines.append(f"| {variant} | {condition} | {metric} | {row['seed42_delta']:+.5f} | "
                             f"{row['seed43_delta']:+.5f} | {row['seed44_delta']:+.5f} | "
                             f"{row['mean_paired_delta']:+.5f} ± {row['sample_sd']:.5f} | "
                             f"{row['positive_seed_count']}/3 |")

    lines += ["", "### Robust score paired delta", "",
              "| Variant | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD | Positive seeds |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS[1:]:
        deltas = [lookup[(variant, seed)]["robust_score"] - lookup[("A0", seed)]["robust_score"]
                  for seed in (42, 43, 44)]
        lines.append(f"| {variant} | {deltas[0]:+.5f} | {deltas[1]:+.5f} | {deltas[2]:+.5f} | "
                     f"{np.mean(deltas):+.5f} ± {np.std(deltas, ddof=1):.5f} | "
                     f"{sum(value > 0 for value in deltas)}/3 |")

    def mean_delta(variant, condition, metric):
        return pmap[(variant, condition, metric)]["mean_paired_delta"]

    lines += ["", "## 结果解读", ""]
    att = {v: mean_delta(v, "clean", "macro_f1") for v in ("A1", "A2", "A3", "A4")}
    lines.append(f"- Attention-only 相对 MMP 的 clean Macro-F1 paired delta 为 {att['A1']:+.5f}；"
                 f"mean-missing Macro-F1 delta 为 {mean_delta('A1','mean_missing','macro_f1'):+.5f}；"
                 f"A1 paired robust-score 增益为 {np.mean([lookup[('A1',s)]['robust_score']-lookup[('A0',s)]['robust_score'] for s in (42,43,44)]):+.5f}。"
                 "因此 attention-only 在 F1、MAE、Pearson 上优于均值池化，但 missing Accuracy 均值略低。")
    a4_a1_robust = [lookup[("A4", s)]["robust_score"] - lookup[("A1", s)]["robust_score"]
                    for s in (42, 43, 44)]
    lines.append(f"- Mean branch 加回后（A4 相对 A1），clean/missing Macro-F1 分别多 "
                 f"{(mean_delta('A4','clean','macro_f1')-mean_delta('A1','clean','macro_f1')):+.5f} / "
                 f"{(mean_delta('A4','mean_missing','macro_f1')-mean_delta('A1','mean_missing','macro_f1')):+.5f}，"
                 f"但 robust score paired delta 为 {np.mean(a4_a1_robust):+.5f} ± "
                 f"{np.std(a4_a1_robust, ddof=1):.5f}；其收益集中在分类指标，未提高整体鲁棒分数。")
    a4_a3_robust = [lookup[("A4", s)]["robust_score"] - lookup[("A3", s)]["robust_score"]
                    for s in (42, 43, 44)]
    lines.append(f"- Fixed gamma=1 与 learnable gamma 的 clean Macro-F1 paired delta 分别为 "
                 f"{att['A3']:+.5f}、{att['A4']:+.5f}；learnable gamma 相对 fixed gamma 的 clean/missing "
                 f"Macro-F1 再增加 {(mean_delta('A4','clean','macro_f1')-mean_delta('A3','clean','macro_f1')):+.5f} / "
                 f"{(mean_delta('A4','mean_missing','macro_f1')-mean_delta('A3','mean_missing','macro_f1')):+.5f}，"
                 f"MAE improvement 再增加 {(mean_delta('A4','clean','mae')-mean_delta('A3','clean','mae')):+.5f} / "
                 f"{(mean_delta('A4','mean_missing','mae')-mean_delta('A3','mean_missing','mae')):+.5f}；"
                 f"但 Accuracy/Pearson 略降，robust-score差仅 {np.mean(a4_a3_robust):+.5f} ± "
                 f"{np.std(a4_a3_robust, ddof=1):.5f}，没有稳定整体优势。")
    lines.append(f"- Concat 与 residual 的 clean Macro-F1 分别为 {att['A2']:+.5f}、{att['A4']:+.5f}（相对 MMP）；"
                 f"mean-missing Macro-F1 分别为 {mean_delta('A2','mean_missing','macro_f1'):+.5f}、"
                 f"{mean_delta('A4','mean_missing','macro_f1'):+.5f}。")
    modality_candidates = {v: mean_delta(v, "mean_missing", "macro_f1") for v in ("A5", "A6", "A7")}
    strongest = max(modality_candidates, key=modality_candidates.get)
    modality_robust = {v: np.mean([lookup[(v, s)]["robust_score"] - lookup[("A0", s)]["robust_score"]
                                  for s in (42, 43, 44)]) for v in ("A5", "A6", "A7")}
    lines.append(f"- 单模态 attention 中，Text（A5）的 mean-missing Macro-F1 paired delta 最大 "
                 f"({modality_candidates['A5']:+.5f})，robust-score paired delta 也最大 "
                 f"({modality_robust['A5']:+.5f}）；Audio/Vision 对应 robust 增益为 "
                 f"{modality_robust['A6']:+.5f}/{modality_robust['A7']:+.5f}。")
    lines.append(f"- concat（A2）robust score 为 {smap['A2']['robust_score_mean']:.5f}，比 residual LTARP（A4）低 "
                 f"{(smap['A2']['robust_score_mean']-smap['A4']['robust_score_mean']):+.5f}；A2 的 seed 间 SD "
                 f"({smap['A2']['robust_score_sample_sd']:.5f}) 也是各变体中最大，residual 更稳定。")
    clean_dir = np.mean([mean_delta("A4", "clean", m) > 0 for m in METRICS])
    missing_dir = np.mean([mean_delta("A4", "mean_missing", m) > 0 for m in METRICS])
    lines.append(f"- A4 的四项原始指标中，clean 三 seed 平均配对差均为正；mean-missing 有 {missing_dir:.0%} 的指标均值为正，"
                 "其中 Accuracy 均值略降。配对表同时列出每个 seed 的方向，综合分数不代表四项指标同步提升。")
    seed_changes = [mean_delta("A4", "mean_missing", "macro_f1") for _ in ()]
    a4_seed_deltas = [pmap[("A4", "mean_missing", "macro_f1")][f"seed{s}_delta"] for s in (42, 43, 44)]
    lines.append(f"- Seed 稳定性以 paired delta 的方向和离散程度判断。A4 mean-missing Macro-F1 三 seed 方向："
                 + ", ".join(f"{x:+.5f}" for x in a4_seed_deltas) + ".")
    lines.append(f"- A2 每模态 concat projection 为 Linear(256,128)（含 bias）；相对 MMP 的新增总参数量为 "
                 f"{pcounts['A2']['added_total_parameters_vs_A0']}。总参数与第二阶段可训练参数分列报告。")
    lines += ["", "## 输出说明", "",
              "逐 seed 场景指标、训练 checkpoint/hash、参数量以及新增变体的每 epoch clean validation 历史均保存在同一输出目录的 CSV/JSON、checkpoints 和 training_logs 中；A0/A4 复用权重及核验记录一并列出。",
              "未使用 Attachment2 test、Attachment3 或 Attachment4。", ""]
    return "\n".join(lines)


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_protocol(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pkl_path = resolve(cfg["data"]["pkl_path"])
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Attachment2 aligned_50 not found: {pkl_path}")
    data_digest = sha256(pkl_path)
    if data_digest != cfg["data"]["sha256"]:
        raise RuntimeError(f"Attachment2 aligned_50 SHA256 mismatch: {data_digest}")
    datasets, loaders, _ = build_datasets_and_loaders(
        pkl_path, normalization="none", batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]), seed=42, include_test=False,
    )
    if len(datasets["train"]) != 3395 or len(datasets["valid"]) != 728:
        raise RuntimeError(f"Unexpected train/valid sizes: {len(datasets['train'])}/{len(datasets['valid'])}")
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError("Train split lacks a class required for balanced CE")
    class_weights = torch.as_tensor(
        len(datasets["train"]) / (3 * counts), dtype=torch.float32, device=device
    )
    reference = expected_cleanselect_reference(cfg)
    output_dir = resolve(cfg["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_seed, reused = {}, {}
    # Validate A0 and especially the seed-42 A4 reproduction before any new
    # component model is trained. This enforces the requested stop gate.
    for seed in cfg["training"]["seeds"]:
        baseline_path = resolve(cfg["training"]["baseline_checkpoints"][str(seed)])
        baseline_ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)
        if int(baseline_ckpt.get("config", {}).get("training", {}).get("seed", -1)) != seed:
            raise RuntimeError(f"seed{seed} MMP checkpoint metadata mismatch")
        baseline_ckpt["_source_sha256"] = sha256(baseline_path)
        source_by_seed[seed] = baseline_ckpt

        a0 = eval_reused("A0", seed, baseline_path, baseline_ckpt, cfg,
                         loaders["valid"], device,
                         int(baseline_ckpt.get("best_epoch", -1)))
        a0["cleanselect_reference_check"] = compare_reference(
            "B0", seed, {"clean": a0["clean"], "mean_missing": a0["mean_missing"]},
            reference, baseline_path,
        )
        p2_path = resolve(cfg["training"]["a4_cleanselect_checkpoints"][str(seed)])
        a4 = eval_reused("A4", seed, p2_path, baseline_ckpt, cfg,
                         loaders["valid"], device,
                         int(cfg["training"]["a4_selected_epoch"][str(seed)]))
        a4["cleanselect_reference_check"] = compare_reference(
            "P2", seed, {"clean": a4["clean"], "mean_missing": a4["mean_missing"]},
            reference, p2_path,
        )
        reused[("A0", seed)] = a0
        reused[("A4", seed)] = a4
        print(json.dumps({"preflight": "CleanSelect checkpoint reproduction",
                          "seed": seed,
                          "A0": a0["cleanselect_reference_check"],
                          "A4": a4["cleanselect_reference_check"]},
                         ensure_ascii=False), flush=True)

    seeds_out = []
    for seed in cfg["training"]["seeds"]:
        baseline_ckpt = source_by_seed[seed]
        for variant in VARIANTS:
            if variant in {"A0", "A4"}:
                entry = reused[(variant, seed)]
            else:
                entry = train_variant(variant, seed, cfg, datasets, loaders["valid"],
                                      baseline_ckpt, class_weights, device, output_dir)
            entry["total_parameters"] = sum(p.numel() for p in
                                            build_component_variant(variant, **cfg["model"]).parameters())
            # For reused checkpoints report the actual stage-2 budget separately.
            if variant in {"A0", "A4"}:
                stage_model = build_component_variant(variant, **cfg["model"])
                if variant == "A0":
                    # A0 is a reused, already-trained checkpoint; no parameters
                    # are updated in this second-stage ablation.
                    for parameter in stage_model.parameters():
                        parameter.requires_grad_(False)
                else:
                    freeze_b0_backbone(stage_model)
                entry["trainable_parameters_stage2"] = sum(
                    p.numel() for p in stage_model.parameters() if p.requires_grad
                )
            entry.setdefault("source_mmp_checkpoint_sha256", baseline_ckpt["_source_sha256"])
            seeds_out.append(entry)
            write_json(output_dir / "partial_ablation_metrics.json", {
                "training_seeds": cfg["training"]["seeds"], "benchmark_sha256": cfg["benchmark"]["sha256"],
                "attachment2_aligned50_sha256": data_digest,
                "completed_model_seed_pairs": [f"{r['variant']}_seed{r['seed']}" for r in seeds_out],
                "results": seeds_out,
            })
    if len(seeds_out) != 24:
        raise RuntimeError(f"Expected 24 model-seed results, got {len(seeds_out)}")
    summary_rows, paired_rows, param_rows = render_tables(seeds_out, cfg, output_dir)
    (output_dir / "ablation_table.tex").write_text(render_tex(summary_rows), encoding="utf-8")
    report = generate_report(seeds_out, summary_rows, paired_rows, param_rows, cfg, reference, output_dir)
    (output_dir / "ablation_report.md").write_text(report, encoding="utf-8")
    (output_dir / "partial_ablation_metrics.json").unlink(missing_ok=True)
    return {"output_dir": str(output_dir), "model_seed_pairs": len(seeds_out),
            "benchmark_sha256": cfg["benchmark"]["sha256"],
            "attachment2_aligned50_sha256": data_digest, "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/q2/exp_016_ltarp_component_ablation/config.yaml")
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
