"""Compare the locked B0 and P2 checkpoints under clean-score selection.

Reads Attachment2 validation only and reuses the frozen 54-scenario evaluator.
No model is trained or modified. Run from the E2026 repository root with
``python scripts/run_q2_cleanselect_fair_comparison.py``.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import Aligned50Dataset  # noqa: E402
from src.evaluation.missing_benchmark import (  # noqa: E402
    BENCHMARK_SEED,
    evaluate_benchmark,
    scenarios,
    summary as benchmark_summary,
)
from src.models.baseline import B0Baseline  # noqa: E402
from src.models.pooling_residual import B5PoolingResidual  # noqa: E402


BENCHMARK_HASH = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
SEEDS = (42, 43, 44)
METRICS = ("accuracy", "macro_f1", "mae", "pearson")
CONDITIONS = ("clean", "mean_missing")
OUT = ROOT / "outputs/final/q2/checkpoint_selection_fairness"
CHECKPOINTS = ROOT / "outputs/checkpoints"
MANIFEST = ROOT / "outputs/final/q2/q2_checkpoint_manifest.json"
BENCHMARK = ROOT / "outputs/metrics/b2_benchmark_definition.json"
PKL = ROOT / "data/raw/aligned_50.pkl"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def history_best_epoch(model: str, seed: int) -> int:
    metrics_dir = ROOT / "outputs/metrics"
    if model == "B0":
        filename = ("b0_weighted_ce_score_selection_history.json" if seed == 42
                    else f"b21_b0_seed_{seed}_history.json")
        rows = read_json(metrics_dir / filename)
        score = lambda row: float(row["valid_selection_score"])
    else:
        rows = (read_json(metrics_dir / "b5_pooling_history.json")["P2"] if seed == 42
                else read_json(metrics_dir / "b5_p2_multiseed_history.json")[str(seed)])
        score = lambda row: float(row["clean"]["selection_score"])
    require(rows and all("epoch" in row for row in rows), f"{model}/{seed}: empty history")
    # max returns the earliest occurrence, matching the original strict-greater update.
    return int(max(rows, key=score)["epoch"])


def chosen_checkpoints(manifest: dict) -> list[dict]:
    indexed = {(row["model"], int(row["seed"])): row
               for row in manifest["checkpoints"]}
    selected = []
    for seed in SEEDS:
        b0 = indexed[("B0-WCE", seed)]
        p2 = (manifest["p2_best_clean_checkpoints_retained_reference"]["42"]
              if seed == 42 else indexed[("B5-P2", seed)]["checkpoint"])
        for model, checkpoint in (("B0", b0["checkpoint"]), ("P2", p2)):
            require(checkpoint["exists_locally"] if "exists_locally" in checkpoint else True,
                    f"{model}/{seed}: checkpoint absent in manifest")
            path = ROOT / checkpoint["relative_path"]
            require(path.is_file(), f"{model}/{seed}: missing checkpoint {path}")
            digest = sha256(path)
            require(digest == checkpoint["sha256"],
                    f"{model}/{seed}: checkpoint SHA256 differs from lock")
            selected.append({"model": model, "seed": seed, "path": path,
                             "relative_path": path.relative_to(ROOT).as_posix(),
                             "sha256": digest,
                             "best_clean_epoch": history_best_epoch(model, seed)})
    return selected


def load_validation() -> Aligned50Dataset:
    require(sha256(BENCHMARK) == BENCHMARK_HASH, "benchmark definition SHA256 changed")
    definition = read_json(BENCHMARK)
    require(definition["seed"] == BENCHMARK_SEED == 20260923,
            "benchmark seed changed")
    require(definition["scenarios"] == [s.as_dict() for s in scenarios()],
            "benchmark scenario definitions changed")
    with PKL.open("rb") as stream:
        container = pickle.load(stream)
    require("valid" in container, "valid split is missing")
    valid = Aligned50Dataset(container["valid"], "valid")
    del container
    gc.collect()
    require(len(valid) == 728 and valid.padding_mask.shape == (728, 50),
            "validation sample count or padding shape changed")
    require(int(valid.padding_mask.sum()) == 18628,
            "validation valid-timestep count changed")
    return valid


def check_frozen_b0(p2: B5PoolingResidual, b0_state: dict, seed: int) -> None:
    candidate = p2.state_dict()
    changed = [name for name, tensor in b0_state.items()
               if name not in candidate or not torch.equal(candidate[name].cpu(), tensor.cpu())]
    require(not changed, f"P2/{seed}: frozen B0 tensors differ: {changed[:5]}")


def evaluate_one(item: dict, valid: Aligned50Dataset, b0_states: dict) -> dict:
    state = torch.load(item["path"], map_location="cpu", weights_only=False)
    cfg = state["config"]
    require(cfg["preprocessing"]["normalization"] == "none",
            f"{item['model']}/{item['seed']}: unexpected normalization")
    require(int(cfg["data"]["batch_size"]) == 128,
            f"{item['model']}/{item['seed']}: unexpected batch size")
    if item["model"] == "B0":
        require(int(cfg["training"]["seed"]) == item["seed"], "B0 seed mismatch")
        model = B0Baseline(**cfg["model"])
        epoch = int(state["best_epoch"])
        b0_states[item["seed"]] = state["model_state_dict"]
    else:
        require(state["mode"] == "mean_attention" and
                state["benchmark_sha256"] == BENCHMARK_HASH,
                "P2 mode or benchmark differs from lock")
        if "seed" in state:
            require(int(state["seed"]) == item["seed"], "P2 seed mismatch")
        else:
            require(int(cfg["training"]["seed"]) == item["seed"], "P2 seed mismatch")
        model = B5PoolingResidual("mean_attention", **cfg["model"])
        epoch = int(state["epoch"])
    require(epoch == item["best_clean_epoch"],
            f"{item['model']}/{item['seed']}: weight epoch is not clean-score best")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.float().eval()
    if item["model"] == "P2":
        check_frozen_b0(model, b0_states[item["seed"]], item["seed"])
    loader = DataLoader(valid, batch_size=128, shuffle=False, num_workers=0)
    rows = evaluate_benchmark(model, loader, torch.device("cpu"), scenario_chunk_size=6)
    expected_ids = ["clean", *(scenario.scenario_id for scenario in scenarios())]
    require([row["scenario_id"] for row in rows] == expected_ids,
            f"{item['model']}/{item['seed']}: scenario order changed")
    require(len(rows) == 55 and all(row["sample_count"] == 728 for row in rows),
            f"{item['model']}/{item['seed']}: scenario coverage incomplete")
    result = benchmark_summary(rows)
    require(all(math.isfinite(float(result[c][m]))
                for c in CONDITIONS for m in (*METRICS, "selection_score")),
            f"{item['model']}/{item['seed']}: non-finite metric")
    return {
        "model": item["model"], "selection_rule": "CleanSelect", "seed": item["seed"],
        "checkpoint_path": item["relative_path"], "checkpoint_sha256": item["sha256"],
        "selected_epoch": epoch, "benchmark_sha256": BENCHMARK_HASH,
        "validation_samples": len(valid),
        **{f"{condition}_{metric}": result[condition][metric]
           for condition in CONDITIONS for metric in (*METRICS, "selection_score")},
        "robust_score": result["robust_score"],
    }


def check_against_saved(row: dict) -> None:
    manifest = read_json(MANIFEST)
    if row["model"] == "P2" and row["seed"] == 42:
        expected = read_json(ROOT / "outputs/metrics/b5_pooling_checkpoint_comparison.json")[
            "checkpoint_rules"]["P2"]["best_clean"]["validation"]
    else:
        name = "B0-WCE" if row["model"] == "B0" else "B5-P2"
        expected = next(value["validation_metrics"] for value in manifest["checkpoints"]
                        if value["model"] == name and value["seed"] == row["seed"])
    for condition in CONDITIONS:
        for metric in (*METRICS, "selection_score"):
            difference = abs(float(row[f"{condition}_{metric}"]) -
                             float(expected[condition][metric]))
            require(difference <= 1e-5,
                    f"{row['model']}/{row['seed']} {condition}/{metric}: "
                    f"evaluation differs from saved result by {difference:.3g}")
    require(abs(row["robust_score"] - float(expected["robust_score"])) <= 1e-5,
            f"{row['model']}/{row['seed']}: robust score differs from saved result")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(seedwise: list[dict]) -> tuple[list[dict], list[dict], dict]:
    by_key = {(row["model"], row["seed"]): row for row in seedwise}
    require(len(by_key) == 6, "expected two models × three seeds")
    summary_rows = []
    for model in ("B0", "P2"):
        rows = [by_key[(model, seed)] for seed in SEEDS]
        agg = {"model": model, "selection_rule": "CleanSelect", "n_seeds": 3}
        for condition in CONDITIONS:
            for metric in (*METRICS, "selection_score"):
                vals = [float(row[f"{condition}_{metric}"]) for row in rows]
                agg[f"{condition}_{metric}_mean"] = mean(vals)
                agg[f"{condition}_{metric}_sample_sd"] = stdev(vals)
        vals = [float(row["robust_score"]) for row in rows]
        agg["robust_score_mean"] = mean(vals)
        agg["robust_score_sample_sd"] = stdev(vals)
        summary_rows.append(agg)

    paired_rows = []
    paired_summary = {}
    for condition in CONDITIONS:
        pairs = []
        for seed in SEEDS:
            base, improved = by_key[("B0", seed)], by_key[("P2", seed)]
            row = {"condition": condition, "seed": seed,
                   "accuracy_improvement": improved[f"{condition}_accuracy"] - base[f"{condition}_accuracy"],
                   "macro_f1_improvement": improved[f"{condition}_macro_f1"] - base[f"{condition}_macro_f1"],
                   "mae_improvement": base[f"{condition}_mae"] - improved[f"{condition}_mae"],
                   "pearson_improvement": improved[f"{condition}_pearson"] - base[f"{condition}_pearson"]}
            pairs.append(row)
            paired_rows.append(row)
        combined = {"condition": condition, "seed": "mean"}
        paired_summary[condition] = {}
        for metric in METRICS:
            key = f"{metric}_improvement"
            values = [float(row[key]) for row in pairs]
            paired_summary[condition][metric] = {
                "values": values, "mean": mean(values), "sample_sd": stdev(values),
                "positive_seeds": sum(value > 0 for value in values),
                "negative_seeds": sum(value < 0 for value in values),
                "zero_seeds": sum(value == 0 for value in values),
            }
            combined[key] = mean(values)
            combined[f"{metric}_sample_sd"] = stdev(values)
            combined[f"{metric}_positive_seeds"] = paired_summary[condition][metric]["positive_seeds"]
        paired_rows.append(combined)
    # Give all rows the same machine-readable columns.
    fields = list(dict.fromkeys(key for row in paired_rows for key in row))
    paired_rows = [{field: row.get(field, "") for field in fields} for row in paired_rows]
    return summary_rows, paired_rows, paired_summary


def fmt(value: float, sd: float, places: int = 4) -> str:
    # Stack mean and sample SD so the eight-metric table remains legible at
    # the manuscript's normal text width without shrinking the entire table.
    return rf"\shortstack[c]{{{value:.{places}f} \\ $\pm$ {sd:.{places}f}}}"


def write_table(summary_rows: list[dict], paired: dict) -> None:
    by_model = {row["model"]: row for row in summary_rows}
    lines = [
        r"% Generated by scripts/run_q2_cleanselect_fair_comparison.py; requires booktabs.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{统一完整输入验证集选模规则下的三种子性能（均值$\pm$样本标准差）}",
        r"\label{tab:q2_cleanselect_fair_comparison}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{完整输入} & \multicolumn{4}{c}{54种缺失场景平均} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"模型 & Accuracy$\uparrow$ & Macro-F1$\uparrow$ & MAE$\downarrow$ & Pearson$\uparrow$ & Accuracy$\uparrow$ & Macro-F1$\uparrow$ & MAE$\downarrow$ & Pearson$\uparrow$ \\",
        r"\midrule",
    ]
    for model, label in (("B0", "B0"), ("P2", "本文模型（P2）")):
        if model == "P2":
            lines.append(r"\addlinespace[2pt]")
        row = by_model[model]
        cells = [fmt(row[f"{condition}_{metric}_mean"],
                     row[f"{condition}_{metric}_sample_sd"])
                 for condition in CONDITIONS for metric in METRICS]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{本文模型相对B0的同种子配对改善（均值$\pm$样本标准差）}",
        r"\label{tab:q2_cleanselect_paired_delta}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"条件 & $\Delta$Accuracy & $\Delta$Macro-F1 & $\Delta$MAE$^{*}$ & $\Delta$Pearson \\",
        r"\midrule",
    ])
    for condition, label in (("clean", "完整输入"), ("mean_missing", "缺失场景平均")):
        if condition == "mean_missing":
            lines.append(r"\addlinespace[2pt]")
        cells = [fmt(paired[condition][metric]["mean"],
                     paired[condition][metric]["sample_sd"], 5)
                 for metric in METRICS]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\footnotesize $^{*}$MAE改善定义为B0减P2，其余指标为P2减B0；正值表示P2较优。",
        r"\end{table}",
    ])
    (OUT / "q2_cleanselect_fair_comparison_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_report(seedwise: list[dict], summary_rows: list[dict], paired: dict) -> None:
    by_model = {row["model"]: row for row in summary_rows}
    lines = [
        "# Q2 统一 CleanSelect 比较",
        "",
        f"三种子：42、43、44。完整输入验证集与固定54种缺失场景；场景定义 SHA256 `{BENCHMARK_HASH}`。",
        "P2 seed43/44 的既有保存权重分别对应完整输入验证分数最优的 epoch 12/1。",
        "",
        "## 使用的 checkpoint 与逐种子结果",
        "",
        "| 模型 | seed | epoch | checkpoint | SHA256 | Clean Acc / F1 / MAE / Pearson | Missing Acc / F1 / MAE / Pearson |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in seedwise:
        clean = " / ".join(f"{row[f'clean_{metric}']:.6f}" for metric in METRICS)
        missing = " / ".join(f"{row[f'mean_missing_{metric}']:.6f}" for metric in METRICS)
        lines.append(f"| {row['model']} | {row['seed']} | {row['selected_epoch']} | "
                     f"`{row['checkpoint_path']}` | `{row['checkpoint_sha256']}` | "
                     f"{clean} | {missing} |")
    lines.extend(["", "## 三种子均值 ± 样本标准差", "",
                  "| 条件 | 模型 | Accuracy | Macro-F1 | MAE | Pearson |",
                  "|---|---|---:|---:|---:|---:|"])
    for condition, label in (("clean", "完整输入"), ("mean_missing", "缺失场景平均")):
        for model in ("B0", "P2"):
            row = by_model[model]
            cells = [f"{row[f'{condition}_{metric}_mean']:.6f} ± "
                     f"{row[f'{condition}_{metric}_sample_sd']:.6f}" for metric in METRICS]
            lines.append(f"| {label} | {model} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 配对改善：P2 相对 B0", "",
                  "Accuracy、Macro-F1、Pearson 按 P2−B0；MAE 按 B0−P2。正值表示 P2 较优。",
                  "", "| 条件 | 指标 | seed42 | seed43 | seed44 | 均值 ± 样本标准差 | 正向 seed 数 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for condition, label in (("clean", "完整输入"), ("mean_missing", "缺失场景平均")):
        for metric in METRICS:
            stat = paired[condition][metric]
            values = [f"{value:+.6f}" for value in stat["values"]]
            lines.append(f"| {label} | {metric} | " + " | ".join(values) +
                         f" | {stat['mean']:+.6f} ± {stat['sample_sd']:.6f} | "
                         f"{stat['positive_seeds']}/3 |")
    lines.extend(["", "## 结果结论", ""])
    for condition, label in (("clean", "完整输入"), ("mean_missing", "54种缺失场景平均")):
        improved = [metric for metric in METRICS if paired[condition][metric]["mean"] > 0]
        decreased = [metric for metric in METRICS if paired[condition][metric]["mean"] < 0]
        stable = [metric for metric in METRICS if paired[condition][metric]["positive_seeds"] == 3]
        lines.append(f"- {label}：均值改善指标为 {', '.join(improved) or '无'}；"
                     f"均值下降指标为 {', '.join(decreased) or '无'}。"
                     f"三 seed 同方向改善的指标为 {', '.join(stable) or '无'}。")
    lines.append("- 这些数值仅描述三个固定训练种子的配对结果；论文表述按各项原始指标逐项给出。")
    (OUT / "q2_cleanselect_fair_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    torch.set_num_threads(4)
    selected = chosen_checkpoints(read_json(MANIFEST))
    valid = load_validation()
    b0_states: dict[int, dict] = {}
    seedwise = []
    for item in selected:
        print(f"Evaluating {item['model']} seed{item['seed']} epoch{item['best_clean_epoch']}",
              flush=True)
        row = evaluate_one(item, valid, b0_states)
        check_against_saved(row)
        seedwise.append(row)
        print(f"  clean={row['clean_selection_score']:.6f} "
              f"missing={row['mean_missing_selection_score']:.6f} "
              f"R={row['robust_score']:.6f}", flush=True)
    summary_rows, paired_rows, paired = aggregate(seedwise)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "q2_cleanselect_fair_comparison_seedwise.csv", seedwise)
    write_csv(OUT / "q2_cleanselect_fair_comparison_summary.csv", summary_rows)
    write_csv(OUT / "q2_cleanselect_fair_comparison_paired_delta.csv", paired_rows)
    write_table(summary_rows, paired)
    write_report(seedwise, summary_rows, paired)
    print(f"Saved five comparison files to {OUT}", flush=True)


if __name__ == "__main__":
    main()
