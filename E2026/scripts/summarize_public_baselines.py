"""Generate Q2 public-baseline tables from completed per-seed result files."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


METRICS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
MODELS = ("TFN", "MulT", "MISA")
SEEDS = (42, 43, 44)


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values), "values": values}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path):
    base = root / "experiments/q2/public_baselines"
    output = root / "outputs/final/q2/public_baselines"
    output.mkdir(parents=True, exist_ok=True)
    clean_rows, miss_rows, group_rows = [], [], []
    results = {}
    for model in MODELS:
        per_seed = []
        for seed in SEEDS:
            path = base / model / f"metrics_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing formal result: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["smoke"] or result["seed"] != seed or result["model"] != model:
                raise ValueError(f"Invalid result identity: {path}")
            if result["missing"] is None:
                raise ValueError(f"Missing frozen benchmark: {path}")
            per_seed.append(result)
            clean = result["clean_valid"]
            missing = result["missing"]["summary"]["mean_missing"]
            clean_score = result["clean_selection_score"]
            robust = result["missing"]["summary"]["robust_score"]
            clean_rows.append({"model": model, "seed": seed, "parameters": result["parameter_count"],
                               "training_seconds": result["training_seconds"],
                               **{m: clean_score if m == "selection_score" else clean[m] for m in METRICS}})
            miss_rows.append({"model": model, "seed": seed, "robust_score": robust,
                              **{m: missing[m] for m in METRICS}})
            for category in ("by_ratio", "by_location", "by_modality", "double_stress"):
                for group, values in result["missing"]["summary"][category].items():
                    group_rows.append({"model": model, "seed": seed, "category": category,
                                       "group": group, **{m: values[m] for m in METRICS}})
        results[model] = {"parameters": per_seed[0]["parameter_count"],
                          "train_seconds": mean_std([r["training_seconds"] for r in per_seed]),
                          "clean": {m: mean_std([r["clean_selection_score"] if m == "selection_score"
                                                  else r["clean_valid"][m] for r in per_seed]) for m in METRICS},
                          "mean_missing": {m: mean_std([r["missing"]["summary"]["mean_missing"][m]
                                                         for r in per_seed]) for m in METRICS},
                          "robust_score": mean_std([r["missing"]["summary"]["robust_score"]
                                                     for r in per_seed])}
    write_csv(output / "baseline_clean_per_seed.csv", clean_rows)
    write_csv(output / "baseline_missing_per_seed.csv", miss_rows)
    write_csv(output / "baseline_missing_groups_per_seed.csv", group_rows)
    for split in ("clean", "mean_missing"):
        rows = []
        for model in MODELS:
            row = {"model": model}
            for metric in METRICS:
                row[metric + "_mean"] = results[model][split][metric]["mean"]
                row[metric + "_sd"] = results[model][split][metric]["std"]
            rows.append(row)
        filename = "baseline_clean_mean_sd.csv" if split == "clean" else "baseline_missing_mean_sd.csv"
        write_csv(output / filename, rows)
    reference_file = root / "outputs/metrics/b5_p2_multiseed_summary.json"
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    if reference["benchmark_sha256"] != per_seed[0]["benchmark_definition_sha256"]:
        raise ValueError("Historical Q2 and baseline benchmark definitions differ")
    rows = []
    for model in (*MODELS, "B5-P2"):
        source = reference["models"]["B5-P2"] if model == "B5-P2" else results[model]
        row = {"model": model, "source": "historical locked Q2" if model == "B5-P2" else "new train/valid"}
        for split in ("clean", "mean_missing"):
            for metric in METRICS:
                row[f"{split}_{metric}"] = source[split][metric]["mean"]
                row[f"{split}_{metric}_sd"] = source[split][metric]["std"]
        row["robust_score"] = source["robust_score"]["mean"]
        row["robust_score_sd"] = source["robust_score"]["std"]
        rows.append(row)
    write_csv(output / "baseline_comparison_table.csv", rows)
    (output / "baseline_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    def fmt(obj):
        return f"{obj['mean']:.4f} ± {obj['std']:.4f}"
    lines = ["# Q2 public baseline comparison", "",
             "Attachment2 aligned-50 train (3395) and valid (728) only. Seeds 42/43/44; sample SD."
             " All results below are newly trained on this data except the locked historical Q2 reference.", "",
             "| Model | Params | Clean Acc | Clean Macro-F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson | Robust |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        r = results[model]
        lines.append("| " + " | ".join([model, str(r["parameters"]),
            *[fmt(r["clean"][m]) for m in METRICS[:4]],
            *[fmt(r["mean_missing"][m]) for m in METRICS[:4]],
            fmt(r["robust_score"])]) + " |")
    p2 = reference["models"]["B5-P2"]
    lines.append("| " + " | ".join(["Locked Q2 attention residual", "see lock manifest",
        *[fmt(p2["clean"][m]) for m in METRICS[:4]],
        *[fmt(p2["mean_missing"][m]) for m in METRICS[:4]],
        fmt(p2["robust_score"])]) + " |")
    lines += ["", "Selection: clean validation project score; no test-based model choice."
              " Missing evaluation uses the unchanged 54-scenario benchmark definition"
              " (SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`).",
              "", "## Source and adaptation", "",
              "- TFN: independently implemented tensor outer-product fusion from the paper; the inspected public repo has no LICENSE file, so no source was copied. Text uses a packed LSTM; audio and vision use valid-step mean for utterance input.",
              "- MulT: six directed crossmodal attention streams and per-modality memory with sinusoidal positions; padding is excluded from attention keys and final valid-step selection.",
              "- MISA: shared/private decomposition with difference, CMD, and reconstruction terms; precomputed BERT text features replace its tokenizer and BERT inference.",
              "", "These are aligned-feature architecture adaptations, not direct execution of original repository training scripts. Original paper metrics are not mixed with these results.",
              "No outside sentiment data or pretrained task weights were used. Attachment2 test, Attachment3, and Attachment4 were not evaluated for these baselines.", ""]
    lines += ["## Frozen missing benchmark groups", "",
              "Full per-seed modality, ratio, location, and double-modality metrics are in `baseline_missing_groups_per_seed.csv`.", "",
              "| Model | Group | Mean selection score ± SD |", "|---|---|---:|"]
    for model in MODELS:
        for category, group in (("by_modality", "text"), ("by_modality", "audio"),
                                ("by_modality", "vision"), ("double_stress", "text+audio"),
                                ("double_stress", "text+vision"), ("double_stress", "audio+vision")):
            scores = [r["missing"]["summary"][category][group]["selection_score"]
                      for r in [json.loads((base / model / f"metrics_seed{seed}.json").read_text(encoding="utf-8"))
                                for seed in SEEDS]]
            lines.append(f"| {model} | {group} | {fmt(mean_std(scores))} |")
    lines.append("")
    report = "\n".join(lines)
    (output / "Baseline_Experiment_Report.md").write_text(report, encoding="utf-8")
    (output / "baseline_comparison_report.md").write_text(report, encoding="utf-8")
    return results


if __name__ == "__main__":
    summarize(Path(__file__).resolve().parents[1])
