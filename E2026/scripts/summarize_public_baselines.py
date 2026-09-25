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
            scenario_rows = result["missing"]["rows"][1:]
            for modality in ("text", "audio", "vision"):
                for category, keys in (("modality_rho", ("0.1", "0.2", "0.3", "0.4", "0.5")),
                                       ("modality_location", ("early", "middle", "late"))):
                    for key in keys:
                        subset = [r for r in scenario_rows if r["modalities"] == modality and
                                  (str(r["rho"]) if category == "modality_rho" else r["location"]) == key]
                        expected_count = 3 if category == "modality_rho" else 5
                        if len(subset) != expected_count:
                            raise ValueError(f"unexpected scenario count for {modality}/{category}/{key}")
                        group_rows.append({"model": model, "seed": seed, "category": category,
                                           "group": f"{modality}/{key}",
                                           **{m: statistics.mean(r[m] for r in subset) for m in METRICS}})
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
    group_summary_rows = []
    for model in MODELS:
        for category, group in sorted({(r["category"], r["group"]) for r in group_rows if r["model"] == model}):
            members = [r for r in group_rows if (r["model"], r["category"], r["group"]) ==
                       (model, category, group)]
            if len(members) != 3:
                raise ValueError(f"expected 3 seeds for {model}/{category}/{group}")
            row = {"model": model, "category": category, "group": group}
            for metric in METRICS:
                values = mean_std([r[metric] for r in members])
                row[metric + "_mean"] = values["mean"]
                row[metric + "_sd"] = values["std"]
            group_summary_rows.append(row)
    write_csv(output / "baseline_missing_groups_mean_sd.csv", group_summary_rows)
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
    for model in (*MODELS, "B0-WCE", "B5-P2"):
        source = reference["models"][model] if model in ("B0-WCE", "B5-P2") else results[model]
        row = {"model": model, "source": "historical locked Q2" if model in ("B0-WCE", "B5-P2") else "new train/valid"}
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
             " The TFN, MulT, and MISA results were newly trained; B0-WCE and P2 are historical locked references.", "",
             "| Model | Params | Clean Acc | Clean Macro-F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson | Robust |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        r = results[model]
        lines.append("| " + " | ".join([model, str(r["parameters"]),
            *[fmt(r["clean"][m]) for m in METRICS[:4]],
            *[fmt(r["mean_missing"][m]) for m in METRICS[:4]],
            fmt(r["robust_score"])]) + " |")
    b0 = reference["models"]["B0-WCE"]
    lines.append("| " + " | ".join(["Historical B0-WCE", "see lock manifest",
        *[fmt(b0["clean"][m]) for m in METRICS[:4]],
        *[fmt(b0["mean_missing"][m]) for m in METRICS[:4]],
        fmt(b0["robust_score"])]) + " |")
    p2 = reference["models"]["B5-P2"]
    lines.append("| " + " | ".join(["Locked Q2 attention residual", "see lock manifest",
        *[fmt(p2["clean"][m]) for m in METRICS[:4]],
        *[fmt(p2["mean_missing"][m]) for m in METRICS[:4]],
        fmt(p2["robust_score"])]) + " |")
    lines += ["", "Baseline checkpoint selection used the clean-validation project score. The historical B0-WCE checkpoint was also clean-score selected; the historical P2 checkpoint was robust-score selected. Thus the P2 missing-score comparison has a checkpoint-selection advantage and should be read descriptively. No test-based model choice occurred."
              " Missing evaluation uses the unchanged 54-scenario benchmark definition"
              " (SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`).",
              "", "## Source and adaptation", "",
              "- TFN: [reference repo](https://github.com/Justin1904/TensorFusionNetworks) `ef0e78b5583159de9b74ef2cdef6031bd9f94b37`; independently implemented tensor outer-product fusion from the paper. The inspected repo has no LICENSE file, so no source was copied. Text uses a packed LSTM; audio and vision use valid-step mean for utterance input.",
              "- MulT: [official repo](https://github.com/yaohungt/Multimodal-Transformer) `a670936824ee722c8494fd98d204977a1d663c7a` (MIT); six directed crossmodal attention streams and per-modality memory with sinusoidal positions; padding is excluded from attention keys and final valid-step selection.",
              "- MISA: [official repo](https://github.com/declare-lab/MISA) `ec42faddde0d210cf7368aebf2118fe9570e7102` (MIT); shared/private decomposition with difference, CMD, and reconstruction terms. Precomputed BERT text features replace its tokenizer and BERT inference; audio and vision use valid-step means.",
              "", "These are aligned-feature architecture adaptations, not direct execution of original repository training scripts. Original paper metrics are not mixed with these results.",
              "No outside sentiment data or pretrained task weights were used. Attachment2 test, Attachment3, and Attachment4 were not evaluated for these baselines.", ""]
    lines += ["## Training cost and selection", "", "| Model | Mean training seconds ± SD | Best epochs by seed 42/43/44 |", "|---|---:|---|" ]
    for model in MODELS:
        epochs = [json.loads((base / model / f"metrics_seed{seed}.json").read_text(encoding="utf-8"))["best_clean_epoch"] for seed in SEEDS]
        lines.append(f"| {model} | {fmt(results[model]['train_seconds'])} | {' / '.join(map(str, epochs))} |")
    lines.append("")
    lines += ["## Frozen missing benchmark groups", "",
              "Per-seed and three-seed mean/SD metrics for modality, ratio, location, modality × ratio, modality × location, and double-modality conditions are in `baseline_missing_groups_per_seed.csv` and `baseline_missing_groups_mean_sd.csv`.", "",
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
    lines += ["## Reproducibility and data boundary", "",
              "Training host: NVIDIA GeForce RTX 3090 (24 GB), PyTorch 2.14.0+cu130."
              " Attachment2 `aligned_50.pkl` SHA256 on both local and server:"
              " `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`."
              " Model adapter SHA256: `f7e5ee67c265d52f9d8d8086b0d1e047d6c106ac8f1d59d7305b72c4b7eafaa8`;"
              " training script SHA256: `e10170957b5ba3c790e4bf54faddde485f8e07fe06ba7f4d76f000b44765ccd2`.",
              "", "The pickle is a monolithic container, so it is deserialized as a whole."
              " The loader constructs only train/valid datasets (`include_test=False`); test samples and labels are never indexed, evaluated, or used for selection."
              " All class weights use train counts only. The benchmark definition SHA is verified before every run."
              " Nine per-seed checkpoints, logs, and metrics are preserved locally; checkpoint hashes are in `baseline_checkpoint_manifest.json`.", ""]
    report = "\n".join(lines)
    (output / "Baseline_Experiment_Report.md").write_text(report, encoding="utf-8")
    (output / "baseline_comparison_report.md").write_text(report, encoding="utf-8")
    return results


if __name__ == "__main__":
    summarize(Path(__file__).resolve().parents[1])
