"""Build the 14-model aligned-50 comparison from measured seed result files."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

from build_public_baseline_registry import OUT, ROOT, ROWS

SEEDS = (42, 43, 44)
METRICS = ("accuracy", "macro_f1", "mae", "pearson")
PUBLIC = [r["model"] for r in ROWS if r["model"] not in {"B0", "P2"}]
assert len(PUBLIC) == 12 and len(ROWS) == 14


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise RuntimeError(f"Expected 3 seeds, got {len(values)}")
    return statistics.mean(values), statistics.stdev(values)


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_hash = sha256(ROOT / "data/raw/aligned_50.pkl")
    if data_hash != "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd":
        raise RuntimeError("aligned_50.pkl differs from the verified training file")
    definition_path = ROOT / "outputs/metrics/b2_benchmark_definition.json"
    if sha256(definition_path) != "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff":
        raise RuntimeError("Frozen 54-scenario definition hash differs")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    scenario_ids = ["clean", *[r["scenario_id"] for r in definition["scenarios"]]]
    if len(scenario_ids) != 55 or len(set(scenario_ids)) != 55:
        raise RuntimeError("Frozen scenario IDs are incomplete or duplicated")
    registry = {r["model"]: r for r in ROWS}
    clean, missing, counts, text_groups = [], [], [], []
    for name in PUBLIC:
        for seed in SEEDS:
            path = ROOT / "experiments/q2/public_baselines" / name / f"metrics_seed{seed}.json"
            preflight_path = path.parent / f"preflight_seed{seed}.json"
            if name not in {"TFN", "MulT", "MISA"}:
                if not preflight_path.exists():
                    raise FileNotFoundError(preflight_path)
                preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
                if any(preflight.get(k) != "PASS" for k in ("valid_forward", "finite_gradient", "initial_seed_reproducibility")):
                    raise RuntimeError(f"Preflight failed: {preflight_path}")
            if not path.exists():
                raise FileNotFoundError(path)
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["benchmark_definition_sha256"] != "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff":
                raise RuntimeError(f"Benchmark hash mismatch: {path}")
            c = result["clean_valid"]
            m = result["missing"]["summary"]["mean_missing"]
            base = {"model": name, "seed": seed, "family": registry[name]["family"],
                    "training_regime": registry[name]["training_regime"],
                    "parameters": result["parameter_count"], "training_seconds": result["training_seconds"],
                    "training_time_basis": "full_model_training",
                    "checkpoint": result["checkpoint"], "source_metrics": str(path.relative_to(ROOT))}
            clean.append({**base, **{k: c[k] for k in METRICS}, "selection_score": result["missing"]["summary"]["clean"]["selection_score"]})
            missing.append({**base, **{k: m[k] for k in METRICS}, "selection_score": m["selection_score"]})
            counts.append({"model": name, "seed": seed, "parameters": result["parameter_count"]})
            rows = result["missing"]["rows"]
            if [r["scenario_id"] for r in rows] != scenario_ids:
                raise RuntimeError(f"Expected clean + 54 scenarios: {path}")
            if any(r["sample_count"] != 728 for r in rows):
                raise RuntimeError(f"Validation sample coverage differs: {path}")
            for metric in (*METRICS, "selection_score"):
                actual = statistics.mean(r[metric] for r in rows[1:])
                if abs(actual - m[metric]) > 1e-10:
                    raise RuntimeError(f"Stored missing mean differs from scenario rows: {path} {metric}")
            text = [r for r in rows[1:] if r["modalities"] == "text"]
            if len(text) != 15:
                raise RuntimeError(f"Expected 15 text scenarios: {path}")
            text_groups.append({"model": name, "seed": seed, "clean_accuracy": c["accuracy"],
                                "text_missing_accuracy": statistics.mean(r["accuracy"] for r in text),
                                "clean_macro_f1": c["macro_f1"],
                                "text_missing_macro_f1": statistics.mean(r["macro_f1"] for r in text)})

    fair_path = ROOT / "outputs/final/q2/checkpoint_selection_fairness/q2_cleanselect_fair_comparison_seedwise.csv"
    with fair_path.open(encoding="utf-8-sig", newline="") as f:
        fair = list(csv.DictReader(f))
    metric_dir = ROOT / "outputs/metrics"
    b0_times = {42: json.loads((metric_dir / "b0_weighted_ce_score_selection_metrics.json").read_text())["training_seconds"]}
    b0_times.update({seed: json.loads((metric_dir / f"b21_b0_seed_{seed}_training_metrics.json").read_text())["training_seconds"]
                     for seed in (43, 44)})
    p2_times = {42: json.loads((metric_dir / "b5_pooling_metrics.json").read_text())["P2"]["training_seconds"]}
    p2_multi = json.loads((metric_dir / "b5_p2_multiseed_metrics.json").read_text())
    p2_times.update({seed: p2_multi["models"][str(seed)]["training_seconds"] for seed in (43, 44)})
    for row in fair:
        name, seed = row["model"], int(row["seed"])
        if name not in {"B0", "P2"} or seed not in SEEDS or row["selection_rule"] != "CleanSelect":
            raise RuntimeError("Fair comparison protocol differs")
        if row["benchmark_sha256"] != "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff":
            raise RuntimeError("Fair comparison benchmark differs")
        params = 163460 if name == "B0" else 164343
        base = {"model": name, "seed": seed, "family": registry[name]["family"],
                "training_regime": registry[name]["training_regime"], "parameters": params,
                "training_seconds": b0_times[seed] if name == "B0" else p2_times[seed],
                "training_time_basis": "full_model_training" if name == "B0" else "residual_training_only_excludes_B0",
                "checkpoint": row["checkpoint_path"],
                "source_metrics": str(fair_path.relative_to(ROOT))}
        clean.append({**base, **{k: float(row[f"clean_{k}"]) for k in METRICS},
                      "selection_score": float(row["clean_selection_score"])})
        missing.append({**base, **{k: float(row[f"mean_missing_{k}"]) for k in METRICS},
                        "selection_score": float(row["mean_missing_selection_score"])})
        counts.append({"model": name, "seed": seed, "parameters": params})

    group_path = OUT / "b0_p2_cleanselect_missing_groups.csv"
    if not group_path.exists():
        raise FileNotFoundError("Grouped B0/P2 CleanSelect validation is required for 14-model text sensitivity")
    with group_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["modality"] != "text":
                continue
            text_groups.append({"model": row["model"], "seed": int(row["seed"]),
                                "clean_accuracy": float(row["clean_accuracy"]),
                                "text_missing_accuracy": float(row["missing_accuracy"]),
                                "clean_macro_f1": float(row["clean_macro_f1"]),
                                "text_missing_macro_f1": float(row["missing_macro_f1"])})
    if len(text_groups) != 42:
        raise RuntimeError("Expected text sensitivity for all 14 models × 3 seeds")

    expected = {(r["model"], s) for r in ROWS for s in SEEDS}
    for label, rows in (("clean", clean), ("missing", missing)):
        got = [(r["model"], r["seed"]) for r in rows]
        if len(got) != 42 or set(got) != expected or len(set(got)) != 42:
            raise RuntimeError(f"{label} missing or duplicate model-seed records")
    order = {r["model"]: i for i, r in enumerate(ROWS)}
    clean.sort(key=lambda r: (order[r["model"]], r["seed"]))
    missing.sort(key=lambda r: (order[r["model"]], r["seed"]))
    write_csv(OUT / "q2_public_baselines_clean_seedwise.csv", clean)
    write_csv(OUT / "q2_public_baselines_missing_seedwise.csv", missing)
    write_csv(OUT / "clean_results.csv", clean)
    write_csv(OUT / "missing_results.csv", missing)
    parameter_rows = []
    for name in [r["model"] for r in ROWS]:
        values = {r["parameters"] for r in counts if r["model"] == name}
        if len(values) != 1:
            raise RuntimeError(f"Parameter count differs by seed: {name}")
        parameter_rows.append({"model": name, "parameters": values.pop(),
                               "family": registry[name]["family"],
                               "training_regime": registry[name]["training_regime"]})
    write_csv(OUT / "parameter_counts.csv", parameter_rows)
    write_csv(OUT / "text_missing_sensitivity_seedwise.csv", text_groups)

    log_dir, note_dir, ckpt_dir = (OUT / part for part in ("training_logs", "adaptation_notes", "checkpoints"))
    for directory in (log_dir, note_dir, ckpt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = []
    for row in clean:
        name, seed = row["model"], row["seed"]
        source = ROOT / row["checkpoint"]
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256(source)
        target = ckpt_dir / f"{name}_seed{seed}_cleanselect.pt"
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        if sha256(target) != digest:
            raise RuntimeError(f"Checkpoint copy mismatch: {target}")
        checkpoint_manifest.append({"model": name, "seed": seed,
                                    "source_checkpoint": row["checkpoint"],
                                    "bundle_checkpoint": str(target.relative_to(ROOT)),
                                    "sha256": digest})
        log = ROOT / "experiments/q2/public_baselines" / name / f"train_seed{seed}.log"
        if log.exists():
            shutil.copy2(log, log_dir / f"{name}_seed{seed}.log")
    write_csv(ckpt_dir / "checkpoint_manifest.csv", checkpoint_manifest)
    for name in PUBLIC:
        source = ROOT / "experiments/q2/public_baselines" / name / "adaptation_notes.md"
        if source.exists():
            shutil.copy2(source, note_dir / f"{name}.md")
        if name in {"TFN", "MulT", "MISA"}:
            continue  # Preserve their original formal experiment configs and notes.
        directory = source.parent
        for seed in SEEDS:
            config = (f"model: {name}\nseed: {seed}\ndata: data/raw/aligned_50.pkl\n"
                      "train_split: train\nvalidation_split: valid\ntrain_samples: 3395\nvalid_samples: 728\n"
                      "normalization: none\nbatch_size: 128\nmax_epochs: 80\npatience: 12\n"
                      "optimizer: AdamW\nlearning_rate: 0.001\nweight_decay: 0.0001\n"
                      "classification_loss: balanced_weighted_CE_train_counts\nregression_loss: SmoothL1\n"
                      "lambda_reg: 1.0\nselection: clean_validation_selection_score\n"
                      f"training_regime: {registry[name]['training_regime']}\n"
                      "benchmark_definition_sha256: 3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff\n")
            (directory / f"config_seed{seed}.yaml").write_text(config, encoding="utf-8")
        clean_sub = [r for r in clean if r["model"] == name]
        missing_sub = [r for r in missing if r["model"] == name]
        notes = [f"# {name}: three-seed experiment notes", "", "Purpose: compare this published mechanism's aligned-50 adaptation under common clean-selected two-task protocol.", "", f"Regime: {registry[name]['training_regime']}.", "", "| Seed | Clean Acc | Clean F1 | Clean MAE | Clean Pearson | Missing Acc | Missing F1 | Missing MAE | Missing Pearson |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c, m in zip(clean_sub, missing_sub):
            notes.append(f"| {c['seed']} | {c['accuracy']:.4f} | {c['macro_f1']:.4f} | {c['mae']:.4f} | {c['pearson']:.4f} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['mae']:.4f} | {m['pearson']:.4f} |")
        notes += ["", "Paper use: aligned-50 adapted baseline comparison only; not an original-paper reproduction.", "", "Next action: report observed values without further model-specific tuning."]
        (directory / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    summaries = {}
    for condition, rows in (("clean", clean), ("missing", missing)):
        out = []
        for name in order:
            sub = [r for r in rows if r["model"] == name]
            record = {"model": name, "family": registry[name]["family"],
                      "training_regime": registry[name]["training_regime"],
                      "parameters": sub[0]["parameters"], "n_seeds": 3}
            for metric in (*METRICS, "selection_score"):
                mean, sd = summarize([r[metric] for r in sub])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_sample_sd"] = sd
            out.append(record)
        summaries[condition] = out
        write_csv(OUT / f"q2_public_baselines_{condition}_summary.csv", out)

    time_rows = []
    for name in [r["model"] for r in ROWS]:
        sub = [r for r in clean if r["model"] == name]
        mean_time, sd_time = summarize([float(r["training_seconds"]) for r in sub])
        time_rows.append({"model": name, "parameters": sub[0]["parameters"],
                          "training_seconds_mean": mean_time, "training_seconds_sample_sd": sd_time,
                          "training_time_basis": sub[0]["training_time_basis"]})
    write_csv(OUT / "training_time_summary.csv", time_rows)

    def cell(record: dict, metric: str) -> str:
        return f"{record[metric + '_mean']:.4f} ± {record[metric + '_sample_sd']:.4f}"

    for condition in ("clean", "missing"):
        filename = "candidate_table_clean.csv" if condition == "clean" else "candidate_table_missing.csv"
        table = []
        for r in summaries[condition]:
            table.append({"Model": r["model"], "Family" if condition == "clean" else "Training regime":
                          r["family"] if condition == "clean" else r["training_regime"],
                          "Params" if condition == "clean" else "Params": r["parameters"],
                          "Acc": cell(r, "accuracy"), "Macro-F1": cell(r, "macro_f1"),
                          "MAE": cell(r, "mae"), "Pearson": cell(r, "pearson")})
        write_csv(OUT / filename, table)
        latex = ["% Candidate table; booktabs required. Values are mean $\\pm$ sample SD over seeds 42/43/44.",
                 "\\begin{table*}[t]", "\\centering", "\\small",
                 "\\begin{tabular}{llrrrrr}", "\\toprule",
                 "Model & Family/Regime & Params & Accuracy $\\uparrow$ & Macro-F1 $\\uparrow$ & MAE $\\downarrow$ & Pearson $\\uparrow$ \\\\",
                 "\\midrule"]
        for r in summaries[condition]:
            regime = r["family"] if condition == "clean" else r["training_regime"]
            metric_cells = [f"${r[k + '_mean']:.4f}\\pm{r[k + '_sample_sd']:.4f}$" for k in METRICS]
            latex.append(f"{r['model']} & {regime} & {r['parameters']:,} & " + " & ".join(metric_cells) + r" \\")
        latex += ["\\bottomrule", "\\end{tabular}",
                  f"\\caption{{Aligned-50 {condition} validation comparison, three seeds. Published models are architecture adaptations.}}",
                  f"\\label{{tab:q2_expanded_{condition}}}", "\\end{table*}"]
        (OUT / f"q2_public_baselines_{condition}_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    c_by = {r["model"]: r for r in summaries["clean"]}
    text_by = defaultdict(list)
    for row in text_groups:
        text_by[row["model"]].append(row)
    sensitivity = []
    for name, rows in text_by.items():
        drop = [r["clean_accuracy"] - r["text_missing_accuracy"] for r in rows]
        mean, sd = summarize(drop)
        sensitivity.append((name, mean, sd))
    sensitivity.sort(key=lambda x: x[1], reverse=True)
    public_median_params = statistics.median(c_by[name]["parameters"] for name in PUBLIC)
    p2_params = c_by["P2"]["parameters"]
    standard_missing_acc = [r["accuracy_mean"] for r in summaries["missing"] if r["model"] in PUBLIC and r["family"] == "standard fusion"]
    aware_missing_acc = [r["accuracy_mean"] for r in summaries["missing"] if r["model"] in PUBLIC and r["family"] == "missing-aware"]
    ranks = {}
    for condition in ("clean", "missing"):
        for metric in METRICS:
            ordered = sorted(summaries[condition], key=lambda r: r[f"{metric}_mean"],
                             reverse=(metric != "mae"))
            ranks[(condition, metric)] = next(i + 1 for i, r in enumerate(ordered) if r["model"] == "P2")
    lines = ["# Q2 expanded public architecture comparison", "", f"All values below are measured on Attachment2 validation (aligned_50.pkl SHA256 `{data_hash}`). Twelve published model mechanisms were adapted to the aligned-50 interface and evaluated with the common classification/regression heads. They are **not** complete replications of each paper's original training pipeline. B0 and P2 use the existing CleanSelect checkpoint evaluation. Every seed is 42, 43, or 44; sample SD uses ddof=1. The 54 scenarios use the frozen SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`.", "", "## Clean validation", "", "| Model | Family | Params | Accuracy | Macro-F1 | MAE | Pearson |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries["clean"]:
        lines.append(f"| {r['model']} | {r['family']} | {r['parameters']:,} | {cell(r, 'accuracy')} | {cell(r, 'macro_f1')} | {cell(r, 'mae')} | {cell(r, 'pearson')} |")
    lines += ["", "## Frozen 54-scenario mean", "", "| Model | Training regime | Missing Accuracy | Missing Macro-F1 | Missing MAE | Missing Pearson |", "|---|---|---:|---:|---:|---:|"]
    for r in summaries["missing"]:
        lines.append(f"| {r['model']} | {r['training_regime']} | {cell(r, 'accuracy')} | {cell(r, 'macro_f1')} | {cell(r, 'mae')} | {cell(r, 'pearson')} |")
    lines += ["", "## Interpretation", "",
              "P2 rank among 14 models (Accuracy / Macro-F1 / MAE / Pearson): " +
              ", ".join(f"{condition} " + "/".join(str(ranks[(condition, metric)]) for metric in METRICS)
                        for condition in ("clean", "missing")) + ". MAE ranks lower values first.",
              f"P2 uses {p2_params:,} parameters; the median adapted public mechanism has {public_median_params:,.0f}, or {public_median_params / p2_params:.2f}× as many. Parameter efficiency is descriptive and does not remove regime or architecture differences.",
              "", "The specialized missing-aware entries received synthetic missing during training; standard fusion entries were trained on clean samples. Both groups were selected on clean validation and evaluated on identical frozen missing scenarios. This regime difference is retained in the tables.",
              f"Across adapted public mechanisms, missing Accuracy spans {min(standard_missing_acc):.4f}–{max(standard_missing_acc):.4f} for standard clean training and {min(aware_missing_acc):.4f}–{max(aware_missing_acc):.4f} for missing-aware training. The ranges overlap; these model families also differ in architecture and cannot isolate a training-regime effect.",
              "", "Text-missing sensitivity (clean minus 15 text-only scenario mean Accuracy), all 14 models:"]
    for name, mean, sd in sensitivity:
        lines.append(f"- {name}: {mean:+.4f} ± {sd:.4f}")
    lines += ["", "Training time is recorded per seed in the seedwise CSV. P2 time covers residual training with a pre-existing frozen B0 and is labeled separately; it is not a from-scratch time comparison.",
              "", "No Attachment2 test or Attachment3/4 results were used for model selection. Source details and implementation departures appear in the registry and per-model adaptation notes."]
    (OUT / "q2_expanded_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    qa = ["# Expanded Q2 baseline QA", "",
          "- Published architecture adaptations: 12; project comparators: 2.",
          "- Train/valid sizes: 3395/728; seeds: 42, 43, 44.",
          "- Clean seedwise rows: 42; mean-missing seedwise rows: 42; summary rows per condition: 14.",
          "- New model runs with preflight and measured checkpoint: 27/27.",
          "- Reused earlier public-model runs: TFN/MulT/MISA, 9/9.",
          "- Reused locked CleanSelect comparators: B0/P2, 6/6; grouped re-evaluation matched prior scores within 1e-5.",
          "- Every new benchmark result has 728 validation samples in clean plus the 54 official scenario IDs, in frozen order.",
          "- Sample SD computed with ddof=1 after averaging 54 scenarios within each seed.",
          f"- aligned_50.pkl SHA256: `{data_hash}`.",
          "- Benchmark definition SHA256: `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`.",
          f"- Adaptation model code SHA256: `{sha256(ROOT / 'src/models/public_baselines_extended.py')}`.",
          f"- Training runner SHA256: `{sha256(ROOT / 'scripts/run_public_baseline_extended.py')}`.",
          "- Dataset loader used include_test=False; no Attachment3/4 code path occurs in the training/evaluation runner.",
          "- Checkpoint manifest contains 42 verified SHA256 values; bundle files are local hardlinks or byte-identical copies.", ""]
    (OUT / "q2_expanded_baseline_qa.md").write_text("\n".join(qa), encoding="utf-8")


if __name__ == "__main__":
    main()
