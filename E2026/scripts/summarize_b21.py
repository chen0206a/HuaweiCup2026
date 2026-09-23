"""Summarize paired training-seed stability on the unchanged B2 valid benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import yaml

from scripts.run_b21_seed import DEFINITION_SHA256, MODELS, ROOT, assert_frozen_benchmark

METRICS = ROOT / "outputs" / "metrics"
FIELDS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
PAIRS = {
    "B2-B0 minus B0-WCE": ("B2-B0-BlockMask", "B0-WCE"),
    "B2-B1 minus B1-WCE": ("B2-B1-BlockMask", "B1-WCE"),
    "B0 main line minus B1 main line": ("B2-B0-BlockMask", "B2-B1-BlockMask"),
    "B0-WCE minus B1-WCE": ("B0-WCE", "B1-WCE"),
}


def prior_seed_42() -> dict[str, dict]:
    """Reuse the actual historical training seed, without relabeling it."""
    for _, (_, base, _) in MODELS.items():
        if yaml.safe_load((ROOT / base).read_text(encoding="utf-8"))["training"]["seed"] != 42:
            raise RuntimeError("historical config is not seed 42")
    comparison = json.loads((METRICS / "b2_summary.json").read_text(encoding="utf-8"))
    epochs = {
        "B0-WCE": json.loads((METRICS / "b0_weighted_ce_score_selection_metrics.json").read_text())["best_selection_score"]["epoch"],
        "B1-WCE": json.loads((METRICS / "b1_weighted_ce_metrics.json").read_text())["best_selection_score"]["epoch"],
        "B2-B0-BlockMask": json.loads((METRICS / "b2_b0_metrics.json").read_text())["best_robust_epoch"],
        "B2-B1-BlockMask": json.loads((METRICS / "b2_b1_metrics.json").read_text())["best_robust_epoch"],
    }
    return {name: {"model": name, "seed": 42, "benchmark_sha256": DEFINITION_SHA256,
                   "best_epoch": int(epochs[name]), "validation": comparison[name]["summary"],
                   "source": "B2 original frozen checkpoint evaluation"}
            for name in MODELS}


def load_results(seeds: list[int]) -> dict[int, dict[str, dict]]:
    assert_frozen_benchmark()
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct training seeds are required")
    result = {}
    for seed in seeds:
        if seed == 42:
            result[seed] = prior_seed_42()
            continue
        result[seed] = {}
        for name, (_, _, short) in MODELS.items():
            path = METRICS / f"b21_{short}_seed_{seed}_validation.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["seed"] != seed or value["model"] != name or value["benchmark_sha256"] != DEFINITION_SHA256:
                raise RuntimeError(f"wrong seed/model/benchmark in {path}")
            result[seed][name] = value
    return result


def aggregate(results: dict[int, dict[str, dict]]) -> dict:
    seeds = list(results)
    grouped = {}
    for name in MODELS:
        grouped[name] = {}
        for section in ("clean", "mean_missing"):
            grouped[name][section] = {}
            for field in FIELDS:
                values = [results[seed][name]["validation"][section][field] for seed in seeds]
                grouped[name][section][field] = {"mean": mean(values), "std": stdev(values), "values": values}
        values = [results[seed][name]["validation"]["robust_score"] for seed in seeds]
        grouped[name]["robust_score"] = {"mean": mean(values), "std": stdev(values), "values": values}
    contrasts = {}
    for title, (left, right) in PAIRS.items():
        contrasts[title] = {}
        for section, field in (("clean", "selection_score"),
                               ("mean_missing", "selection_score"),
                               ("robust_score", None)):
            differences = []
            for seed in seeds:
                a = results[seed][left]["validation"]
                b = results[seed][right]["validation"]
                differences.append((a[section][field] - b[section][field]) if field
                                   else a[section] - b[section])
            contrasts[title][section] = {
                "seed_deltas": dict(zip(map(str, seeds), differences)),
                "mean_delta": mean(differences), "std_delta": stdev(differences),
                "direction_consistent": all(x > 0 for x in differences) or all(x < 0 for x in differences),
            }
    return {"training_seeds": seeds, "benchmark_sha256": DEFINITION_SHA256,
            "std_definition": "sample standard deviation, ddof=1", "models": grouped,
            "paired_contrasts": contrasts}


def write_outputs(results: dict[int, dict[str, dict]], combined: dict) -> None:
    prefix = "b21_multiseed"
    (METRICS / f"{prefix}_summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for seed, models in results.items():
        for name in MODELS:
            value = models[name]
            v = value["validation"]
            rows.append({"model": name, "seed": seed,
                         "clean_score": v["clean"]["selection_score"],
                         "missing_score": v["mean_missing"]["selection_score"],
                         "robust_score": v["robust_score"],
                         "best_epoch": value["best_epoch"]})
    with (METRICS / f"{prefix}_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def fmt(x: dict) -> str:
        return f"{x['mean']:.4f} ± {x['std']:.4f}"
    lines = ["# Q2 B2.1 training-seed stability", "",
             f"Training seeds: {', '.join(map(str, combined['training_seeds']))}. Sample standard deviation (ddof=1).",
             "Frozen 54-scenario benchmark; attachment2 test and attachment3 unused.", "",
             "| Model | Section | Accuracy | Macro-F1 | MAE | Pearson | Selection score |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, grouped in combined["models"].items():
        for section in ("clean", "mean_missing"):
            lines.append(f"| {name} | {section} | " + " | ".join(fmt(grouped[section][field]) for field in FIELDS) + " |")
        lines.append(f"| {name} | robust score | — | — | — | — | {fmt(grouped['robust_score'])} |")
    lines += ["", "## Paired score differences", "",
              "| Contrast | Score | Per-seed deltas | Same direction? |",
              "|---|---|---|---|"]
    for title, sections in combined["paired_contrasts"].items():
        for section, value in sections.items():
            delta = ", ".join(f"{seed}: {amount:+.4f}" for seed, amount in value["seed_deltas"].items())
            lines.append(f"| {title} | {section} | {delta} | {value['direction_consistent']} |")
    lines += ["", "Individual model/seed scores and best epochs: `b21_multiseed_runs.csv`."]
    (METRICS / f"{prefix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs=3, type=int, required=True)
    args = parser.parse_args()
    results = load_results(args.seeds)
    combined = aggregate(results)
    write_outputs(results, combined)
    print(json.dumps({"seeds": args.seeds, "paired_contrasts": combined["paired_contrasts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
