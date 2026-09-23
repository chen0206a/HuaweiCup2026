"""Aggregate B3.1/B3.2 fixed-alpha results across training seeds 42/43/44."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
SEEDS = (42, 43, 44)
ALPHAS = ("0.0", "0.25", "0.5", "0.75", "1.0")
FOUR = ("accuracy", "macro_f1", "mae", "pearson")
FIVE = (*FOUR, "selection_score")


def load(seed: int) -> dict:
    path = (METRICS / "b31_alpha_sweep.json" if seed == 42 else
            METRICS / f"b32_alpha_sweep_seed_{seed}.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["training_seed"] != seed or value["benchmark_seed"] != 20260923:
        raise RuntimeError(f"seed/benchmark mismatch in {path}")
    if tuple(value["alphas"]) != ALPHAS:
        raise RuntimeError(f"alpha grid mismatch in {path}")
    expected = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
    if value["benchmark_sha256"] != expected:
        raise RuntimeError(f"frozen benchmark mismatch in {path}")
    return value


def vision_metrics(value: dict, group: str) -> dict:
    source = value["vision_all_zero"][group]
    result = {key: source[key] for key in FOUR}
    result["selection_score"] = 0.25 * result["accuracy"] + 0.25 * result["macro_f1"] + \
        0.25 * (1.0 - result["mae"] / 6.0) + 0.25 * (result["pearson"] + 1.0) / 2.0
    return result


def get_metrics(sweep: dict, alpha: str, segment: str, group: str | None = None) -> dict:
    record = sweep["alphas"][alpha]
    if segment == "vision_all_zero":
        return vision_metrics(record, group or "clean")
    if segment in ("clean", "mean_missing"):
        return record[segment]
    if segment == "robust_score":
        return {"robust_score": record[segment]}
    if segment == "modality":
        return record["by_modality"][group]
    raise KeyError(segment)


def moments(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}


def main() -> None:
    runs = {seed: load(seed) for seed in SEEDS}
    frozen_checks = {}
    alpha0_checks = {}
    for seed in SEEDS:
        training_file = (METRICS / "b31_training_metrics.json" if seed == 42 else
                        METRICS / f"b32_training_metrics_seed_{seed}.json")
        equivalence_file = (METRICS / "b32_optimized_validation_verification_seed42.json" if seed == 42 else
                            METRICS / f"b32_alpha0_equivalence_seed_{seed}.json")
        frozen_checks[str(seed)] = json.loads(training_file.read_text(encoding="utf-8"))["frozen_backbone_check"]
        alpha0_checks[str(seed)] = json.loads(equivalence_file.read_text(encoding="utf-8"))
        if not frozen_checks[str(seed)]["passed"] or not frozen_checks[str(seed)]["matches_source_checkpoint"]:
            raise RuntimeError(f"seed-{seed} frozen B0 parameter check failed")
        if not alpha0_checks[str(seed)]["passed"]:
            raise RuntimeError(f"seed-{seed} alpha=0 B0 equivalence failed")
    results = {}
    for alpha in ALPHAS:
        results[alpha] = {}
        for segment in ("clean", "mean_missing", "robust_score"):
            metric_names = ("robust_score",) if segment == "robust_score" else FIVE
            results[alpha][segment] = {
                metric: moments([get_metrics(runs[s], alpha, segment)[metric] for s in SEEDS])
                for metric in metric_names
            }
        results[alpha]["modality"] = {
            modality: {metric: moments([
                get_metrics(runs[s], alpha, "modality", modality)[metric] for s in SEEDS
            ]) for metric in FIVE}
            for modality in ("text", "audio", "vision")
        }
        results[alpha]["vision_all_zero"] = {
            group: {metric: moments([
                get_metrics(runs[s], alpha, "vision_all_zero", group)[metric] for s in SEEDS
            ]) for metric in FIVE}
            for group in ("clean", "mean_missing")
        }

    paired = {}
    for alpha in ALPHAS[1:]:
        paired[alpha] = {}
        for segment in ("clean", "mean_missing", "robust_score"):
            metric_names = ("robust_score",) if segment == "robust_score" else FIVE
            paired[alpha][segment] = {}
            for metric in metric_names:
                deltas = [get_metrics(runs[s], alpha, segment)[metric] -
                          get_metrics(runs[s], "0.0", segment)[metric] for s in SEEDS]
                improves = [d < 0 if metric == "mae" else d > 0 for d in deltas]
                paired[alpha][segment][metric] = {
                    **moments(deltas), "per_seed_delta": deltas,
                    "improvement_direction_count": sum(improves),
                    "improved_all_seeds": all(improves),
                }
        for group in ("text", "audio", "vision"):
            paired[alpha].setdefault("modality", {})[group] = {}
            for metric in FIVE:
                deltas = [get_metrics(runs[s], alpha, "modality", group)[metric] -
                          get_metrics(runs[s], "0.0", "modality", group)[metric] for s in SEEDS]
                improves = [d < 0 if metric == "mae" else d > 0 for d in deltas]
                paired[alpha]["modality"][group][metric] = {
                    **moments(deltas), "per_seed_delta": deltas,
                    "improvement_direction_count": sum(improves),
                    "improved_all_seeds": all(improves),
                }
        for group in ("clean", "mean_missing"):
            paired[alpha].setdefault("vision_all_zero", {})[group] = {}
            for metric in FIVE:
                deltas = [get_metrics(runs[s], alpha, "vision_all_zero", group)[metric] -
                          get_metrics(runs[s], "0.0", "vision_all_zero", group)[metric] for s in SEEDS]
                improves = [d < 0 if metric == "mae" else d > 0 for d in deltas]
                paired[alpha]["vision_all_zero"][group][metric] = {
                    **moments(deltas), "per_seed_delta": deltas,
                    "improvement_direction_count": sum(improves),
                    "improved_all_seeds": all(improves),
                }

    run_rows = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            s = runs[seed]["alphas"][alpha]
            run_rows.append({"seed": seed, "alpha": float(alpha),
                             "clean_score": s["clean"]["selection_score"],
                             "mean_missing_score": s["mean_missing"]["selection_score"],
                             "robust_score": s["robust_score"]})
    result = {"training_seeds": SEEDS, "benchmark_seed": 20260923,
              "benchmark_sha256": runs[42]["benchmark_sha256"],
              "alpha_grid": [float(a) for a in ALPHAS],
              "frozen_backbone_checks": frozen_checks,
              "alpha0_equivalence_checks": alpha0_checks,
              "summary_mean_sample_std": results, "paired_delta_vs_alpha0": paired,
              "per_seed_alpha_scores": run_rows,
              "per_seed_best_alpha_by_robust_score": {
                  str(seed): max(ALPHAS, key=lambda alpha: runs[seed]["alphas"][alpha]["robust_score"])
                  for seed in SEEDS},
              "robust_gain_consistent_across_all_seeds": {
                  alpha: paired[alpha]["robust_score"]["robust_score"]["improved_all_seeds"]
                  for alpha in ALPHAS[1:]},
              "mean_missing_gain_consistent_across_all_seeds": {
                  alpha: paired[alpha]["mean_missing"]["selection_score"]["improved_all_seeds"]
                  for alpha in ALPHAS[1:]},
              "test_or_attachment3_used": False}
    (METRICS / "b32_multiseed_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (METRICS / "b32_multiseed_report.md").write_text(render_report(result), encoding="utf-8")


def fmt(stats: dict) -> str:
    return f"{stats['mean']:.4f} ± {stats['sample_std']:.4f}"


def render_report(result: dict) -> str:
    sm, paired = result["summary_mean_sample_std"], result["paired_delta_vs_alpha0"]
    lines = ["# B3.2 frozen reconstruction multi-seed stability", "",
             "Seeds 42/43/44; benchmark seed 20260923 and SHA256 unchanged. Validation only; no test or attachment3 data.",
             "All displayed uncertainty is mean ± sample standard deviation (ddof=1). Alpha is fixed globally, never selected per seed.", "",
             "## Clean, mean-missing, robust", "",
             "| Alpha | Clean Acc | Clean Macro-F1 | Clean MAE | Clean Pearson | Clean score | Missing Acc | Missing Macro-F1 | Missing MAE | Missing Pearson | Missing score | Robust score |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        c, m, r = sm[alpha]["clean"], sm[alpha]["mean_missing"], sm[alpha]["robust_score"]
        lines.append(f"| {alpha} | " + " | ".join(fmt(x) for x in
                     (c["accuracy"], c["macro_f1"], c["mae"], c["pearson"], c["selection_score"],
                      m["accuracy"], m["macro_f1"], m["mae"], m["pearson"], m["selection_score"],
                      r["robust_score"])) + " |")
    lines += ["", "## Paired deltas against alpha=0", "",
              "Each cell is mean paired delta ± sample std; count is seeds whose delta improves the metric (MAE improves when negative).", "",
              "| Alpha | Segment | Metric | Delta | Direction |", "|---:|---|---|---:|---:|"]
    for alpha in ALPHAS[1:]:
        for segment in ("clean", "mean_missing", "robust_score"):
            for metric, data in paired[alpha][segment].items():
                lines.append(f"| {alpha} | {segment} | {metric} | {fmt(data)} | "
                             f"{data['improvement_direction_count']}/3 |")
    lines += ["", "## Per-modality mean-missing results", "",
              "| Alpha | Modality | Accuracy | Macro-F1 | MAE | Pearson | Score |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        for modality in ("text", "audio", "vision"):
            values = sm[alpha]["modality"][modality]
            lines.append(f"| {alpha} | {modality} | " + " | ".join(fmt(values[m]) for m in FIVE) + " |")
    lines += ["", "## vision_all_zero subset", "",
              "| Alpha | Segment | Accuracy | Macro-F1 | MAE | Pearson | Score |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        for segment in ("clean", "mean_missing"):
            values = sm[alpha]["vision_all_zero"][segment]
            lines.append(f"| {alpha} | {segment} | " + " | ".join(fmt(values[m]) for m in FIVE) + " |")
    lines += ["", "## Per-seed scores", "",
              "| Training seed | Alpha | Clean score | Mean-missing score | Robust score |",
              "|---:|---:|---:|---:|---:|"]
    for row in result["per_seed_alpha_scores"]:
        lines.append(f"| {row['seed']} | {row['alpha']:.2f} | {row['clean_score']:.4f} | "
                     f"{row['mean_missing_score']:.4f} | {row['robust_score']:.4f} |")
    lines += ["", "## Findings", "",
              "1. No nonzero alpha improves robust score in all three seeds. Each tested nonzero alpha improves seed 42, but lowers robust score in seeds 43 and 44; their per-seed robust-optimal alpha is 0.",
              "2. Mean-missing selection score does not improve across seeds: seeds 43 and 44 are below alpha=0 for every nonzero alpha, so the seed-42 gains do not reproduce. The three-seed mean-missing score is lower at every nonzero alpha.",
              "3. Accuracy decreases for all seeds on both clean and mean-missing sets at every nonzero alpha. Macro-F1 is mixed. On mean-missing, MAE improves in seeds 42 and 44 but worsens in seed 43 at every nonzero alpha. Pearson improves only in seed 42 and declines in seeds 43/44.",
              "4. Text missing has no stable gain. The mean text score changes only slightly (about 0.7350 at alpha=0 to at most 0.7352 at alpha=0.75), with mixed paired directions.",
              "5. The vision_all_zero mean-missing subset is harmed in all three seeds at every nonzero alpha: Accuracy and Macro-F1 fall, MAE rises, and its selection score falls.",
              "6. There is no globally defensible nonzero alpha across seeds. Keep alpha=0 / B0-WCE as the stable choice and terminate the reconstruction main line; do not proceed to B4."]
    (METRICS / "b32_multiseed_runs.csv").write_text(
        "seed,alpha,clean_score,mean_missing_score,robust_score\n" + "\n".join(
            f"{r['seed']},{r['alpha']:.2f},{r['clean_score']:.10f},{r['mean_missing_score']:.10f},{r['robust_score']:.10f}"
            for r in result["per_seed_alpha_scores"]) + "\n", encoding="utf-8")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
