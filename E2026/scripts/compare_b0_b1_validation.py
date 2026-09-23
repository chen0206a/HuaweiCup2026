"""Compare all four models at their best project-validation-score checkpoint."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs/metrics"
RUNS = {
    "B0-CE": "b0_ce_score_selection_metrics.json",
    "B0-WeightedCE": "b0_weighted_ce_score_selection_metrics.json",
    "B1-CE": "b1_ce_metrics.json",
    "B1-WeightedCE": "b1_weighted_ce_metrics.json",
}
CLASSES = ("Negative", "Neutral", "Positive")


def main() -> None:
    results = {}
    for name, filename in RUNS.items():
        raw = json.loads((METRICS / filename).read_text(encoding="utf-8"))
        metrics = raw["best_selection_score"]["validation_metrics"]
        results[name] = {
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "mae": metrics["mae"], "pearson": metrics["pearson"],
            "selection_score": raw["best_selection_score"]["value"],
            "best_score_epoch": raw["best_selection_score"]["epoch"],
            "best_loss_epoch": raw["best_valid_loss"]["epoch"],
            "best_loss_and_score_epochs_differ": (
                raw["best_valid_loss"]["epoch"] != raw["best_selection_score"]["epoch"]
            ),
            "parameter_count": raw["parameter_count"],
            "training_seconds": raw["training_seconds"],
            "per_class": {
                cls: {key: metrics["per_class"][cls][key] for key in ("recall", "f1")}
                for cls in CLASSES
            },
            "vision_all_zero_count": metrics["vision_all_zero_count"],
            "vision_all_zero_metrics": metrics["vision_all_zero_metrics"],
            "test_used": False,
        }

    report = {
        "scope": "train/validation only; no attachment2 test data used for selection or comparison",
        "selection_score_definition": "0.25*Accuracy + 0.25*Macro-F1 + 0.25*(1-MAE/6) + 0.25*(Pearson+1)/2",
        "models": results,
    }
    json_path = METRICS / "b0_b1_validation_comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    order = list(RUNS)
    lines = [
        "# B0 / B1 validation comparison", "",
        "Every row uses that run's best project validation selection-score checkpoint. "
        "Attachment 2 test was not used.", "",
        "| Model | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Best score epoch | Best loss epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        row = results[name]
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['mae']:.4f} | "
            f"{row['pearson']:.4f} | {row['selection_score']:.4f} | {row['best_score_epoch']} | {row['best_loss_epoch']} |"
        )
    lines.extend(["", "## Per-class recall / F1", "",
                  "| Model | Negative | Neutral | Positive |",
                  "|---|---:|---:|---:|"])
    for name in order:
        row = results[name]
        cells = [f"{row['per_class'][cls]['recall']:.4f} / {row['per_class'][cls]['f1']:.4f}"
                 for cls in CLASSES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Best-loss vs best-score", "",
                  "| Model | Different epochs? | Best loss epoch | Best score epoch |",
                  "|---|---:|---:|---:|"])
    for name in order:
        row = results[name]
        lines.append(
            f"| {name} | {'Yes' if row['best_loss_and_score_epochs_differ'] else 'No'} | "
            f"{row['best_loss_epoch']} | {row['best_score_epoch']} |"
        )
    lines.extend(["", "## Vision-all-zero validation subsets", "",
                  "| Model | N | Accuracy | Macro-F1 | MAE | Pearson |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name in order:
        row = results[name]
        subset = row["vision_all_zero_metrics"]
        if subset is None:
            lines.append(f"| {name} | 0 | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | {row['vision_all_zero_count']} | {subset['accuracy']:.4f} | "
                f"{subset['macro_f1']:.4f} | {subset['mae']:.4f} | {subset['pearson']:.4f} |"
            )
    (METRICS / "b0_b1_validation_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
