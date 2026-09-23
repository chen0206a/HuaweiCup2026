"""Create a validation-only comparison; deliberately excludes test metrics."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs/metrics"
FIELDS = ("accuracy", "macro_f1", "mae", "pearson")
CLASSES = ("Negative", "Neutral", "Positive")


def validation_summary(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    valid = record["metrics"]["valid"]
    summary = {key: valid[key] for key in FIELDS}
    summary["per_class"] = {
        name: {key: valid["per_class"][name][key] for key in ("recall", "f1")}
        for name in CLASSES
    }
    return summary


def main() -> None:
    base = validation_summary(METRICS / "b0_metrics.json")
    weighted = validation_summary(METRICS / "b0_weighted_ce_metrics.json")
    report = {
        "scope": "validation only; attachment2 test split is excluded",
        "models": {"B0": base, "B0-weighted-CE": weighted},
        "weighted_minus_b0": {
            **{key: weighted[key] - base[key] for key in FIELDS},
            "per_class": {
                name: {
                    metric: weighted["per_class"][name][metric] - base["per_class"][name][metric]
                    for metric in ("recall", "f1")
                }
                for name in CLASSES
            },
        },
    }
    path = METRICS / "b0_weighted_ce_validation_comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [
        ("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"),
        ("MAE", "mae"), ("Pearson", "pearson"),
    ]
    lines = [
        "# B0 vs B0-weighted-CE validation comparison", "",
        "Validation only; the attachment 2 test split was not used.", "",
        "| Metric | B0 | B0-weighted-CE | Weighted − B0 |",
        "|---|---:|---:|---:|",
    ]
    for title, key in rows:
        lines.append(
            f"| {title} | {base[key]:.4f} | {weighted[key]:.4f} | {weighted[key] - base[key]:+.4f} |"
        )
    lines.extend(["", "## Per-class recall and F1", "",
                  "| Class | B0 recall / F1 | Weighted recall / F1 | Recall Δ | F1 Δ |",
                  "|---|---:|---:|---:|---:|"])
    for name in CLASSES:
        br, bf = base["per_class"][name]["recall"], base["per_class"][name]["f1"]
        wr, wf = weighted["per_class"][name]["recall"], weighted["per_class"][name]["f1"]
        lines.append(f"| {name} | {br:.4f} / {bf:.4f} | {wr:.4f} / {wf:.4f} | {wr-br:+.4f} | {wf-bf:+.4f} |")
    (METRICS / "b0_weighted_ce_validation_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
