"""Compare B0 checkpoint selection epoch with validation metric optima only."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "outputs/metrics/b0_training_history.json"
OUTPUT = ROOT / "outputs/metrics/b0_selection_analysis.json"


def main() -> None:
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    criteria = {
        "valid_loss": ("valid_loss", min),
        "accuracy": ("valid_accuracy", max),
        "macro_f1": ("valid_macro_f1", max),
        "mae": ("valid_mae", min),
        "pearson": ("valid_pearson", max),
    }
    chosen = {name: fn(history, key=lambda row: row[field]) for name, (field, fn) in criteria.items()}
    loss_epoch = chosen["valid_loss"]["epoch"]
    report = {
        "scope": "validation only; test metrics were not read",
        "best_epoch_by_valid_loss": loss_epoch,
        "criteria": {
            name: {
                "epoch": row["epoch"],
                "value": row[field],
                "value_at_valid_loss_best_epoch": chosen["valid_loss"][field],
                "difference_from_valid_loss_best": row[field] - chosen["valid_loss"][field],
                "epoch_offset": row["epoch"] - loss_epoch,
            }
            for name, (field, _) in criteria.items()
            for row in [chosen[name]]
        },
        "history_epochs": len(history),
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# B0 validation checkpoint-selection analysis", "",
        "Validation history only; no test metrics were read.", "",
        f"Checkpoint selected by minimum validation total loss: epoch {loss_epoch}.", "",
        "| Criterion | Optimal epoch | Value at criterion optimum | Value at loss-selected epoch | Epoch offset |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "valid_loss": "Minimum valid loss", "accuracy": "Maximum Accuracy",
        "macro_f1": "Maximum Macro-F1", "mae": "Minimum MAE", "pearson": "Maximum Pearson",
    }
    for name, (field, _) in criteria.items():
        row = chosen[name]
        lines.append(
            f"| {labels[name]} | {row['epoch']} | {row[field]:.6f} | "
            f"{chosen['valid_loss'][field]:.6f} | {row['epoch'] - loss_epoch:+d} |"
        )
    (ROOT / "outputs/metrics/b0_selection_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
