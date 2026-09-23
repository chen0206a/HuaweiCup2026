#!/usr/bin/env python3
"""Summarize formal experiment folders into CSV and Markdown."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
CSV_OUT = ROOT / "results" / "experiment_summary.csv"
MD_OUT = ROOT / "results" / "experiment_summary.md"
FIELDS = ["experiment_id", "model", "features", "split", "seed", "metrics", "notes"]

def load_config(path):
    if not path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except ImportError:
        # Small fallback for simple scalar YAML fields; PyYAML is listed in requirements.
        data = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
            if m:
                value = m.group(2).strip('\"\'')
                if value not in ("", "[]", "{}", "null", "None"):
                    data[m.group(1)] = value
        return data

def short_notes(path):
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"^\s*[-*]\s*", "", line.strip())
        if line and not line.startswith("#"):
            return line[:240]
    return ""

def main():
    rows = []
    if EXPERIMENTS.exists():
        for folder in sorted(EXPERIMENTS.iterdir()):
            if not folder.is_dir() or not (folder.name.startswith("exp_") or folder.name == "example_exp"):
                continue
            config = load_config(folder / "config.yaml")
            try:
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics = {}
            if not isinstance(metrics, dict):
                metrics = {"value": metrics}
            features = config.get("features", "")
            if isinstance(features, (dict, list, tuple)):
                features = json.dumps(features, ensure_ascii=False)
            model = config.get("model", "")
            if isinstance(model, (dict, list)):
                model = json.dumps(model, ensure_ascii=False)
            split = config.get("split", config.get("data_split", ""))
            seed = config.get("seed", config.get("random_seed", ""))
            rows.append({"experiment_id": folder.name, "model": model or "", "features": features or "", "split": split or "", "seed": "" if seed is None else seed, "metrics": json.dumps(metrics, ensure_ascii=False, sort_keys=True), "notes": short_notes(folder / "notes.md")})
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Experiment Summary", "", f"实验记录数（含示例占位）：{len(rows)}", "", "| " + " | ".join(FIELDS) + " |", "|" + "|".join(["---"] * len(FIELDS)) + "|"]
    for row in rows:
        vals = [str(row[k]).replace("|", "\\|").replace("\n", " ") for k in FIELDS]
        lines.append("| " + " | ".join(vals) + " |")
    if not rows:
        lines.append("| — | — | — | — | — | — | 尚无正式实验结果 |")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {CSV_OUT.relative_to(ROOT)} and {MD_OUT.relative_to(ROOT)} ({len(rows)} records, including any example placeholder)")

if __name__ == "__main__":
    main()
