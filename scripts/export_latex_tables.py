#!/usr/bin/env python3
"""Export the experiment summary CSV as a compact LaTeX table."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "experiment_summary.csv"
OUT = ROOT / "paper" / "generated" / "model_comparison.tex"
COLS = ["experiment_id", "model", "features", "split", "seed", "metrics"]
def esc(value):
    value = str(value or "")
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        value = value.replace(a, b)
    return value.replace("\n", " ")
def main():
    rows = []
    if SRC.exists():
        with SRC.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spec = "l" + "l" * (len(COLS) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(esc(c.replace("_", " ")) for c in COLS) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(esc(row.get(c, "")) for c in COLS) + r" \\")
    if not rows:
        lines.append(" & ".join(["—"] * len(COLS)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Experiment comparison (generated from the current summary).}", r"\label{tab:model-comparison}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")
if __name__ == "__main__":
    main()
