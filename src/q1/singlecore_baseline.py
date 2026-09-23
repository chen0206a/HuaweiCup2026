"""官方单核基准：Makespan 的分母，用于判断多核到底有没有加速。

直接调用官方 `code/singlecore_evaluate.py`（全图一个子图、核心 0），
不修改任何官方代码。这不是优化算法，只是测量工具。

用法：
    python src/q1/singlecore_baseline.py case_093 case_036 case_025
    python src/q1/singlecore_baseline.py --auto-select
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.q1"

from .paths import REPO_ROOT, case_path, config_path, official_code_dir  # noqa: E402
from .problem import Problem  # noqa: E402

RUN_DIR = REPO_ROOT / ".cache" / "q1_eval" / "_singlecore"
DEFAULT_CSV = REPO_ROOT / "results" / "q1_v0" / "singlecore_baseline.csv"


def evaluate_singlecore(graph_path: Path, timeout: float) -> dict:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RUN_DIR / f"{graph_path.stem}_singlecore_res.json"
    trace = RUN_DIR / f"{graph_path.stem}_trace.json"
    log = RUN_DIR / f"{graph_path.stem}_log.txt"
    for stale in (out_file,):
        if stale.exists():
            stale.unlink()

    command = [
        sys.executable, "singlecore_evaluate.py", str(graph_path),
        "--config", str(config_path()),
        "-o", str(out_file), "--trace-output", str(trace), "--log-output", str(log),
    ]
    started = time.perf_counter()
    reason, returncode, stderr = None, None, ""
    try:
        completed = subprocess.run(
            command, cwd=str(official_code_dir()), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout)
        returncode, stderr = completed.returncode, completed.stderr or ""
        if returncode != 0:
            reason = "nonzero_returncode"
    except subprocess.TimeoutExpired as expired:
        stderr = (expired.stderr or "") if isinstance(expired.stderr, str) else ""
        reason = "timeout"
    runtime = time.perf_counter() - started

    makespan = None
    if reason is None:
        if not out_file.is_file():
            reason = "result_json_missing"
        else:
            try:
                makespan = json.loads(out_file.read_text(encoding="utf-8"))["makespan"]
            except (OSError, ValueError, KeyError):
                reason = "result_json_invalid"
    return {"success": reason is None, "makespan": makespan,
            "runtime": runtime, "failure_reason": reason,
            "stderr_tail": stderr.strip()[-400:]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="官方单核基准")
    parser.add_argument("cases", nargs="*", default=None)
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args(argv)

    if args.auto_select:
        from .batch_run import auto_select
        paths = auto_select(log=print)
    elif args.cases:
        paths = [case_path(name) for name in args.cases]
    else:
        raise SystemExit("需要 case 名称或 --auto-select")

    rows = []
    for path in paths:
        problem = Problem.from_file(path)
        summary = problem.summary()
        result = evaluate_singlecore(path, args.timeout)
        print(f"[{problem.name}] singlecore makespan={result['makespan']} "
              f"({result['runtime']:.1f}s) reason={result['failure_reason']}")
        rows.append({
            "case": problem.name,
            "num_ops": summary["num_ops_non_copy"],
            "num_tensors": summary["num_tensors"],
            "makespan": result["makespan"],
            "success": result["success"],
            "failure_reason": result["failure_reason"],
            "runtime": round(result["runtime"], 6),
            "total_cycles": summary["total_cycles"],
        })

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
