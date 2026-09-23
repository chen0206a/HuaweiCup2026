"""Q1 V1-lite ablations with equal official-evaluator candidate budgets.

Run: python -m src.q1.experiments_v1 --cases case_093 case_026 --cores 2 3 4 5
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .batch_run import load_singlecore_makespans
from .evaluator import evaluate_plan
from .official import read_official_config
from .partition import candidate_granularities, partition
from .partition_v1 import partition_v1, scene_a_copy_bytes
from .paths import REPO_ROOT, case_path
from .problem import Problem
from .schedule import list_schedule
from .schedule_v1 import global_aware_schedule
from .solver import build_plan
from .validate import validate_plan

ALGORITHMS = ("v0", "v1_partition", "v1_schedule", "v1")
FIELDS = ("case", "K", "algorithm", "makespan", "singlecore_makespan",
          "speedup", "added_copy_bytes", "num_subgraphs", "topo_mode",
          "candidate_count", "evaluator_count", "cache_hits", "solver_time", "evaluator_time",
          "total_wall_time", "success", "failure_reason")


def construct(problem, k, algorithm, waits):
    modes = ("deterministic", "critical", "branch") if algorithm in (
        "v1_partition", "v1") else ("deterministic",)
    choices = []
    for mode in modes:
        for count in candidate_granularities(k, len(problem.nc_ids)):
            started = time.perf_counter()
            part = (partition(problem, count) if algorithm in ("v0", "v1_schedule")
                    else partition_v1(problem, count, mode))
            sched_func = (global_aware_schedule if algorithm in ("v1_schedule", "v1")
                          else list_schedule)
            sched = sched_func(part, k,
                               same_core_wait=waits["task_same_core_wait_cycles"],
                               cross_core_wait=waits["task_cross_core_wait_cycles"])
            plan = build_plan(part, sched, k)
            check = validate_plan(problem, plan)
            if not check.ok:
                raise ValueError(f"invalid candidate {mode}, G={count}: {check.errors[:3]}")
            copy_bytes = scene_a_copy_bytes(problem, part.node_to_sg)
            proxy = max(sched.makespan_proxy, copy_bytes / 60)
            choices.append({"mode": mode, "count": len(part.blocks), "plan": plan,
                            "proxy": proxy, "copy_bytes": copy_bytes,
                            "solver_time": time.perf_counter() - started})
    return choices


def shortlist(choices, budget, algorithm):
    if algorithm in ("v0", "v1_schedule"):
        return choices[:budget]
    selected = []
    for mode in ("deterministic", "critical", "branch"):
        options = [c for c in choices if c["mode"] == mode]
        if options:
            selected.append(min(options, key=lambda c: (c["proxy"], c["copy_bytes"])))
    for candidate in sorted(choices, key=lambda c: (c["proxy"], c["copy_bytes"])):
        if len(selected) >= budget:
            break
        if candidate not in selected:
            selected.append(candidate)
    return selected[:budget]


def run_one(problem, graph_path, k, algorithm, budget, output_dir, single):
    wall_start = time.perf_counter()
    _, _, waits = read_official_config()
    choices = construct(problem, k, algorithm, waits)
    chosen = shortlist(choices, budget, algorithm)
    records = []
    for candidate in chosen:
        result = evaluate_plan(problem.graph, candidate["plan"],
                               graph_name=problem.name, graph_path=graph_path,
                               timeout=600.0)
        records.append({
            "mode": candidate["mode"], "count": candidate["count"],
            "proxy": candidate["proxy"], "copy_bytes": candidate["copy_bytes"],
            "success": result.success, "makespan": result.makespan,
            "added_copy_bytes": result.added_copy_bytes,
            "cached": result.cached, "evaluator_runtime": result.evaluator_runtime,
            "failure_reason": result.failure_reason,
            "plan_hash": result.plan_hash,
        })
    successful = [(candidate, record) for candidate, record in zip(chosen, records)
                  if record["success"]]
    best = min(successful, key=lambda pair: (pair[1]["makespan"],
                                              pair[1]["added_copy_bytes"]), default=None)
    if best:
        plan_path = output_dir / "plans" / f"{problem.name}_k{k}_{algorithm}_multicore_res.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(best[0]["plan"], ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    report_path = output_dir / "reports" / f"{problem.name}_k{k}_{algorithm}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    makespan = best[1]["makespan"] if best else None
    return {
        "case": problem.name, "K": k, "algorithm": algorithm,
        "makespan": makespan, "singlecore_makespan": single or "",
        "speedup": round(single / makespan, 4) if single and makespan else "",
        "added_copy_bytes": best[1]["added_copy_bytes"] if best else "",
        "num_subgraphs": best[0]["count"] if best else "",
        "topo_mode": best[0]["mode"] if best else "",
        "candidate_count": len(records),
        "evaluator_count": sum(not r["cached"] for r in records),
        "cache_hits": sum(r["cached"] for r in records),
        "solver_time": round(sum(c["solver_time"] for c in choices), 6),
        "evaluator_time": round(sum(r["evaluator_runtime"] for r in records
                                    if not r["cached"]), 6),
        "total_wall_time": round(time.perf_counter() - wall_start, 6),
        "success": bool(best),
        "failure_reason": "" if best else ";".join(sorted({
            r["failure_reason"] or "unknown" for r in records})),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Q1 V1-lite equal-budget ablations")
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--cores", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--algorithm", choices=(*ALGORITHMS, "all"), default="all")
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "results" / "q1_v1_lite")
    args = p.parse_args(argv)
    if args.budget < 1:
        p.error("--budget must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    singles = load_singlecore_makespans()
    csv_path = args.output_dir / "ablation.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    algorithms = ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    for name in args.cases:
        graph_path = case_path(name)
        problem = Problem.from_file(graph_path)
        for k in args.cores:
            for algorithm in algorithms:
                row = run_one(problem, graph_path, k, algorithm, args.budget,
                              args.output_dir, singles.get(problem.name))
                rows = [old for old in rows if not (
                    old["case"] == row["case"] and str(old["K"]) == str(row["K"])
                    and old["algorithm"] == row["algorithm"])]
                rows.append(row)
                print(problem.name, "K", k, algorithm, row["makespan"],
                      "eval", row["evaluator_count"], "cached", row["cache_hits"],
                      "wall", row["total_wall_time"], flush=True)
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
