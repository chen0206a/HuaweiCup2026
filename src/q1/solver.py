"""问题 1 Baseline / V0 求解入口。

流程：case.json → 连续工作量分块 → 关键路径列表调度 → validate_plan
     → 官方 problem_1 evaluator → 真实 Makespan / added_copy_bytes。

对 K、2K、4K、8K 四种粒度分别真实评价，取官方结果最优者。
best 只由官方 evaluator 的真实结果决定，proxy 不参与最终选择。

用法：
    python src/q1/solver.py case_019 -n 4
    python src/q1/solver.py <case.json> -n 4 --mode problem1 -o out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # 允许 python src/q1/solver.py 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.q1"

from .evaluator import DEFAULT_TIMEOUT, evaluate_plan  # noqa: E402
from .official import official_fingerprint, read_official_config  # noqa: E402
from .partition import candidate_granularities, partition  # noqa: E402
from .paths import REPO_ROOT, case_path  # noqa: E402
from .problem import Problem  # noqa: E402
from .schedule import list_schedule  # noqa: E402
from .validate import validate_plan  # noqa: E402

DEFAULT_PLAN_DIR = REPO_ROOT / "results" / "q1_v0" / "plans"


def build_plan(partition_result, schedule_result, num_cores: int) -> dict:
    """组装官方方案：只含 node_to_subgraph 与 core_schedules 两个字段。"""
    return {
        "node_to_subgraph": {int(op_id): sgid
                             for op_id, sgid in partition_result.node_to_sg.items()},
        "core_schedules": [list(order) for order in schedule_result.core_orders],
    }


def solve_case(problem: Problem, num_cores: int, granularities=None,
               timeout: float = DEFAULT_TIMEOUT, use_cache: bool = True,
               graph_path=None, log=print) -> dict:
    """跑完所有候选粒度并返回逐候选记录与最优方案。"""
    if granularities is None:
        granularities = candidate_granularities(num_cores, len(problem.nc_ids))
    bandwidth, capacity, waits = read_official_config()

    candidates = []
    for num_subgraphs in granularities:
        started = time.perf_counter()
        part = partition(problem, num_subgraphs)
        sched = list_schedule(
            part, num_cores,
            same_core_wait=waits["task_same_core_wait_cycles"],
            cross_core_wait=waits["task_cross_core_wait_cycles"])
        plan = build_plan(part, sched, num_cores)
        solver_time = time.perf_counter() - started

        report = validate_plan(problem, plan)
        record = {
            "case": problem.name,
            "num_cores": num_cores,
            "granularity": len(part.blocks),
            "num_subgraphs": len(part.blocks),
            "proxy_makespan": sched.makespan_proxy,
            "solver_time": round(solver_time, 6),
            "valid": report.ok,
            "validation_errors": report.errors[:5],
        }
        if not report.ok:
            record.update({"success": False, "makespan": None,
                           "added_copy_bytes": None, "evaluator_runtime": 0.0,
                           "failure_reason": "invalid_plan"})
            candidates.append(record)
            log(f"  G={len(part.blocks):>4}  proxy={sched.makespan_proxy:>12}  "
                f"[方案非法] {report.errors[:1]}")
            continue

        result = evaluate_plan(problem.graph, plan, graph_name=problem.name,
                               problem=1, timeout=timeout,
                               use_cache=use_cache, graph_path=graph_path)
        record.update({
            "success": result.success,
            "makespan": result.makespan,
            "added_copy_bytes": result.added_copy_bytes,
            "evaluator_runtime": round(result.evaluator_runtime, 6),
            "failure_reason": result.failure_reason,
            "cached": result.cached,
            "plan_hash": result.plan_hash,
            "stderr_tail": result.stderr_tail,
        })
        candidates.append(record)
        if result.success:
            log(f"  G={len(part.blocks):>4}  proxy={sched.makespan_proxy:>12}  "
                f"makespan={result.makespan:>12}  added_copy={result.added_copy_bytes:>12}"
                f"  ({result.evaluator_runtime:.2f}s{', cached' if result.cached else ''})")
        else:
            log(f"  G={len(part.blocks):>4}  [失败] {result.failure_reason}"
                f"{' ' + result.stderr_tail[:160] if result.stderr_tail else ''}")

    successful = [c for c in candidates if c.get("success")]
    best = min(successful,
               key=lambda c: (c["makespan"], c["added_copy_bytes"], c["granularity"]),
               default=None)
    return {"candidates": candidates, "best": best, "num_cores": num_cores,
            "bandwidth": bandwidth, "capacity": capacity,
            "waits": waits, "granularities": granularities}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="问题 1 Baseline V0：连续工作量分块 + 关键路径列表调度")
    parser.add_argument("case", help="case 名称或计算图 JSON 路径")
    parser.add_argument("-n", "--num-cores", type=int, required=True)
    parser.add_argument("--mode", default="problem1", choices=["problem1"])
    parser.add_argument("--granularities", type=int, nargs="+", default=None,
                        help="覆盖默认的 K/2K/4K/8K 候选粒度")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="单次官方评价的子进程超时（秒）")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("-o", "--output", help="最优方案 JSON 输出路径")
    parser.add_argument("--report", help="逐候选记录 JSON 输出路径")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    graph_path = case_path(args.case)
    problem = Problem.from_file(graph_path)
    summary = problem.summary()
    log(f"[{problem.name}] 非 COPY 操作 {summary['num_ops_non_copy']}，"
        f"tensor {summary['num_tensors']}，总 cycles {summary['total_cycles']}，"
        f"M={summary['m_cycles']} V={summary['v_cycles']}，核数 {args.num_cores}")

    started = time.perf_counter()
    outcome = solve_case(problem, args.num_cores,
                         granularities=args.granularities,
                         timeout=args.timeout, use_cache=not args.no_cache,
                         graph_path=graph_path, log=log)
    wall = time.perf_counter() - started

    best = outcome["best"]
    if best is None:
        log("未取得任何通过官方评价的方案；不输出 multicore_res.json")
        return 1

    part = partition(problem, best["granularity"])
    sched = list_schedule(part, args.num_cores,
                          same_core_wait=outcome["waits"]["task_same_core_wait_cycles"],
                          cross_core_wait=outcome["waits"]["task_cross_core_wait_cycles"])
    plan = build_plan(part, sched, args.num_cores)
    output = Path(args.output) if args.output else (
        DEFAULT_PLAN_DIR / f"{problem.name}_k{args.num_cores}_multicore_res.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")

    log(f"best: G={best['granularity']} makespan={best['makespan']} "
        f"added_copy_bytes={best['added_copy_bytes']} wall={wall:.2f}s -> {output}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "case": problem.name,
            "num_cores": args.num_cores,
            "problem": 1,
            "mode": args.mode,
            "wall_time": wall,
            "summary": summary,
            "official_fingerprint": official_fingerprint(),
            "candidates": outcome["candidates"],
            "best": best,
            "plan_file": str(output),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
