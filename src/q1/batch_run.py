"""问题 1 Baseline V0 批量实验入口。

对若干 case × 若干核数运行 V0，并把官方 evaluator 的真实结果汇总成 CSV。
超时按真实情况记录为 success=False / failure_reason=timeout，不虚构 Makespan。

用法：
    python src/q1/batch_run.py --problem 1 --cores 2 3 4 5 --cases case_019 case_035
    python src/q1/batch_run.py --problem 1 --cores 2 3 4 5 --auto-select
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.q1"

from .evaluator import DEFAULT_TIMEOUT  # noqa: E402
from .partition import partition  # noqa: E402
from .paths import REPO_ROOT, case_path, list_cases  # noqa: E402
from .problem import Problem  # noqa: E402
from .schedule import list_schedule  # noqa: E402
from .solver import build_plan, solve_case  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "results" / "q1_v0" / "baseline_v0.csv"

CSV_FIELDS = [
    "case", "tier", "num_ops", "num_tensors", "K", "num_subgraphs",
    "granularities_tried", "makespan", "singlecore_makespan", "speedup",
    "added_copy_bytes", "proxy_makespan",
    "solver_time", "evaluator_time",
    "success", "failure_reason", "total_cycles", "M_cycles", "V_cycles",
]

SINGLECORE_CSV = REPO_ROOT / "results" / "q1_v0" / "singlecore_baseline.csv"


def load_singlecore_makespans(path: Path = SINGLECORE_CSV) -> dict:
    """读取官方单核基准作为加速比分母；文件不存在时留空，不编造分母。"""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["case"]: int(row["makespan"])
                for row in csv.DictReader(handle) if row.get("makespan")}


def auto_select(counts=(2, 2, 1), log=print) -> list:
    """按非 COPY 操作数从 100 个 case 中挑小/中/大代表。

    去掉最小与最大的极端者后按操作数三分位取代表，选择完全由排序决定，
    不含随机性。返回的每个档位内按操作数升序取前若干个。
    """
    sized = []
    for path in list_cases():
        problem = Problem.from_file(path)
        sized.append((len(problem.nc_ids), path))
    sized.sort(key=lambda item: (item[0], item[1].name))
    if len(sized) < 12:
        raise RuntimeError("可用 case 太少，无法自动分档")

    trimmed = sized[2:-2]  # 去掉极小的退化图与最大的算力黑洞
    third = len(trimmed) // 3
    tiers = [trimmed[:third], trimmed[third:2 * third], trimmed[2 * third:]]
    picked = []
    for tier, count in zip(tiers, counts):
        step = max(1, len(tier) // count)
        picked.extend(path for _, path in tier[::step][:count])
    for size, path in ((n, p) for n, p in sized):
        if path in picked:
            log(f"  auto-select: {path.stem} ({size} 非 COPY 操作)")
    return picked


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="问题 1 Baseline V0 批量实验")
    parser.add_argument("--problem", type=int, default=1, choices=[1])
    parser.add_argument("--cores", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--tier-map", nargs="*", default=[],
                        metavar="CASE=TIER", help="给 case 打标签，写入 CSV 的 tier 列")
    parser.add_argument("--auto-select", action="store_true",
                        help="自动挑选 2 小 / 2 中 / 1 大 代表 case")
    parser.add_argument("--granularities", type=int, nargs="+", default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--report-dir", default=None,
                        help="写出每个 (case, K) 的逐候选记录 JSON")
    parser.add_argument("--plan-dir", default=str(REPO_ROOT / "results" / "q1_v0" / "plans"),
                        help="写出每个 (case, K) 最优方案的 multicore_res.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.auto_select:
        print("自动挑选代表 case：")
        paths = auto_select(log=print)
    elif args.cases:
        paths = [case_path(name) for name in args.cases]
    else:
        raise SystemExit("需要 --cases 或 --auto-select")

    tiers = {}
    for item in args.tier_map:
        name, _, tier = item.partition("=")
        tiers[name] = tier

    singlecore = load_singlecore_makespans()
    rows = []
    started = time.perf_counter()
    for path in paths:
        problem = Problem.from_file(path)
        summary = problem.summary()
        tier = tiers.get(problem.name, "")
        for num_cores in args.cores:
            print(f"[{problem.name}] K={num_cores}")
            outcome = solve_case(problem, num_cores,
                                 granularities=args.granularities,
                                 timeout=args.timeout,
                                 use_cache=not args.no_cache,
                                 graph_path=path, log=print)
            best = outcome["best"]
            row = {
                "case": problem.name, "tier": tier,
                "num_ops": summary["num_ops_non_copy"],
                "num_tensors": summary["num_tensors"], "K": num_cores,
                "num_subgraphs": best["granularity"] if best else None,
                "granularities_tried": len(outcome["candidates"]),
                "makespan": best["makespan"] if best else None,
                "added_copy_bytes": best["added_copy_bytes"] if best else None,
                "proxy_makespan": best["proxy_makespan"] if best else None,
                "solver_time": round(sum(c["solver_time"]
                                         for c in outcome["candidates"]), 6),
                "evaluator_time": round(sum(c.get("evaluator_runtime", 0.0)
                                            for c in outcome["candidates"]), 6),
                "success": best is not None,
                "failure_reason": "" if best else ";".join(sorted({
                    c.get("failure_reason") or "invalid_plan"
                    for c in outcome["candidates"]})),
            }
            row.update({"total_cycles": summary["total_cycles"],
                        "M_cycles": summary["m_cycles"],
                        "V_cycles": summary["v_cycles"]})
            single = singlecore.get(problem.name)
            row["singlecore_makespan"] = single if single else ""
            row["speedup"] = (round(single / row["makespan"], 4)
                              if single and row["makespan"] else "")
            rows.append(row)

            if best is not None and args.plan_dir:
                plan_dir = Path(args.plan_dir)
                plan_dir.mkdir(parents=True, exist_ok=True)
                part = partition(problem, best["granularity"])
                sched = list_schedule(
                    part, num_cores,
                    same_core_wait=outcome["waits"]["task_same_core_wait_cycles"],
                    cross_core_wait=outcome["waits"]["task_cross_core_wait_cycles"])
                (plan_dir / f"{problem.name}_k{num_cores}_multicore_res.json").write_text(
                    json.dumps(build_plan(part, sched, num_cores),
                               ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")

            if args.report_dir:
                report_dir = Path(args.report_dir)
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / f"{problem.name}_k{num_cores}.json").write_text(
                    json.dumps(outcome["candidates"], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["success"])
    print(f"\n完成 {len(rows)} 组 (case, K)，成功 {ok} 组；"
          f"总耗时 {time.perf_counter() - started:.1f}s -> {csv_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
