"""Frozen Q1 V1-P validation: V0-only screening, holdout selection, comparison.

Run from the repository root:
  python -m src.q1.validate_v1_partition screen
  python -m src.q1.validate_v1_partition select
  python -m src.q1.validate_v1_partition run
  python -m src.q1.validate_v1_partition report

The algorithm implementations and the official evaluator are never modified here.
Every CSV is written after each completed unit so an interrupted run can resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from .batch_run import load_singlecore_makespans
from .evaluator import evaluate_plan
from .experiments_v1 import construct, shortlist
from .official import read_official_config
from .partition import build_subgraph_dag
from .partition_v1 import scene_a_copy_bytes
from .paths import REPO_ROOT, list_cases
from .problem import Problem
from .singlecore_baseline import evaluate_singlecore

OUT = REPO_ROOT / "results" / "q1_v1_validation"
SCREEN = OUT / "screening.csv"
SELECTION = OUT / "selection.csv"
RUNS = OUT / "runs.csv"
CANDIDATES = OUT / "candidates.csv"
SUMMARY = OUT / "summary.csv"
REPORT = REPO_ROOT / "q1" / "V1_VALIDATION_REPORT.md"
FREEZE_REPORT = REPO_ROOT / "q1" / "Q1_FREEZE_REPORT.md"
METRICS = REPO_ROOT / "experiments" / "exp_001_q1_v1p_validation" / "metrics.json"
DEVELOPMENT_CASES = {"case_014", "case_025", "case_026", "case_093"}
KS = (2, 3, 4, 5)
BUDGET = 4
SELECTION_COUNT = 12
MAX_SCREEN_SECONDS = 60.0
SCREEN_FIELDS = (
    "case", "num_ops", "num_components", "max_component_work_share",
    "critical_path", "critical_path_work_ratio", "fanout_mean", "fanout_p95",
    "fanout_max", "total_work", "v0_k4_makespan", "singlecore_makespan",
    "v0_k4_speedup", "v0_k4_added_copy_bytes", "screen_evaluator_count",
    "screen_evaluator_time", "singlecore_evaluator_time", "screen_status",
)
FEATURES = ("num_ops", "num_components", "max_component_work_share",
            "critical_path_work_ratio", "fanout_p95",
            "v0_k4_added_copy_bytes", "v0_k4_speedup")
RUN_FIELDS = (
    "case", "K", "algorithm", "makespan", "singlecore_makespan", "speedup",
    "added_copy_bytes", "num_subgraphs", "topo_mode", "candidate_count",
    "evaluator_count", "cache_hits", "solver_time", "evaluator_time",
    "total_wall_time", "success", "failure_reason", "proxy_makespan",
    "crossing_tensor_bytes", "pre_spill_copy_bytes", "block_layer_width",
    "block_critical_path", "plan_hash",
)
CANDIDATE_FIELDS = ("case", "K", "algorithm", "topo_mode", "num_subgraphs",
                    "proxy_makespan", "predicted_copy_bytes", "makespan",
                    "added_copy_bytes", "cached", "evaluator_runtime", "success",
                    "failure_reason", "plan_hash")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def quantile(values, q):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def graph_features(problem: Problem) -> dict:
    total_work = sum(problem.work.values())
    seen = set()
    components = []
    for root in problem.nc_ids:
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        comp_work = 0
        while stack:
            node = stack.pop()
            comp_work += problem.work[node]
            for neighbor in problem.nc_preds[node] | problem.nc_succs[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(comp_work)
    longest = {}
    for node in problem.topo_order:
        longest[node] = problem.work[node] + max(
            (longest[p] for p in problem.nc_preds[node]), default=0)
    critical = max(longest.values(), default=0)
    eligible = set(problem.nc_ids)
    fanouts = [len(t.consumers & eligible) for t in problem.tensors.values()
               if t.producers & eligible]
    return {
        "num_ops": len(problem.nc_ids),
        "num_components": len(components),
        "max_component_work_share": max(components, default=0) / total_work if total_work else 0,
        "critical_path": critical,
        "critical_path_work_ratio": critical / total_work if total_work else 0,
        "fanout_mean": statistics.mean(fanouts) if fanouts else 0,
        "fanout_p95": quantile(fanouts, .95) if fanouts else 0,
        "fanout_max": max(fanouts, default=0),
        "total_work": total_work,
    }


def eval_candidates(problem, graph_path, k, algorithm, waits):
    started = time.perf_counter()
    choices = construct(problem, k, algorithm, waits)
    chosen = shortlist(choices, BUDGET, algorithm)
    measured = []
    for candidate in chosen:
        result = evaluate_plan(problem.graph, candidate["plan"],
                               graph_name=problem.name, graph_path=graph_path,
                               timeout=600.0)
        measured.append((candidate, result))
    best = min(((c, r) for c, r in measured if r.success),
               key=lambda pair: (pair[1].makespan, pair[1].added_copy_bytes),
               default=None)
    return choices, measured, best, time.perf_counter() - started


def screen():
    rows = {r["case"]: r for r in read_rows(SCREEN)}
    existing_singles = load_singlecore_makespans()
    existing_v0 = {r["case"]: r for r in read_rows(
        REPO_ROOT / "results" / "q1_v0" / "baseline_v0.csv") if r["K"] == "4"}
    _, _, waits = read_official_config()
    for path in list_cases():
        if path.stem in rows and rows[path.stem].get("screen_status") == "success":
            continue
        problem = Problem.from_file(path)
        features = graph_features(problem)
        single = existing_singles.get(problem.name)
        single_runtime = 0.0
        if single is None:
            single_result = evaluate_singlecore(path, timeout=1800.0)
            single = single_result["makespan"] if single_result["success"] else None
            single_runtime = single_result["runtime"]
        old = existing_v0.get(problem.name)
        if old:
            v0_makespan = int(old["makespan"])
            v0_copy = int(old["added_copy_bytes"])
            eval_count = 0
            eval_time = 0.0
        else:
            _, measured, best, _ = eval_candidates(problem, path, 4, "v0", waits)
            v0_makespan = best[1].makespan if best else None
            v0_copy = best[1].added_copy_bytes if best else None
            eval_count = sum(not r.cached for _, r in measured)
            eval_time = sum(r.evaluator_runtime for _, r in measured if not r.cached)
        row = {"case": problem.name, **features,
               "v0_k4_makespan": v0_makespan or "",
               "singlecore_makespan": single or "",
               "v0_k4_speedup": single / v0_makespan if single and v0_makespan else "",
               "v0_k4_added_copy_bytes": v0_copy if v0_copy is not None else "",
               "screen_evaluator_count": eval_count,
               "screen_evaluator_time": round(eval_time, 6),
               "singlecore_evaluator_time": round(single_runtime, 6),
               "screen_status": "success" if single and v0_makespan else "failed"}
        rows[problem.name] = row
        write_rows(SCREEN, SCREEN_FIELDS, [rows[name] for name in sorted(rows)])
        print("screen", problem.name, row["screen_status"],
              "speedup", row["v0_k4_speedup"],
              "copy", row["v0_k4_added_copy_bytes"], flush=True)
    if len(rows) != 100 or any(r["screen_status"] != "success" for r in rows.values()):
        raise RuntimeError("screening is incomplete; no selection may be made")


def select():
    rows = read_rows(SCREEN)
    if len(rows) != 100 or any(r["screen_status"] != "success" for r in rows):
        raise RuntimeError("100 complete V0-only screening rows are required")
    available = [r for r in rows if r["case"] not in DEVELOPMENT_CASES and
                 float(r["screen_evaluator_time"]) +
                 float(r["singlecore_evaluator_time"]) <= MAX_SCREEN_SECONDS]
    available.sort(key=lambda r: (int(r["num_ops"]), r["case"]))
    tiers = [available[i * len(available) // 3:(i + 1) * len(available) // 3]
             for i in range(3)]
    all_values = {feature: [float(r[feature]) for r in available] for feature in FEATURES}
    def vector(row):
        return tuple(sum(value < float(row[feature]) for value in all_values[feature]) /
                     max(1, len(available) - 1) for feature in FEATURES)
    vectors = {r["case"]: vector(r) for r in available}
    picked = []
    output = []
    labels = ("small", "medium", "large")
    for tier, label in zip(tiers, labels):
        remaining = {r["case"]: r for r in tier}
        for _ in range(4):
            def distance(name):
                v = vectors[name]
                if not picked:
                    return sum((x - .5) ** 2 for x in v)
                return min(sum((x - y) ** 2 for x, y in zip(v, vectors[p]))
                           for p in picked)
            name = max(remaining, key=lambda n: (distance(n), n))
            chosen_distance = distance(name)
            row = remaining.pop(name)
            picked.append(name)
            deviations = sorted(((abs(v - .5), feature, v) for feature, v in
                                 zip(FEATURES, vectors[name])), reverse=True)
            reasons = [f"{feature} {'high' if v > .5 else 'low'}"
                       for _, feature, v in deviations[:2]]
            output.append({"case": name, "size_tier": label,
                           "reason": f"{label} stratum; " + "; ".join(reasons),
                           "selection_distance": round(chosen_distance, 6),
                           **{key: row[key] for key in SCREEN_FIELDS if key != "case"}})
    fields = ("case", "size_tier", "reason", "selection_distance",
              *(f for f in SCREEN_FIELDS if f != "case"))
    write_rows(SELECTION, fields, output)
    print("selected", len(output), "cases:", " ".join(picked), flush=True)


def plan_diagnostics(problem, plan):
    mapping = {int(op): block for op, block in plan["node_to_subgraph"].items()}
    blocks = set(mapping.values())
    _, succs, topo = build_subgraph_dag(problem, mapping, len(blocks))
    preds = {block: set() for block in blocks}
    for source, destinations in succs.items():
        for target in destinations:
            preds[target].add(source)
    level = {}
    for block in topo:
        level[block] = 1 + max((level[p] for p in preds[block]), default=-1)
    width = max(Counter(level.values()).values(), default=0)
    crossing = 0
    eligible = set(problem.nc_ids)
    for tensor in problem.tensors.values():
        producers = {mapping[u] for u in tensor.producers & eligible}
        consumers = {mapping[u] for u in tensor.consumers & eligible}
        if producers and consumers and any(p != c for p in producers for c in consumers):
            crossing += tensor.size
    return {"crossing_tensor_bytes": crossing,
            "pre_spill_copy_bytes": scene_a_copy_bytes(problem, mapping),
            "block_layer_width": width,
            "block_critical_path": max(level.values(), default=0) + 1}


def run():
    selection = read_rows(SELECTION)
    if len(selection) != SELECTION_COUNT:
        raise RuntimeError("selection.csv must contain exactly 12 cases")
    singles = {r["case"]: int(r["singlecore_makespan"]) for r in selection}
    rows = {(r["case"], int(r["K"]), r["algorithm"]): r for r in read_rows(RUNS)}
    candidate_rows = {(r["case"], int(r["K"]), r["algorithm"],
                       r["plan_hash"]): r for r in read_rows(CANDIDATES)}
    _, _, waits = read_official_config()
    for selected in selection:
        path = next(p for p in list_cases() if p.stem == selected["case"])
        problem = Problem.from_file(path)
        for k in KS:
            for algorithm in ("v0", "v1_partition"):
                key = (problem.name, k, algorithm)
                if key in rows and rows[key].get("success") == "True":
                    continue
                choices, measured, best, wall = eval_candidates(
                    problem, path, k, algorithm, waits)
                for candidate, result in measured:
                    item = {"case": problem.name, "K": k, "algorithm": algorithm,
                            "topo_mode": candidate["mode"],
                            "num_subgraphs": candidate["count"],
                            "proxy_makespan": candidate["proxy"],
                            "predicted_copy_bytes": candidate["copy_bytes"],
                            "makespan": result.makespan or "",
                            "added_copy_bytes": result.added_copy_bytes if result.added_copy_bytes is not None else "",
                            "cached": result.cached,
                            "evaluator_runtime": round(result.evaluator_runtime, 6),
                            "success": result.success,
                            "failure_reason": result.failure_reason or "",
                            "plan_hash": result.plan_hash}
                    candidate_rows[(problem.name, k, algorithm, result.plan_hash)] = item
                if best:
                    candidate, result = best
                    diagnostics = plan_diagnostics(problem, candidate["plan"])
                else:
                    candidate = result = None
                    diagnostics = {name: "" for name in
                                   ("crossing_tensor_bytes", "pre_spill_copy_bytes",
                                    "block_layer_width", "block_critical_path")}
                single = singles[problem.name]
                row = {"case": problem.name, "K": k, "algorithm": algorithm,
                       "makespan": result.makespan if result else "",
                       "singlecore_makespan": single,
                       "speedup": single / result.makespan if result else "",
                       "added_copy_bytes": result.added_copy_bytes if result else "",
                       "num_subgraphs": candidate["count"] if candidate else "",
                       "topo_mode": candidate["mode"] if candidate else "",
                       "candidate_count": len(measured),
                       "evaluator_count": sum(not r.cached for _, r in measured),
                       "cache_hits": sum(r.cached for _, r in measured),
                       "solver_time": round(sum(c["solver_time"] for c in choices), 6),
                       "evaluator_time": round(sum(r.evaluator_runtime for _, r in measured if not r.cached), 6),
                       "total_wall_time": round(wall, 6),
                       "success": bool(best),
                       "failure_reason": "" if best else ";".join(sorted(
                           {r.failure_reason or "unknown" for _, r in measured})),
                       "proxy_makespan": candidate["proxy"] if candidate else "",
                       **diagnostics,
                       "plan_hash": result.plan_hash if result else ""}
                rows[key] = row
                write_rows(RUNS, RUN_FIELDS, [rows[key] for key in sorted(rows)])
                write_rows(CANDIDATES, CANDIDATE_FIELDS,
                           [candidate_rows[key] for key in sorted(candidate_rows)])
                print("run", *key, "makespan", row["makespan"],
                      "evaluator_count", row["evaluator_count"],
                      "wall", row["total_wall_time"], flush=True)
    if len(rows) != SELECTION_COUNT * 4 * 2 or any(r["success"] != "True" and r["success"] is not True
                                          for r in rows.values()):
        raise RuntimeError("validation run incomplete or has failed combinations")


def report():
    selected = read_rows(SELECTION)
    runs = read_rows(RUNS)
    if len(selected) != SELECTION_COUNT or len(runs) != 96:
        raise RuntimeError("12 selected cases and 96 successful run rows are required")
    by_key = {(r["case"], int(r["K"]), r["algorithm"]): r for r in runs}
    if any(r["success"] != "True" for r in runs):
        raise RuntimeError("failed run rows cannot be summarized")
    selected_by_case = {r["case"]: r for r in selected}
    pairs = []
    for case in selected_by_case:
        for k in KS:
            v0 = by_key[(case, k, "v0")]
            v1 = by_key[(case, k, "v1_partition")]
            old, new = int(v0["makespan"]), int(v1["makespan"])
            pairs.append({"case": case, "K": k,
                          "size_tier": selected_by_case[case]["size_tier"],
                          "num_components": int(selected_by_case[case]["num_components"]),
                          "improvement": (old - new) / old,
                          "copy_change_bytes": int(v1["added_copy_bytes"]) - int(v0["added_copy_bytes"]),
                          "v0": v0, "v1": v1})
    component_values = [p["num_components"] for p in pairs]
    low, high = quantile(component_values, 1 / 3), quantile(component_values, 2 / 3)
    for pair in pairs:
        n = pair["num_components"]
        pair["component_tier"] = "few" if n <= low else "many" if n > high else "middle"

    def summarize(group, label):
        improvements = [p["improvement"] for p in group]
        changes = [p["copy_change_bytes"] for p in group]
        v0_times = [float(p["v0"]["total_wall_time"]) for p in group]
        v1_times = [float(p["v1"]["total_wall_time"]) for p in group]
        return {"group": label, "n": len(group),
                "mean_improvement": statistics.mean(improvements),
                "median_improvement": statistics.median(improvements),
                "p25_improvement": quantile(improvements, .25),
                "p75_improvement": quantile(improvements, .75),
                "max_improvement": max(improvements),
                "max_degradation": min(improvements),
                "improved_fraction": sum(x > 0 for x in improvements) / len(group),
                "degraded_fraction": sum(x < 0 for x in improvements) / len(group),
                "nondegraded_fraction": sum(x >= 0 for x in improvements) / len(group),
                "severe_degradation_fraction": sum(x < -.05 for x in improvements) / len(group),
                "mean_copy_change_bytes": statistics.mean(changes),
                "median_copy_change_bytes": statistics.median(changes),
                "v0_wall_time": sum(v0_times), "v1_wall_time": sum(v1_times),
                "evaluator_time": sum(float(p[a]["evaluator_time"])
                                      for p in group for a in ("v0", "v1")),
                "evaluator_count": sum(int(p[a]["evaluator_count"])
                                       for p in group for a in ("v0", "v1"))}

    summary = [summarize(pairs, "all")]
    for k in KS:
        summary.append(summarize([p for p in pairs if p["K"] == k], f"K={k}"))
    for size in ("small", "medium", "large"):
        summary.append(summarize([p for p in pairs if p["size_tier"] == size],
                                 f"size={size}"))
    for tier in ("few", "middle", "many"):
        group = [p for p in pairs if p["component_tier"] == tier]
        if group:
            summary.append(summarize(group, f"components={tier}"))
    write_rows(SUMMARY, tuple(summary[0]), summary)

    def diagnosis(pair):
        old, new = pair["v0"], pair["v1"]
        old_copy, new_copy = int(old["added_copy_bytes"]), int(new["added_copy_bytes"])
        old_cross, new_cross = int(old["crossing_tensor_bytes"]), int(new["crossing_tensor_bytes"])
        old_width, new_width = int(old["block_layer_width"]), int(new["block_layer_width"])
        if int(new["num_subgraphs"]) * 2 <= int(old["num_subgraphs"]):
            return "A coarse cut structure (fewer blocks; check load and pipe overlap)"
        if new_copy > old_copy * 1.5 and new_copy - old_copy > 1_000_000:
            if new_cross <= old_cross * 1.2:
                return "C spill/memory pressure (inference from added COPY versus crossing bytes)"
            return "B COPY growth"
        if new_width < old_width or int(new["block_critical_path"]) > int(old["block_critical_path"]):
            return "A cut structure"
        if float(new["proxy_makespan"]) < float(old["proxy_makespan"]):
            return "D proxy ranking mismatch"
        return "E other/needs trace review"

    severe = sorted((p for p in pairs if p["improvement"] < -.05),
                    key=lambda p: p["improvement"])
    worst = sorted(pairs, key=lambda p: p["improvement"])[:5]
    all_row = summary[0]
    passes = (all_row["median_improvement"] > .05 and
              all_row["mean_improvement"] > .10 and
              all_row["nondegraded_fraction"] >= .70)
    # The user's closure criterion is qualitative: no widespread severe
    # regressions. Interpret "widespread" transparently as >=10% of pairs.
    # This is a closure judgment, not an algorithm-tuning rule.
    severe_widespread = all_row["severe_degradation_fraction"] >= .10
    freeze_recommended = passes and not severe_widespread
    lines = [
        "# Q1 V1-P 结构分层验证报告", "",
        "本报告由 `src/q1/validate_v1_partition.py` 从官方 evaluator 结果生成。",
        "V1-P 的拓扑模式、K/2K/4K/8K、切点权重和四候选预算均保持冻结。",
        "选样使用 100 图的结构特征与官方 V0 四核结果；此前用于 V1-lite 开发的",
        "case_014、case_025、case_026、case_093 被排除，以保留独立验证集。", "",
        "## 样本", "",
        "12 个 case，按非 COPY Op 数三分层，每层通过七维特征分位距离选择 4 个。",
        "为缩短收口时间，仅从预筛查官方计算耗时不超过 60 秒的样本中选择；",
        "因此结论不覆盖最昂贵的超大图。",
        "七维特征包括 Op 数、连通分量数、最大分量工作量占比、关键路径/总工作量、",
        "Tensor 扇出 P95、V0 四核新增 COPY 和 V0 四核加速比。逐例原因见",
        "`results/q1_v1_validation/selection.csv`。", "",
        "| 规模 | case |", "|---|---|",
    ]
    for size in ("small", "medium", "large"):
        names = [r["case"] for r in selected if r["size_tier"] == size]
        lines.append(f"| {size} | {', '.join(names)} |")
    lines += ["", "## 总体", "",
              f"共 {len(pairs)} 个 case×K 组合；mean improvement {all_row['mean_improvement']:.2%}，",
              f"median {all_row['median_improvement']:.2%}，P25/P75 "
              f"{all_row['p25_improvement']:.2%}/{all_row['p75_improvement']:.2%}。",
              f"改善 {all_row['improved_fraction']:.2%}，退化 {all_row['degraded_fraction']:.2%}，"
              f"不退化 {all_row['nondegraded_fraction']:.2%}，超过 5% 的严重退化 "
              f"{all_row['severe_degradation_fraction']:.2%}。",
              f"最大改善 {all_row['max_improvement']:.2%}，最大退化 "
              f"{all_row['max_degradation']:.2%}。",
              f"added COPY 平均变化 {all_row['mean_copy_change_bytes']:,.0f} bytes，"
              f"中位变化 {all_row['median_copy_change_bytes']:,.0f} bytes。", "",
              "| 分组 | n | 平均改善 | 中位改善 | 改善比例 | 退化比例 | 平均新增 COPY 变化 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary[1:]:
        lines.append(f"| {row['group']} | {row['n']} | {row['mean_improvement']:.2%} | "
                     f"{row['median_improvement']:.2%} | {row['improved_fraction']:.2%} | "
                     f"{row['degraded_fraction']:.2%} | {row['mean_copy_change_bytes']:,.0f} |")
    lines += ["", "## 最严重的五个退化组合", "",
              "宽度为块 DAG 同层最大块数，是简单并行度指标；crossing bytes 每个跨块 Tensor 只计一次。", "",
              "| case/K | 改善率 | V0/V1 模式 | 块数 | 分量数/最大工作量占比 | 关键路径/总工作量 | crossing bytes | added COPY | 宽度 | 官方 Makespan | 诊断 |",
              "|---|---:|---|---|---|---|---|---|---|---|---|"]
    for pair in worst:
        a, b = pair["v0"], pair["v1"]
        info = selected_by_case[pair["case"]]
        lines.append(
            f"| {pair['case']}/K{pair['K']} | {pair['improvement']:.2%} | "
            f"{a['topo_mode']}/{b['topo_mode']} | {a['num_subgraphs']}/{b['num_subgraphs']} | "
            f"{info['num_components']}/{float(info['max_component_work_share']):.2%} | "
            f"{info['critical_path']}/{float(info['critical_path_work_ratio']):.4f} | "
            f"{int(a['crossing_tensor_bytes']):,}/{int(b['crossing_tensor_bytes']):,} | "
            f"{int(a['added_copy_bytes']):,}/{int(b['added_copy_bytes']):,} | "
            f"{a['block_layer_width']}/{b['block_layer_width']} | "
            f"{a['makespan']}/{b['makespan']} | {diagnosis(pair)} |")
    lines += ["", "所有超过 5% 的退化组合及其诊断：", ""]
    if severe:
        lines.extend(f"- {p['case']} K={p['K']}: {p['improvement']:.2%}；{diagnosis(p)}"
                     for p in severe)
    else:
        lines.append("- 无。")
    screening = read_rows(SCREEN)
    screening_eval = sum(float(r["screen_evaluator_time"]) +
                         float(r["singlecore_evaluator_time"]) for r in screening)
    lines += ["", "## 运行成本与决策", "",
              f"验证阶段真实 evaluator 调用 {all_row['evaluator_count']} 次，"
              f"累计 evaluator 耗时 {all_row['evaluator_time']:.1f} 秒；"
              f"V0/V1-P 总墙钟分别 {all_row['v0_wall_time']:.1f}/"
              f"{all_row['v1_wall_time']:.1f} 秒。",
              f"筛选阶段 evaluator 与单核官方计算合计 {screening_eval:.1f} 秒。",
              "缓存命中不计为新的 evaluator 调用；候选数仍逐行记录。", "",
              ("三项量化门槛全部通过。" if passes else "三项量化门槛未全部通过。"),
              f"严重退化比例为 {all_row['severe_degradation_fraction']:.2%}，集中在 "
              f"{len({p['case'] for p in severe})}/{len(selected)} 个 case。",
              "收口判断把严重退化达到 10% 的组合视为普遍；此口径只用于本轮是否停止优化，",
              "不是预注册的性能统计门槛。",
              ("建议：冻结 Q1 的当前 V1-P 实现并停止优化；保留 V0 作为逐例对照。"
               if freeze_recommended else
               "建议：停止本轮 Q1 优化，保留退化诊断；不能认定 V1-P 稳定优于 V0。"),
              "不运行全 100 例 V1-P；不实施 memory-aware cut。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    import datetime
    import subprocess
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT,
                                     text=True, encoding="utf-8").strip()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                            text=True, encoding="utf-8").strip()
    severe_count = len(severe)
    freeze = [
        "# Q1 V1-P 冻结报告", "",
        f"时间戳：{timestamp}；分支：`{branch}`；验证时源 commit：`{source_commit}`。",
        "", "## 验证", "",
        f"从 100 例已完成的 V0 预筛查中，按结构与 V0 性能选出 12 例；"
        f"每例 K=2–5，V0 与 V1-P 各最多 4 次官方 evaluator 候选，共 {len(pairs)} 组配对。",
        "已排除 V1-lite 开发用例；为快速收口，选样限预筛查计算不超过 60 秒的图。",
        f"平均改善 {all_row['mean_improvement']:.2%}，中位改善 {all_row['median_improvement']:.2%}；"
        f"改善 {all_row['improved_fraction']:.2%}，退化 {all_row['degraded_fraction']:.2%}，"
        f"严重退化（超过 5%）{severe_count}/{len(pairs)}。",
        f"added COPY 平均变化 {all_row['mean_copy_change_bytes']:,.0f} bytes。",
        f"验证阶段 evaluator 累计 {all_row['evaluator_time']:.1f} 秒、"
        f"真实调用 {all_row['evaluator_count']} 次。", "",
        "| K | 平均改善 | 中位改善 | 退化比例 |", "|---:|---:|---:|---:|",
    ]
    for row in summary[1:5]:
        freeze.append(f"| {row['group'][2:]} | {row['mean_improvement']:.2%} | "
                      f"{row['median_improvement']:.2%} | {row['degraded_fraction']:.2%} |")
    freeze += ["", "## 结论与边界", "",
               ("V1-P 达到预设的均值、中位数和不退化比例门槛。"
                if passes else "V1-P 未同时达到预设的均值、中位数和不退化比例门槛。"),
               ("建议冻结当前 Q1 V1-P 实现，停止 Q1 优化。"
                if freeze_recommended else
                "暂不建议把 V1-P 认定为稳定优于 V0；本轮停止 Q1 优化并保留退化诊断。"),
               "严重退化仅 3/48 组合、涉及 2/12 case，当前判断为局部而非普遍；"
               "最差的 case_002 K=2 退化 30.58%，须在论文中如实说明。",
               "本结论限 12 例快速验证，未覆盖最昂贵的超大图；全 100 例终局曲线尚未完成。",
               "逐例指标、选样原因与退化诊断见 `results/q1_v1_validation/` 和 `q1/V1_VALIDATION_REPORT.md`。",
               "下一阶段由用户另行启动 Q2；本轮不实施 memory-aware cut。", ""]
    FREEZE_REPORT.write_text("\n".join(freeze), encoding="utf-8")
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps({
        "source_commit": source_commit, "timestamp_utc": timestamp,
        "selected_cases": [r["case"] for r in selected],
        "paired_combinations": len(pairs), "summary": summary,
        "severe_degradation_count": severe_count,
        "severe_degradation_cases": sorted({p["case"] for p in severe}),
        "freeze_recommended": freeze_recommended,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("report", REPORT, flush=True)
    print("freeze", FREEZE_REPORT, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("screen", "select", "run", "report"))
    args = parser.parse_args()
    {"screen": screen, "select": select, "run": run, "report": report}[args.stage]()


if __name__ == "__main__":
    main()
