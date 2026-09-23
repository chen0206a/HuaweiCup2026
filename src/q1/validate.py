"""独立方案校验：不通过本模块的方案禁止送官方 evaluator。

校验项对应官方 `derive_multicore_plan` 的入口条件，但这里是独立实现；
另外调用官方推导做交叉核对，确认我们算出的子图依赖与官方完全一致
（官方用 dependency_pairs 决定 Task 跨核等待，算错会直接改变 Makespan）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .official import derive_multicore_plan
from .problem import COPY_TYPES, GraphModelError, topological_order


@dataclass
class ValidationReport:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    num_subgraphs: int = 0
    num_cores: int = 0
    per_core_counts: list = field(default_factory=list)

    def raise_if_invalid(self):
        if not self.ok:
            raise GraphModelError("方案校验失败:\n  - " + "\n  - ".join(self.errors))
        return self


def validate_plan(problem, plan) -> ValidationReport:
    """校验 node_to_subgraph / core_schedules 的合法性与可执行性。"""
    errors, warnings = [], []

    if not isinstance(plan, dict) or set(plan) != {"node_to_subgraph", "core_schedules"}:
        return ValidationReport(False, ["方案必须且只能包含 node_to_subgraph 与 core_schedules"])

    raw_mapping = plan["node_to_subgraph"]
    schedules = plan["core_schedules"]

    if not isinstance(raw_mapping, dict):
        return ValidationReport(False, ["node_to_subgraph 必须是对象"])
    if not isinstance(schedules, list) or not schedules:
        return ValidationReport(False, ["core_schedules 必须是非空列表"])
    if any(not isinstance(order, list) for order in schedules):
        return ValidationReport(False, ["core_schedules 的每一项必须是子图 id 列表"])

    mapping = {}
    for key, value in raw_mapping.items():
        if isinstance(key, bool) or not isinstance(key, int):
            if not (isinstance(key, str) and key.isascii() and key.isdecimal()):
                errors.append(f"node_to_subgraph 的键必须是整数 op id；得到 {key!r}")
                continue
            key = int(key)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"子图 id 必须是整数；op {key} 得到 {value!r}")
            continue
        mapping[key] = value
    if errors:
        return ValidationReport(False, errors)
    if len(mapping) != len(raw_mapping):
        errors.append("node_to_subgraph 含重复的整数 op id")

    # 覆盖全部非 COPY 操作，且不包含 COPY_IN / COPY_OUT
    eligible = set(problem.nc_ids)
    missing = sorted(eligible - set(mapping))
    extra = sorted(set(mapping) - eligible)
    if missing:
        errors.append(f"漏掉 {len(missing)} 个非 COPY 操作，例如 {missing[:10]}")
    if extra:
        copy_like = [op for op in extra
                     if problem.ops_by_id.get(op, {}).get("op") in COPY_TYPES]
        errors.append(
            f"node_to_subgraph 含 {len(extra)} 个非法节点，例如 {extra[:10]}"
            + (f"（其中 {len(copy_like)} 个是 COPY_IN/COPY_OUT）" if copy_like else ""))
    if errors:
        return ValidationReport(False, errors)

    if any(sgid < 0 for sgid in mapping.values()):
        errors.append("子图 id 必须为非负整数")

    subgraph_ids = set(mapping.values())
    scheduled = [sgid for order in schedules for sgid in order]
    if any(isinstance(sgid, bool) or not isinstance(sgid, int) for sgid in scheduled):
        # 类型不对时后续 set()/sorted() 会直接抛异常，先返回
        return ValidationReport(False, errors + ["core_schedules 中的子图 id 必须是整数"])
    if len(scheduled) != len(set(scheduled)):
        seen, dup = set(), set()
        for sgid in scheduled:
            (dup if sgid in seen else seen).add(sgid)
        errors.append(f"同一子图在 core_schedules 中重复出现: {sorted(dup)[:10]}")
    if set(scheduled) != subgraph_ids:
        errors.append(
            f"core_schedules 未恰好覆盖所有子图；缺失 {sorted(subgraph_ids - set(scheduled))[:10]}，"
            f"多余 {sorted(set(scheduled) - subgraph_ids)[:10]}")
    if errors:
        return ValidationReport(False, errors)

    num_subgraphs = len(subgraph_ids)
    core_of = {sgid: core for core, order in enumerate(schedules) for sgid in order}

    # 子图 DAG 无环（独立重建，不信任上层结构）
    sg_preds = {sgid: set() for sgid in subgraph_ids}
    sg_succs = {sgid: set() for sgid in subgraph_ids}
    for src in problem.nc_ids:
        for dst in problem.nc_succs[src]:
            if mapping[src] != mapping[dst]:
                sg_succs[mapping[src]].add(mapping[dst])
                sg_preds[mapping[dst]].add(mapping[src])
    try:
        topological_order(sorted(subgraph_ids), sg_preds, sg_succs)
    except GraphModelError as error:
        errors.append(str(error))

    # 每核执行顺序不得违反子图依赖
    for core, order in enumerate(schedules):
        position = {sgid: index for index, sgid in enumerate(order)}
        for sgid in order:
            for pred in sg_preds[sgid]:
                if core_of[pred] == core and position[pred] >= position[sgid]:
                    errors.append(f"核 {core} 上顺序违反依赖: {pred} 必须在 {sgid} 之前")
    if errors:
        return ValidationReport(False, errors)

    # 与官方推导交叉核对：子图依赖必须逐条一致
    official = derive_multicore_plan(problem.graph, plan)
    ours = {(a, b) for a in subgraph_ids for b in sg_succs[a]}
    theirs = {tuple(pair) for pair in official["dependency_pairs"]}
    if ours != theirs:
        errors.append(
            "与官方 derive_multicore_plan 的子图依赖不一致；"
            f"本地多出 {sorted(ours - theirs)[:10]}，官方多出 {sorted(theirs - ours)[:10]}")

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        num_subgraphs=num_subgraphs,
        num_cores=len(schedules),
        per_core_counts=[len(order) for order in schedules],
    )
