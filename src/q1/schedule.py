"""问题 1 的列表调度（scene A）。

时序规则直接取自官方 `multicore_cut_evaluate_problem_1.task_release_time`：

    release = 0
    if 该核已有前一个 Task:  release = max(release, 前一 Task 完成 + 100)
    for p in 跨核前驱 Task:   release = max(release, p 完成 + 1000)

即 EST 是若干**时序约束取 max**：100 cycles 是同核前后 Task 的切换等待，
1000 cycles 是跨核前驱约束。二者都不是"依赖边数 × 常数"。

本模块的 Makespan 估计只是代理；真实 Makespan 一律来自官方 evaluator。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

SAME_CORE_WAIT = 100
CROSS_CORE_WAIT = 1000


@dataclass
class ScheduleResult:
    core_orders: list          # core_orders[c] = 该核按顺序执行的 sgid 列表
    core_of: dict              # sgid -> core id
    start: dict                # sgid -> EST
    finish: dict               # sgid -> EFT
    makespan_proxy: int
    assign_order: list         # 分配给子图的全局顺序（应是子图 DAG 的一个拓扑序）


def upstream_ranks(cost: dict, sg_succs: dict, topo_order: list) -> dict:
    """rank(s) = cost(s) + max(rank(t) for t in successors(s))；叶节点 rank = cost。"""
    rank = {}
    for sgid in reversed(topo_order):
        rank[sgid] = cost[sgid] + max(
            (rank[succ] for succ in sg_succs[sgid]), default=0)
    return rank


def estimate_start(sgid, core, core_finish, core_of, finish, sg_preds,
                   same_core_wait: int = SAME_CORE_WAIT,
                   cross_core_wait: int = CROSS_CORE_WAIT) -> int:
    """官方 `task_release_time` 的 EST。

    100 与 1000 都是**时序约束**，与依赖边条数无关：

    - 该核为空：不加同核切换项（对应官方 `core_previous_end is None`）；
    - 该核已有前一个 Task：`前一个 Task 完成 + 100`；
    - 跨核前驱：`前驱完成 + 1000`，多前驱取 max，不相加。
    """
    est = 0
    if core_finish[core] is not None:
        est = max(est, core_finish[core] + same_core_wait)
    for pred in sg_preds[sgid]:
        if core_of[pred] != core:
            est = max(est, finish[pred] + cross_core_wait)
    return est


def list_schedule(partition, num_cores: int,
                  same_core_wait: int = SAME_CORE_WAIT,
                  cross_core_wait: int = CROSS_CORE_WAIT) -> ScheduleResult:
    """关键路径优先的列表调度，按最小 EFT 选核。

    确定性并列规则：rank 相同取较小 sgid；EFT 相同取当前总负载较小的核，
    再取较小 core id。
    """
    if num_cores <= 0:
        raise ValueError("num_cores must be positive")

    cost = partition.cost
    sg_preds, sg_succs = partition.sg_preds, partition.sg_succs
    rank = upstream_ranks(cost, sg_succs, partition.topo_order)

    pending_preds = {sg: len(sg_preds[sg]) for sg in cost}
    ready = [(-rank[sg], sg) for sg in cost if pending_preds[sg] == 0]
    heapq.heapify(ready)

    core_orders = [[] for _ in range(num_cores)]
    core_of, start, finish = {}, {}, {}
    core_finish = [None] * num_cores
    core_load = [0] * num_cores
    assign_order = []

    while ready:
        _, sgid = heapq.heappop(ready)

        best = None
        for core in range(num_cores):
            est = estimate_start(sgid, core, core_finish, core_of, finish,
                                 sg_preds, same_core_wait, cross_core_wait)
            eft = est + cost[sgid]
            key = (eft, core_load[core], core)
            if best is None or key < best[0]:
                best = (key, core, est, eft)

        _, core, est, eft = best
        core_of[sgid] = core
        start[sgid], finish[sgid] = est, eft
        core_orders[core].append(sgid)
        core_finish[core] = eft
        core_load[core] += cost[sgid]
        assign_order.append(sgid)

        for succ in sg_succs[sgid]:
            pending_preds[succ] -= 1
            if pending_preds[succ] == 0:
                heapq.heappush(ready, (-rank[succ], succ))

    if len(assign_order) != len(cost):
        raise RuntimeError("列表调度未能覆盖全部子图（子图 DAG 可能有环）")

    return ScheduleResult(
        core_orders=core_orders,
        core_of=core_of,
        start=start,
        finish=finish,
        makespan_proxy=max(finish.values(), default=0),
        assign_order=assign_order,
    )
