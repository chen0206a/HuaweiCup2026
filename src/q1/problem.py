"""问题 1 的数据模型：非 COPY Op DAG + Tensor 索引。

同时维护两份信息，二者缺一不可：

1. **Tensor 索引**（`Tensor`）：id / size / pos / producers / consumers。
   计算依赖与边界搬运都必须靠它追踪，不能丢。
2. **非 COPY Op DAG**（`Problem.nc_preds` / `nc_succs`）：依赖、拓扑排序、
   关键路径、子图 DAG 都在它上面做。

依赖的建立方式与官方 `stub_multicore_cut_and_schedule._build_op_adjacency`
+ `_contract_excluded_copy_nodes` 完全一致：先把 op-tensor 二部图展开成 op 图
（同一 tensor 的每个生产者 → 每个消费者），再沿 COPY 节点做**可达闭包**，
得到非 COPY op 之间的依赖。这正是"按 Tensor 数据流追踪"：
DDR → COPY_IN → tensor → compute、compute → tensor → compute、
compute → tensor → COPY_OUT → DDR，以及一个 tensor 被多个 compute 消费，
都由同一套 producer/consumer 关系表达。
"""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

COPY_TYPES = frozenset({"COPY_IN", "COPY_OUT"})
# 官方 100 个 case 中非 COPY 操作只使用 PIPE_M / PIPE_V（COPY_IN=PIPE_MTE2，
# COPY_OUT=PIPE_MTE3）。出现别的非 COPY Pipe 时不做猜测，直接报错。
COMPUTE_PIPES = frozenset({"PIPE_M", "PIPE_V"})


class GraphModelError(RuntimeError):
    """图结构或数据模型不合法。"""


@dataclass(frozen=True)
class Tensor:
    id: int
    size: int
    pos: str
    producers: frozenset
    consumers: frozenset


def load_graph(path) -> dict:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise GraphModelError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object)


def build_tensor_index(graph_json: dict) -> dict:
    """op-tensor 二部图 → {tensor_id: Tensor}，保留全部 producer/consumer。"""
    op_ids = {op["id"] for op in graph_json["ops"]}
    producers = defaultdict(set)
    consumers = defaultdict(set)
    for edge in graph_json["edges"]:
        src, dst = edge["source"], edge["target"]
        src_is_op, dst_is_op = src in op_ids, dst in op_ids
        if src_is_op and not dst_is_op:
            producers[dst].add(src)
        elif dst_is_op and not src_is_op:
            consumers[src].add(dst)

    index = {}
    for tensor in graph_json["tensors"]:
        tid = tensor["id"]
        index[tid] = Tensor(
            id=tid,
            size=tensor["size"],
            pos=tensor["pos"],
            producers=frozenset(producers.get(tid, ())),
            consumers=frozenset(consumers.get(tid, ())),
        )
    return index


def build_full_op_adjacency(graph_json: dict):
    """全部 op（含 COPY）之间的直接依赖，由 tensor 数据流与 op-op 直接边组成。"""
    op_ids = {op["id"] for op in graph_json["ops"]}
    preds = {op_id: set() for op_id in op_ids}
    succs = {op_id: set() for op_id in op_ids}
    producers = defaultdict(set)
    consumers = defaultdict(set)
    for edge in graph_json["edges"]:
        src, dst = edge["source"], edge["target"]
        src_is_op, dst_is_op = src in op_ids, dst in op_ids
        if src_is_op and dst_is_op and src != dst:
            succs[src].add(dst)
            preds[dst].add(src)
        elif src_is_op and not dst_is_op:
            producers[dst].add(src)
        elif not src_is_op and dst_is_op:
            consumers[src].add(dst)
    for tid, producer_ids in producers.items():
        for src in producer_ids:
            for dst in consumers.get(tid, ()):
                if src != dst:
                    succs[src].add(dst)
                    preds[dst].add(src)
    return preds, succs


def contract_through_copy(eligible_ids, full_succs):
    """沿 COPY 节点做可达闭包，得到非 COPY op 之间的依赖。

    对每个 eligible 源点做一次带 visited 的前进遍历，跳过非 eligible 节点。
    与官方 `_contract_excluded_copy_nodes` 语义相同。
    """
    eligible = set(eligible_ids)
    succs = {node: set() for node in eligible_ids}
    preds = {node: set() for node in eligible_ids}
    for src in eligible_ids:
        stack = list(full_succs.get(src, ()))
        visited = set()
        while stack:
            dst = stack.pop()
            if dst in eligible:
                if dst != src:
                    succs[src].add(dst)
                    preds[dst].add(src)
                continue
            if dst in visited:
                continue
            visited.add(dst)
            stack.extend(full_succs.get(dst, ()))
    return preds, succs


def topological_order(node_ids, preds, succs) -> list:
    """确定性 Kahn 拓扑排序：并列候选按 op id 升序。

    出现环时立即报错并给出未解析的节点集合，不静默修复。
    """
    node_ids = list(node_ids)
    node_set = set(node_ids)
    indegree = {node: sum(pred in node_set for pred in preds[node]) for node in node_ids}
    ready = [node for node in node_ids if indegree[node] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for succ in sorted(succs[node]):
            if succ not in node_set:
                continue
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(ready, succ)
    if len(order) != len(node_ids):
        unresolved = sorted(node_set - set(order))
        raise GraphModelError(
            f"非 COPY Op DAG 存在环；未解析节点（最多 20 个）={unresolved[:20]} "
            f"共 {len(unresolved)} 个")
    return order


class Problem:
    """一个 case 的问题 1 视图。"""

    def __init__(self, graph_json: dict, name: str = "graph"):
        if not isinstance(graph_json, dict):
            raise GraphModelError("graph must be an object")
        for field in ("ops", "tensors", "edges"):
            if not isinstance(graph_json.get(field), list):
                raise GraphModelError(f"graph.{field} must be a list")
        self.name = name
        self.graph = graph_json
        self.ops_by_id = {op["id"]: op for op in graph_json["ops"]}
        self.tensors = build_tensor_index(graph_json)
        self.all_op_ids = sorted(self.ops_by_id)
        self.nc_ids = sorted(
            op_id for op_id, op in self.ops_by_id.items()
            if op["op"] not in COPY_TYPES)
        self.copy_ids = sorted(set(self.all_op_ids) - set(self.nc_ids))

        self.full_preds, self.full_succs = build_full_op_adjacency(graph_json)
        self.nc_preds, self.nc_succs = contract_through_copy(
            self.nc_ids, self.full_succs)

        self._check_compute_pipes()
        self.work = {op_id: int(self.ops_by_id[op_id]["cycles"])
                     for op_id in self.nc_ids}
        self.topo_order = topological_order(
            self.nc_ids, self.nc_preds, self.nc_succs)

    @classmethod
    def from_file(cls, path) -> "Problem":
        path = Path(path)
        return cls(load_graph(path), name=path.stem)

    def _check_compute_pipes(self):
        bad = {}
        for op_id in self.nc_ids:
            pipe = self.ops_by_id[op_id].get("pipe")
            if pipe not in COMPUTE_PIPES:
                bad.setdefault(pipe, 0)
                bad[pipe] += 1
        if bad:
            raise GraphModelError(
                "非 COPY 操作出现了 PIPE_M / PIPE_V 之外的 Pipe，官方语义未核实，"
                f"拒绝猜测处理: {bad}")

    # ---- 便于测试与调试的派生视图 ----

    def pipe_work(self, op_ids) -> dict:
        """按 Pipe 汇总 cycles（只统计非 COPY 操作）。"""
        totals = defaultdict(int)
        for op_id in op_ids:
            totals[self.ops_by_id[op_id]["pipe"]] += self.work[op_id]
        return dict(totals)

    def internal_critical_path(self, op_ids) -> int:
        """仅在给定节点集合内、按非 COPY Op DAG 与 cycles 计算关键路径。

        这是快速调度代理，不代表官方真实执行时间：它不建模 DDR 争用、
        spill、L1/UB 驻留与四 Pipe 时间线。真实时间只来自官方 evaluator。
        """
        members = set(op_ids)
        if not members:
            return 0
        local_preds = {
            node: [p for p in self.nc_preds[node] if p in members]
            for node in members
        }
        order = topological_order(sorted(members), local_preds, self.nc_succs)
        longest = {}
        for node in order:
            best_pred = max((longest[p] for p in local_preds[node]), default=0)
            longest[node] = best_pred + self.work[node]
        return max(longest.values())

    def summary(self) -> dict:
        from collections import Counter
        return {
            "name": self.name,
            "num_ops_total": len(self.all_op_ids),
            "num_ops_non_copy": len(self.nc_ids),
            "num_copy_ops": len(self.copy_ids),
            "num_tensors": len(self.tensors),
            "num_edges": len(self.graph["edges"]),
            "op_types": dict(Counter(op["op"] for op in self.graph["ops"])),
            "total_cycles": sum(self.work.values()),
            "m_cycles": sum(self.work[o] for o in self.nc_ids
                            if self.ops_by_id[o]["pipe"] == "PIPE_M"),
            "v_cycles": sum(self.work[o] for o in self.nc_ids
                            if self.ops_by_id[o]["pipe"] == "PIPE_V"),
        }
