"""Q1 V1-lite: safe structure-aware topological partitions.

Every block is an interval of a valid topological order. This preserves a DAG
after contraction while allowing independent branches to stay together.
"""
from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict

from .partition import Partition, build_subgraph_dag, compute_cost
from .problem import GraphModelError


def _remaining_path(problem):
    remaining = {}
    for op in reversed(problem.topo_order):
        remaining[op] = problem.work[op] + max(
            (remaining[v] for v in problem.nc_succs[op]), default=0)
    return remaining


def _component_labels(problem):
    labels = {}
    for root in problem.nc_ids:
        if root in labels:
            continue
        stack = [root]
        labels[root] = root
        while stack:
            u = stack.pop()
            for v in problem.nc_preds[u] | problem.nc_succs[u]:
                if v not in labels:
                    labels[v] = root
                    stack.append(v)
    return labels


def topological_order_variant(problem, mode: str):
    if mode == "deterministic":
        return list(problem.topo_order)
    if mode not in {"critical", "branch"}:
        raise ValueError(f"unknown topological order mode: {mode}")
    path = _remaining_path(problem)
    labels = _component_labels(problem) if mode == "branch" else None
    indegree = {u: len(problem.nc_preds[u]) for u in problem.nc_ids}
    ready = []

    def key(u):
        return ((labels[u], -path[u], u) if labels is not None
                else (-path[u], u))

    for u in problem.nc_ids:
        if indegree[u] == 0:
            heapq.heappush(ready, (key(u), u))
    order = []
    while ready:
        _, u = heapq.heappop(ready)
        order.append(u)
        for v in problem.nc_succs[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(ready, (key(v), v))
    if len(order) != len(problem.nc_ids):
        raise GraphModelError("structure-aware topo order incomplete")
    return order


def boundary_features(problem, order):
    """One tensor contributes once per cut, irrespective of its fan-out edges.

    These are cut-selection features, not a claim of exact total DDR traffic.
    Exact scene-A boundary copies are counted by `scene_a_copy_bytes` below.
    """
    n = len(order)
    position = {op: i for i, op in enumerate(order)}
    size_delta = [0] * (n + 1)
    fanout_delta = [0.0] * (n + 1)
    edge_delta = [0] * (n + 1)
    critical_delta = [0] * (n + 1)
    eligible = set(problem.nc_ids)

    for tensor in problem.tensors.values():
        producers = tensor.producers & eligible
        consumers = tensor.consumers & eligible
        touched = producers | consumers
        if len(touched) < 2:
            continue
        lo = min(position[op] for op in touched)
        hi = max(position[op] for op in touched)
        if lo < hi:
            size_delta[lo + 1] += tensor.size
            size_delta[hi + 1] -= tensor.size
            if len(consumers) > 1:
                extra = tensor.size * math.log2(len(consumers))
                fanout_delta[lo + 1] += extra
                fanout_delta[hi + 1] -= extra

    forward = {}
    for u in problem.topo_order:
        forward[u] = problem.work[u] + max(
            (forward[p] for p in problem.nc_preds[u]), default=0)
    backward = _remaining_path(problem)
    longest = max(forward.values(), default=0)
    for u in problem.nc_ids:
        for v in problem.nc_succs[u]:
            a, b = position[u], position[v]
            if a >= b:
                raise GraphModelError("candidate order is not topological")
            edge_delta[a + 1] += 1
            edge_delta[b + 1] -= 1
            if forward[u] + backward[v] == longest:
                critical_delta[a + 1] += 1
                critical_delta[b + 1] -= 1

    def accumulate(delta):
        result = [0] * (n + 1)
        running = 0
        for i in range(n + 1):
            running += delta[i]
            result[i] = running
        return result

    return (accumulate(size_delta), accumulate(edge_delta),
            accumulate(critical_delta), accumulate(fanout_delta))


def scene_a_copy_bytes(problem, node_to_sg):
    """Exact pre-spill COPY bytes from official scene-A Task boundary rules."""
    eligible = set(problem.nc_ids)
    scheduled = 0
    for tensor in problem.tensors.values():
        producer_blocks = {node_to_sg[u] for u in tensor.producers & eligible}
        consumer_blocks = {node_to_sg[u] for u in tensor.consumers & eligible}
        original_copy_out = any(
            problem.ops_by_id[u]["op"] == "COPY_OUT"
            for u in tensor.consumers if u in problem.ops_by_id)
        for block in producer_blocks | consumer_blocks:
            if block in consumer_blocks and block not in producer_blocks:
                scheduled += tensor.size
            if block in producer_blocks and (
                original_copy_out or not consumer_blocks
                or bool(consumer_blocks - {block})
            ):
                scheduled += tensor.size
    return scheduled


def choose_cuts(problem, order, count):
    n = len(order)
    count = min(max(1, count), n)
    if count == 1:
        return [0, n]
    prefix = [0]
    for u in order:
        prefix.append(prefix[-1] + problem.work[u])
    total = prefix[-1]
    target = max(1.0, total / count)
    bytes_at, edges_at, critical_at, fanout_at = boundary_features(problem, order)
    sample = [bytes_at[i] for i in range(1, n, max(1, n // 1000))]
    normal_bytes = max(1, sorted(sample)[len(sample) // 2]) if sample else 1
    sample_edges = [edges_at[i] for i in range(1, n, max(1, n // 1000))]
    normal_edges = max(1, sorted(sample_edges)[len(sample_edges) // 2]) if sample_edges else 1
    cuts = [0]
    for j in range(1, count):
        start = cuts[-1]
        last = n - (count - j)
        desired = total * j / count
        radius = max(target * 0.4, max(problem.work[u] for u in order) * 0.5)
        lo = max(start + 1, bisect.bisect_left(prefix, desired - radius))
        hi = min(last, bisect.bisect_right(prefix, desired + radius) - 1)
        if lo > hi:
            lo = hi = min(last, max(start + 1, bisect.bisect_left(prefix, desired)))

        def score(i):
            balance = abs((prefix[i] - prefix[start]) - target) / target
            boundary = bytes_at[i] / normal_bytes
            edge = edges_at[i] / normal_edges
            critical = critical_at[i]
            fanout = fanout_at[i] / normal_bytes
            return (balance + 0.35 * boundary + 0.08 * edge
                    + 0.25 * critical + 0.08 * fanout,
                    abs(prefix[i] - desired), i)

        cuts.append(min(range(lo, hi + 1), key=score))
    cuts.append(n)
    return cuts


def partition_v1(problem, num_subgraphs: int, mode: str = "branch") -> Partition:
    order = topological_order_variant(problem, mode)
    cuts = choose_cuts(problem, order, num_subgraphs)
    blocks = [order[a:b] for a, b in zip(cuts, cuts[1:])]
    mapping = {op: sg for sg, block in enumerate(blocks) for op in block}
    if set(mapping) != set(problem.nc_ids) or any(not block for block in blocks):
        raise GraphModelError("V1 partition lost an operation or created an empty block")
    preds, succs, topo = build_subgraph_dag(problem, mapping, len(blocks))
    cost = {sg: compute_cost(problem, block) for sg, block in enumerate(blocks)}
    return Partition(mapping, blocks, preds, succs, cost, topo)
