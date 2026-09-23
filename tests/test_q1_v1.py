"""Focused correctness tests for Q1 V1-lite."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q1.partition import Partition  # noqa: E402
from src.q1.partition_v1 import (  # noqa: E402
    boundary_features, partition_v1, scene_a_copy_bytes, topological_order_variant)
from src.q1.problem import Problem  # noqa: E402
from src.q1.schedule_v1 import global_aware_schedule  # noqa: E402
from src.q1.solver import build_plan  # noqa: E402
from src.q1.validate import validate_plan  # noqa: E402


def op(i, cycles=10):
    return {"id": i, "op": "ADD", "pipe": "PIPE_V", "cycles": cycles}


def graph(ops, tensors, edges):
    return {"ops": ops, "tensors": tensors, "edges": edges}


def test_independent_branches_remain_separate():
    p = Problem(graph(
        [op(i) for i in (10, 11, 12, 13)],
        [{"id": 1, "pos": "UB", "size": 64},
         {"id": 2, "pos": "UB", "size": 64}],
        [{"source": 10, "target": 1}, {"source": 1, "target": 12},
         {"source": 11, "target": 2}, {"source": 2, "target": 13}],
    ))
    order = topological_order_variant(p, "branch")
    assert order == [10, 12, 11, 13]
    part = partition_v1(p, 2, "branch")
    assert [set(block) for block in part.blocks] == [{10, 12}, {11, 13}]
    assert all(not deps for deps in part.sg_succs.values())
    plan = build_plan(part, global_aware_schedule(part, 2), 2)
    assert validate_plan(p, plan).ok


def test_tensor_aware_fanout_and_large_boundary():
    p = Problem(graph(
        [op(10), op(11), op(12)],
        [{"id": 1, "pos": "UB", "size": 1000}],
        [{"source": 10, "target": 1}, {"source": 1, "target": 11},
         {"source": 1, "target": 12}],
    ))
    size, deps, critical, fanout = boundary_features(p, [10, 11, 12])
    assert size[1] == 1000 and size[2] == 1000
    assert deps[1] == 2 and deps[2] == 1
    assert fanout[1] > 0
    assert critical[1] >= 1
    assert scene_a_copy_bytes(p, {10: 0, 11: 1, 12: 2}) == 3000
    assert scene_a_copy_bytes(p, {10: 0, 11: 1, 12: 1}) == 2000


def test_global_scheduler_uses_independent_cores_and_max_wait():
    part = Partition(
        node_to_sg={10: 0, 11: 1}, blocks=[[10], [11]],
        sg_preds={0: set(), 1: set()},
        sg_succs={0: set(), 1: set()},
        cost={0: 500, 1: 500}, topo_order=[0, 1])
    scheduled = global_aware_schedule(part, 2)
    assert [len(x) for x in scheduled.core_orders] == [1, 1]
    assert scheduled.makespan_proxy == 500

    chain = Partition(
        node_to_sg={10: 0, 11: 1, 12: 2}, blocks=[[10], [11], [12]],
        sg_preds={0: set(), 1: {0}, 2: {0, 1}},
        sg_succs={0: {1, 2}, 1: {2}, 2: set()},
        cost={0: 10, 1: 10, 2: 10}, topo_order=[0, 1, 2])
    scheduled = global_aware_schedule(chain, 2)
    for s in (1, 2):
        expected = max((scheduled.finish[p] + 1000
                        for p in chain.sg_preds[s]
                        if scheduled.core_of[p] != scheduled.core_of[s]), default=0)
        assert scheduled.start[s] >= expected


def main():
    tests = [(name, f) for name, f in sorted(globals().items())
             if name.startswith("test_") and callable(f)]
    for name, f in tests:
        f()
        print("PASS", name)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
