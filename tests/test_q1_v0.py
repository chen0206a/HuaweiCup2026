"""问题 1 Baseline / V0 单元测试（TEST 1–6）。

运行：python tests/test_q1_v0.py   或   pytest tests/test_q1_v0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.q1.official import derive_multicore_plan  # noqa: E402
from src.q1.partition import (  # noqa: E402
    candidate_granularities, compute_cost, contiguous_blocks, partition)
from src.q1.paths import case_path  # noqa: E402
from src.q1.problem import Problem, build_tensor_index, topological_order  # noqa: E402
from src.q1.schedule import list_schedule  # noqa: E402
from src.q1.solver import build_plan  # noqa: E402
from src.q1.validate import validate_plan  # noqa: E402


def graph(ops, tensors, edges):
    return {"ops": ops, "tensors": tensors, "edges": edges}


def pipe_op(op_id, cycles, name="ADD", pipe="PIPE_V"):
    return {"id": op_id, "op": name, "pipe": pipe, "cycles": cycles}


# ---------------------------------------------------------------- TEST 1
def test_01_chain():
    """A -> B -> C：DAG 正确、拓扑序确定、连续分块合法。"""
    g = graph(
        ops=[pipe_op(100, 10), pipe_op(101, 20), pipe_op(102, 30)],
        tensors=[{"id": 1, "size": 64, "pos": "UB"},
                 {"id": 2, "size": 64, "pos": "UB"}],
        edges=[{"source": 100, "target": 1}, {"source": 1, "target": 101},
               {"source": 101, "target": 2}, {"source": 2, "target": 102}],
    )
    problem = Problem(g, name="chain")
    assert problem.nc_ids == [100, 101, 102]
    assert problem.nc_preds[100] == set()
    assert problem.nc_preds[101] == {100}
    assert problem.nc_preds[102] == {101}
    assert problem.topo_order == [100, 101, 102]
    # 确定性：重复构建结果一致
    assert Problem(g, name="chain").topo_order == problem.topo_order

    assert problem.internal_critical_path([100, 101, 102]) == 60

    # 按工作量均分（各 30 cycles），而不是按节点数对半切
    blocks = contiguous_blocks(problem.topo_order, problem.work, 2)
    assert blocks == [[100, 101], [102]]
    part = partition(problem, 2)
    assert part.num_subgraphs == 2
    assert part.node_to_sg == {100: 0, 101: 0, 102: 1}
    assert part.sg_succs[0] == {1} and part.sg_succs[1] == set()
    assert part.cost[0] == 30 and part.cost[1] == 30
    assert [b for b in part.blocks] == [[100, 101], [102]]

    report = validate_plan(problem, build_plan(part, list_schedule(part, 2), 2))
    assert report.ok, report.errors


# ---------------------------------------------------------------- TEST 2
def test_02_fanout():
    """A -> B、A -> C：扇出依赖完整，Tensor 索引保留两个消费者。"""
    g = graph(
        ops=[pipe_op(100, 10), pipe_op(101, 20), pipe_op(102, 30)],
        tensors=[{"id": 1, "size": 64, "pos": "UB"}],
        edges=[{"source": 100, "target": 1}, {"source": 1, "target": 101},
               {"source": 1, "target": 102}],
    )
    problem = Problem(g, name="fanout")
    assert problem.nc_succs[100] == {101, 102}
    assert problem.nc_preds[101] == {100}
    assert problem.nc_preds[102] == {100}
    assert problem.topo_order == [100, 101, 102]

    tensor = problem.tensors[1]
    assert tensor.producers == {100} and tensor.consumers == {101, 102}

    # 把 100 单独放一块，两个消费者各占一块，检查扇出被完整投影到子图 DAG
    plan = {"node_to_subgraph": {100: 0, 101: 1, 102: 2},
            "core_schedules": [[0], [1], [2]]}
    report = validate_plan(problem, plan)
    assert report.ok, report.errors
    assert set(derive_multicore_plan(g, plan)["dependency_pairs"]) == {(0, 1), (0, 2)}

    part = partition(problem, 2)
    assert validate_plan(problem, build_plan(
        part, list_schedule(part, 2), 2)).ok


# ---------------------------------------------------------------- TEST 3
def test_03_independent_branches():
    """A -> B 与 C -> D 之间不得产生虚假依赖。"""
    g = graph(
        ops=[pipe_op(100, 10), pipe_op(101, 20), pipe_op(102, 30), pipe_op(103, 40)],
        tensors=[{"id": 1, "size": 64, "pos": "UB"},
                 {"id": 2, "size": 64, "pos": "UB"}],
        edges=[{"source": 100, "target": 1}, {"source": 1, "target": 101},
               {"source": 102, "target": 2}, {"source": 2, "target": 103}],
    )
    problem = Problem(g, name="branches")
    assert problem.nc_succs[100] == {101}
    assert problem.nc_succs[102] == {103}
    assert problem.nc_preds[100] == set() and problem.nc_preds[102] == set()
    # 两条分支之间没有边
    assert problem.nc_succs[101] == set() and problem.nc_succs[103] == set()
    assert not (problem.nc_succs[100] & {102, 103})

    # 每条分支各自成块：子图之间不应出现任何依赖
    plan = {"node_to_subgraph": {100: 0, 101: 0, 102: 1, 103: 1},
            "core_schedules": [[0], [1]]}
    report = validate_plan(problem, plan)
    assert report.ok, report.errors
    assert derive_multicore_plan(g, plan)["dependency_pairs"] == []


# ---------------------------------------------------------------- TEST 4
def test_04_copy_contraction():
    """DDR -> COPY_IN -> A -> COPY_OUT -> DDR：非 COPY DAG 只有 A。"""
    g = graph(
        ops=[{"id": 10, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 2},
             pipe_op(100, 50),
             {"id": 11, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 2}],
        tensors=[{"id": 1, "size": 120, "pos": "DDR"},
                 {"id": 2, "size": 120, "pos": "UB"},
                 {"id": 3, "size": 120, "pos": "DDR"}],
        edges=[{"source": 1, "target": 10}, {"source": 10, "target": 2},
               {"source": 100, "target": 2},
               {"source": 2, "target": 11}, {"source": 11, "target": 3}],
    )
    problem = Problem(g, name="copy_roundtrip")
    assert problem.nc_ids == [100]
    assert problem.copy_ids == [10, 11]
    assert problem.nc_preds[100] == set() and problem.nc_succs[100] == set()
    assert problem.topo_order == [100]

    plan = {"node_to_subgraph": {100: 0}, "core_schedules": [[0]]}
    report = validate_plan(problem, plan)
    assert report.ok, report.errors
    assert derive_multicore_plan(g, plan)["dependency_pairs"] == []

    # 中间隔着两跳 COPY 仍然连通
    g2 = graph(
        ops=[{"id": 10, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 2},
             pipe_op(100, 50),
             {"id": 11, "op": "COPY_OUT", "pipe": "PIPE_MTE3", "cycles": 2},
             {"id": 12, "op": "COPY_IN", "pipe": "PIPE_MTE2", "cycles": 2},
             pipe_op(101, 70)],
        tensors=[{"id": i, "size": 120, "pos": "DDR"} for i in (1, 3)] +
                [{"id": 2, "size": 120, "pos": "UB"}, {"id": 4, "size": 120, "pos": "UB"}],
        edges=[{"source": 1, "target": 10}, {"source": 10, "target": 2},
               {"source": 100, "target": 2}, {"source": 2, "target": 11},
               {"source": 11, "target": 3}, {"source": 3, "target": 12},
               {"source": 12, "target": 4}, {"source": 4, "target": 101}],
    )
    problem2 = Problem(g2, name="copy_two_hop")
    assert problem2.nc_succs[100] == {101}
    assert problem2.nc_preds[101] == {100}


# ---------------------------------------------------------------- TEST 5
def _chain_problem(depths=(10, 20, 30)):
    g = graph(
        ops=[pipe_op(100 + i, c) for i, c in enumerate(depths)],
        tensors=[{"id": 1, "size": 64, "pos": "UB"},
                 {"id": 2, "size": 64, "pos": "UB"}],
        edges=[{"source": 100, "target": 1}, {"source": 1, "target": 101},
               {"source": 101, "target": 2}, {"source": 2, "target": 102}],
    )
    return g, Problem(g, name="chain3")


def test_05a_missing_op():
    g, problem = _chain_problem()
    plan = {"node_to_subgraph": {100: 0, 101: 0}, "core_schedules": [[0]]}
    report = validate_plan(problem, plan)
    assert not report.ok and any("漏掉" in e for e in report.errors), report.errors


def test_05b_copy_op_included():
    """把 COPY_IN 写进 node_to_subgraph 必须被拒绝。"""
    g, _ = _chain_problem()
    g = graph(ops=g["ops"] + [{"id": 10, "op": "COPY_IN",
                               "pipe": "PIPE_MTE2", "cycles": 1}],
              tensors=g["tensors"], edges=g["edges"])
    problem = Problem(g, name="chain_with_copy")
    plan = {"node_to_subgraph": {100: 0, 101: 0, 102: 0, 10: 0},
            "core_schedules": [[0]]}
    report = validate_plan(problem, plan)
    assert not report.ok and any("COPY_IN/COPY_OUT" in e for e in report.errors), report.errors


def test_05c_duplicate_sgid():
    g, problem = _chain_problem()
    plan = {"node_to_subgraph": {100: 0, 101: 1, 102: 1},
            "core_schedules": [[0, 1, 1]]}
    report = validate_plan(problem, plan)
    assert not report.ok and any("重复出现" in e for e in report.errors), report.errors


def test_05d_subgraph_cycle():
    """A、C 同块而 B 单独一块 → 子图 DAG 成环。"""
    g, problem = _chain_problem()
    plan = {"node_to_subgraph": {100: 0, 101: 1, 102: 0},
            "core_schedules": [[0], [1]]}
    report = validate_plan(problem, plan)
    assert not report.ok, report.errors
    assert any("环" in e for e in report.errors), report.errors


def test_05e_bad_core_order():
    g, problem = _chain_problem()
    plan = {"node_to_subgraph": {100: 0, 101: 1, 102: 1},
            "core_schedules": [[1, 0], []]}
    report = validate_plan(problem, plan)
    assert not report.ok and any("顺序违反依赖" in e for e in report.errors), report.errors


def test_05f_negative_and_bad_shape():
    g, problem = _chain_problem()
    assert not validate_plan(problem, {
        "node_to_subgraph": {100: -1, 101: 0, 102: 0},
        "core_schedules": [[-1, 0]]}).ok
    assert not validate_plan(problem, {
        "node_to_subgraph": {100: 0, 101: 0, 102: 0},
        "core_schedules": []}).ok
    assert not validate_plan(problem, {
        "node_to_subgraph": {100: 0, 101: 0, 102: 0},
        "core_schedules": [[0]], "extra": 1}).ok


def test_05g_scheduler_timing_rules():
    """100/1000 是取 max 的时序约束，不是"依赖边数 × 常数"。"""
    from src.q1.schedule import estimate_start

    # 核为空：不加同核切换项
    assert estimate_start(0, 0, [None], {}, {}, {0: set()}) == 0
    # 核已有前一个 Task：+100
    assert estimate_start(0, 0, [500], {}, {}, {0: set()}) == 600
    # 跨核前驱：+1000
    assert estimate_start(0, 0, [None], {7: 1}, {7: 500}, {0: {7}}) == 1500
    # 同核前驱：不额外加 1000（由核内顺序表达）
    assert estimate_start(0, 0, [None], {7: 0}, {7: 500}, {0: {7}}) == 0
    # 多个跨核前驱取 max，不相加
    assert estimate_start(0, 0, [None], {7: 1, 8: 2}, {7: 500, 8: 900},
                          {0: {7, 8}}) == 1900
    # 同核 +100 与跨核 +1000 同时存在时也取 max
    assert estimate_start(0, 0, [300], {7: 1}, {7: 400}, {0: {7}}) == 1400

    # 端到端：同核串行
    g = graph(
        ops=[pipe_op(100, 10), pipe_op(101, 20), pipe_op(102, 30)],
        tensors=[{"id": 1, "size": 64, "pos": "UB"},
                 {"id": 2, "size": 64, "pos": "UB"}],
        edges=[{"source": 100, "target": 1}, {"source": 1, "target": 101},
               {"source": 101, "target": 2}, {"source": 2, "target": 102}],
    )
    problem = Problem(g, name="timing")
    part = partition(problem, 1)
    assert part.cost == {0: 60}

    from src.q1.partition import Partition
    one = Partition(node_to_sg={100: 0, 101: 1, 102: 2},
                    blocks=[[100], [101], [102]],
                    sg_preds={0: set(), 1: {0}, 2: {1}},
                    sg_succs={0: {1}, 1: {2}, 2: set()},
                    cost={0: 10, 1: 20, 2: 30},
                    topo_order=[0, 1, 2])
    sched1 = list_schedule(one, 1)
    assert sched1.core_orders[0] == [0, 1, 2]
    assert sched1.finish == {0: 10, 1: 130, 2: 260}
    # 3 核时全部落在核 0（核内只用 +100，比跨核的 +1000 便宜）
    sched3 = list_schedule(one, 3)
    assert sched3.core_orders[0] == [0, 1, 2]
    assert sched3.finish[0] == 10 and sched3.finish[2] == 260


def test_05h_granularity_candidates():
    assert candidate_granularities(4, 10000) == [4, 8, 16, 32]
    assert candidate_granularities(4, 6) == [4, 6]
    assert candidate_granularities(2, 3) == [2, 3]


# ---------------------------------------------------------------- TEST 6
def test_06_real_case_official_evaluator():
    """真实小 case 走官方 problem_1 evaluator，必须成功。"""
    from src.q1.evaluator import evaluate_plan

    path = case_path("case_019")
    problem = Problem.from_file(path)
    assert len(problem.nc_ids) > 0

    for num_cores in (2, 4):
        part = partition(problem, min(num_cores * 2, len(problem.nc_ids)))
        sched = list_schedule(part, num_cores)
        plan = build_plan(part, sched, num_cores)
        report = validate_plan(problem, plan)
        assert report.ok, report.errors
        result = evaluate_plan(problem.graph, plan, graph_name=problem.name,
                               graph_path=path, timeout=300.0)
        print(f"  [TEST 6] {problem.name} K={num_cores} "
              f"G={part.num_subgraphs} makespan={result.makespan} "
              f"added_copy={result.added_copy_bytes} "
              f"reason={result.failure_reason}")
        assert result.success, (result.failure_reason, result.stderr_tail)
        assert isinstance(result.makespan, int) and result.makespan > 0
        assert isinstance(result.added_copy_bytes, int)


def test_07_tensor_index_and_cost():
    """Tensor 索引保留 size/pos/producer/consumer；代价是 max(M,V,CP)。"""
    g = graph(
        ops=[pipe_op(100, 10, "MATMUL", "PIPE_M"),
             pipe_op(101, 20, "RELU", "PIPE_V"),
             pipe_op(102, 30, "RELU", "PIPE_V")],
        tensors=[{"id": 1, "size": 256, "pos": "L1"},
                 {"id": 2, "size": 256, "pos": "UB"}],
        edges=[{"source": 100, "target": 1}, {"source": 1, "target": 101},
               {"source": 101, "target": 2}, {"source": 2, "target": 102}],
    )
    problem = Problem(g, name="mixed")
    index = build_tensor_index(g)
    assert index[1].size == 256 and index[1].pos == "L1"
    assert index[1].producers == {100} and index[1].consumers == {101}
    assert problem.pipe_work([100, 101, 102]) == {"PIPE_M": 10, "PIPE_V": 50}
    # M=10, V=50, CP=60 → 60
    assert compute_cost(problem, [100, 101, 102]) == 60
    assert compute_cost(problem, [101, 102]) == 50
    # 100 -> t1 -> 101 -> t2 -> 100 构成真实环，必须立即报错而不是静默修复
    bad = graph(ops=[pipe_op(100, 1), pipe_op(101, 1)],
                tensors=[{"id": 1, "size": 8, "pos": "UB"},
                         {"id": 2, "size": 8, "pos": "UB"}],
                edges=[{"source": 100, "target": 1}, {"source": 1, "target": 101},
                       {"source": 101, "target": 2}, {"source": 2, "target": 100}])
    try:
        Problem(bad, name="cyclic")
    except Exception as error:  # noqa: BLE001 - 兼容官方与本地校验错误
        assert "环" in str(error) or "cycle" in str(error).lower(), error
    else:
        raise AssertionError("含环图必须报错")


def test_08_determinism_repeatable():
    """同一输入重复运行结果一致（拓扑序、分块、调度、计划 JSON）。"""
    path = case_path("case_019")
    problem = Problem.from_file(path)
    part_a = partition(problem, 8)
    part_b = partition(Problem.from_file(path), 8)
    assert part_a.node_to_sg == part_b.node_to_sg
    assert part_a.blocks == part_b.blocks
    sched_a = list_schedule(part_a, 4)
    sched_b = list_schedule(part_b, 4)
    assert sched_a.core_orders == sched_b.core_orders
    assert build_plan(part_a, sched_a, 4) == build_plan(part_b, sched_b, 4)


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001
            failures.append((name, error))
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
