"""Baseline 连续分块与子图代价。

在一条合法拓扑序上取**不重叠连续区间**作为子图。所有跨块依赖都沿拓扑序
前进，因此子图 DAG 必然无环；代码里仍然显式 validate，不依赖该性质静默通过。
"""

from __future__ import annotations

from dataclasses import dataclass

from .problem import GraphModelError, topological_order


@dataclass(frozen=True)
class Partition:
    """一次分块的完整结果。"""

    node_to_sg: dict          # op_id -> sgid
    blocks: list              # blocks[sgid] = 拓扑序连续的 op id 列表
    sg_preds: dict            # sgid -> set(sgid)
    sg_succs: dict            # sgid -> set(sgid)
    cost: dict                # sgid -> compute_cost
    topo_order: list          # 子图 DAG 的一个确定性拓扑序

    @property
    def num_subgraphs(self) -> int:
        return len(self.blocks)


def candidate_granularities(num_cores: int, num_ops: int) -> list:
    """返回去重、且不超过操作数的候选子图数量：K、2K、4K、8K。"""
    sizes = []
    for factor in (1, 2, 4, 8):
        size = min(num_cores * factor, num_ops)
        if size >= 1 and size not in sizes:
            sizes.append(size)
    return sizes


def contiguous_blocks(order: list, work: dict, num_blocks: int) -> list:
    """沿拓扑序做按计算工作量近似均分的连续分块。

    按累计工作量越过 target_i = total*(i+1)/G 切分，而不是按节点数量平均。
    每个 block 保持为拓扑序上的连续区间且非空。
    """
    n = len(order)
    if num_blocks <= 0:
        raise GraphModelError("num_blocks must be positive")
    if num_blocks > n:
        num_blocks = n
    total = sum(work[op] for op in order)

    blocks, current, cumulative = [], [], 0
    for index, op_id in enumerate(order):
        current.append(op_id)
        cumulative += work[op_id]
        blocks_left = num_blocks - len(blocks) - 1
        if blocks_left <= 0:
            continue
        ops_left_after = n - index - 1
        if ops_left_after < blocks_left:
            continue  # 再切下去后面的块会空
        target = total * (len(blocks) + 1) / num_blocks
        if cumulative >= target or ops_left_after == blocks_left:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def compute_cost(problem, members) -> int:
    """子图代理代价 max(M 工作量, V 工作量, 内部关键路径)。

    这只作为快速调度代理，**不代表官方真实执行时间**：本轮不模拟 DDR 争用、
    spill、L1/UB 精确驻留和四 Pipe 完整时间线，真实时间交给官方 evaluator。
    """
    members = list(members)
    if not members:
        return 0
    pipes = problem.pipe_work(members)
    return max(
        pipes.get("PIPE_M", 0),
        pipes.get("PIPE_V", 0),
        problem.internal_critical_path(members),
    )


def build_subgraph_dag(problem, node_to_sg: dict, num_subgraphs: int):
    """由原 Op DAG 投影出子图 DAG，并显式检查无环。"""
    sg_preds = {sg: set() for sg in range(num_subgraphs)}
    sg_succs = {sg: set() for sg in range(num_subgraphs)}
    for src in problem.nc_ids:
        src_sg = node_to_sg[src]
        for dst in problem.nc_succs[src]:
            dst_sg = node_to_sg[dst]
            if src_sg != dst_sg:
                sg_succs[src_sg].add(dst_sg)
                sg_preds[dst_sg].add(src_sg)

    topo = topological_order(
        sorted(sg_preds), sg_preds, sg_succs)  # 有环时在内部报错并定位
    if len(topo) != num_subgraphs:
        raise GraphModelError("子图 DAG 拓扑序不完整")
    return sg_preds, sg_succs, topo


def partition(problem, num_subgraphs: int) -> Partition:
    """在确定性拓扑序上做连续工作量分块，并组装子图 DAG 与代价。"""
    blocks = contiguous_blocks(problem.topo_order, problem.work, num_subgraphs)
    node_to_sg = {}
    for sgid, members in enumerate(blocks):
        for op_id in members:
            node_to_sg[op_id] = sgid

    if set(node_to_sg) != set(problem.nc_ids):
        raise GraphModelError("分块未覆盖全部非 COPY 操作")

    sg_preds, sg_succs, topo = build_subgraph_dag(
        problem, node_to_sg, len(blocks))

    position = {op_id: index for index, op_id in enumerate(problem.topo_order)}
    for sgid, members in enumerate(blocks):
        positions = [position[op_id] for op_id in members]
        if positions != sorted(positions) or positions[-1] - positions[0] != len(positions) - 1:
            raise GraphModelError(f"子图 {sgid} 不是拓扑序上的连续区间")

    cost = {sgid: compute_cost(problem, members)
            for sgid, members in enumerate(blocks)}

    return Partition(
        node_to_sg=node_to_sg,
        blocks=blocks,
        sg_preds=sg_preds,
        sg_succs=sg_succs,
        cost=cost,
        topo_order=topo,
    )
