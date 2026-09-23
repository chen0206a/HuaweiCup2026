"""Q1 V1-lite: one-step rollout core assignment with global completion forecast."""
from __future__ import annotations

import heapq

from .schedule import ScheduleResult, estimate_start, upstream_ranks


def _rank_topological_order(partition):
    rank = upstream_ranks(partition.cost, partition.sg_succs, partition.topo_order)
    pending = {s: len(partition.sg_preds[s]) for s in partition.cost}
    ready = [(-rank[s], s) for s in partition.cost if pending[s] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, s = heapq.heappop(ready)
        order.append(s)
        for t in partition.sg_succs[s]:
            pending[t] -= 1
            if pending[t] == 0:
                heapq.heappush(ready, (-rank[t], t))
    if len(order) != len(partition.cost):
        raise ValueError("subgraph DAG has a cycle")
    return order, rank


def _assign(s, core, part, state, same_wait, cross_wait):
    core_orders, core_of, start, finish, core_finish, core_load = state
    est = estimate_start(s, core, core_finish, core_of, finish, part.sg_preds,
                         same_wait, cross_wait)
    eft = est + part.cost[s]
    core_orders[core].append(s)
    core_of[s] = core
    start[s] = est
    finish[s] = eft
    core_finish[core] = eft
    core_load[core] += part.cost[s]
    return est, eft


def _copy_state(state):
    orders, core_of, start, finish, core_finish, core_load = state
    return ([x.copy() for x in orders], core_of.copy(), start.copy(),
            finish.copy(), core_finish.copy(), core_load.copy())


def _rollout(partition, order, state, same_wait, cross_wait):
    """Complete remaining blocks with the V0 greedy rule; returns global Cmax."""
    for s in order:
        best = None
        for core in range(len(state[0])):
            est = estimate_start(s, core, state[4], state[1], state[3],
                                 partition.sg_preds, same_wait, cross_wait)
            eft = est + partition.cost[s]
            key = (eft, state[5][core], core)
            if best is None or key < best[0]:
                best = (key, core)
        _assign(s, best[1], partition, state, same_wait, cross_wait)
    return max(state[3].values(), default=0)


def global_aware_schedule(partition, num_cores, same_core_wait=100,
                          cross_core_wait=1000):
    """Assign each block by predicted whole-plan makespan, not current EFT.

    A small parallelism preference opens an idle core only when the forecast
    differs from the best by at most 0.5%; it never forces round-robin.
    """
    if num_cores <= 0:
        raise ValueError("num_cores must be positive")
    order, rank = _rank_topological_order(partition)
    state = ([[] for _ in range(num_cores)], {}, {}, {},
             [None] * num_cores, [0] * num_cores)
    for i, s in enumerate(order):
        candidates = []
        for core in range(num_cores):
            trial = _copy_state(state)
            est, eft = _assign(s, core, partition, trial,
                               same_core_wait, cross_core_wait)
            forecast = _rollout(partition, order[i + 1:], trial,
                                same_core_wait, cross_core_wait)
            # The rank provides a remaining-path bound from this candidate.
            downstream_bound = est + rank[s]
            global_bound = max(forecast, downstream_bound)
            finish_load = [x or 0 for x in trial[4]]
            imbalance = max(finish_load) - min(finish_load)
            candidates.append((global_bound, forecast, imbalance, eft,
                               state[4][core] is None, core))
        best_bound = min(x[0] for x in candidates)
        near = [x for x in candidates if x[0] <= best_bound * 1.005]
        # Prefer a currently idle core only inside the narrow forecast band.
        selected = min(near, key=lambda x: (not x[4], x[0], x[2], x[3], x[5]))
        _assign(s, selected[5], partition, state,
                same_core_wait, cross_core_wait)
    orders, core_of, start, finish, _, _ = state
    return ScheduleResult(orders, core_of, start, finish,
                          max(finish.values(), default=0), order)
