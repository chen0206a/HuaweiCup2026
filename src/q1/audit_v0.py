"""Inspect V0 plan structure without running an evaluator.

Run from repository root: python -m src.q1.audit_v0
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import REPO_ROOT, case_path
from .problem import Problem


def components(problem: Problem):
    remaining = set(problem.nc_ids)
    found = []
    while remaining:
        root = remaining.pop()
        members = {root}
        stack = [root]
        while stack:
            op = stack.pop()
            for adjacent in problem.nc_preds[op] | problem.nc_succs[op]:
                if adjacent in remaining:
                    remaining.remove(adjacent)
                    members.add(adjacent)
                    stack.append(adjacent)
        found.append(members)
    return sorted(found, key=lambda c: -sum(problem.work[x] for x in c))


def main():
    for case, cores in (("case_093", (3, 4)), ("case_026", (2, 5)),
                        ("case_025", (2, 5)), ("case_014", (2, 5))):
        problem = Problem.from_file(case_path(case))
        comps = components(problem)
        work = [sum(problem.work[x] for x in c) for c in comps]
        print(case, "ops", len(problem.nc_ids), "components", len(comps),
              "top work", work[:8])
        for k in cores:
            path = REPO_ROOT / "results" / "q1_v0" / "plans" / f"{case}_k{k}_multicore_res.json"
            if not path.is_file():
                continue
            plan = json.loads(path.read_text(encoding="utf-8"))
            mapping = {int(op): sg for op, sg in plan["node_to_subgraph"].items()}
            core_of = {sg: core for core, blocks in enumerate(plan["core_schedules"])
                       for sg in blocks}
            core_work = [0] * k
            for op, sg in mapping.items():
                core_work[core_of[sg]] += problem.work[op]
            print("  K", k, "blocks/core", list(map(len, plan["core_schedules"])),
                  "work/core", core_work)


if __name__ == "__main__":
    main()
