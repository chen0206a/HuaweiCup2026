# Q1 Handoff

## Status
已完成方案评审，正式求解器和性能实验尚未开始。

## Problem Interpretation
场景A：每子图一个Task，所有跨子图通信经DDR。

## Current Candidate Method
多拓扑序连续分块、张量通信计数、关键路径列表调度、边界移动/合并拆分/分核迁移。
方案详见 `notes/a_plan_review_20260923.md`；候选方法不等同于已验证性能。

## Inputs and Outputs
输入：原始计算图、固定config、核数。输出：node_to_subgraph与core_schedules，以及对应官方评估结果。

## Key Results
暂无正式优化结果。工程语义/计时检查见 `notes/a_review_checks/`，不得作为算法性能证据。

## Known Risks
通信不能按普通边权简单相加；基础DAG合法不保证最终执行图合法；大图真实评价成本需实测。

## Do Not Change Without Review
全局决定、验证协议，以及原始评估器和固定config。

## Next Recommended Actions
先建立官方评价封装和单核基准，再实现等工作量分块基线。

## Last Updated
2026-09-23（A题方案评审）
