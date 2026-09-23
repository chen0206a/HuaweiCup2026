# Q1 Handoff

## Status
已完成方案评审，正式求解器和性能实验尚未开始。

## Problem Interpretation
场景A：每子图一个Task，所有跨子图通信经DDR。

## Current Candidate Method
候选V1：多拓扑序、安全相邻块多层粗化/细化、张量通信计数、含搬运的代理、关键路径列表调度与预算约束自适应局部搜索。
源码评审见 `notes/a_plan_review_20260923.md`；具体数据结构、公式、接口及伪代码见 `q1/DESIGN_v1.md`。V1未实现或实测；真正的破坏—修复式ALNS及非连续修正是后续可选增强，GNN不作为依赖。

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
2026-09-23（合并方案与Q1实现规格）
