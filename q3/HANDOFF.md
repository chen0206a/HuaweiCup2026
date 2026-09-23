# Q3 Handoff

## Status
已完成方案评审，正式求解器和性能实验尚未开始。

## Problem Interpretation
场景B加共享只读FIFO Cache，容量1048576字节、带宽250字节/周期。

## Current Candidate Method
在COPY_IN发射/完成事件层面分析命中、填充和淘汰，借助合法分核/排序改善。
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
先固定Q2方案比较有无L2，再优化Q3，并补做方案与硬件交叉对照。

## Last Updated
2026-09-23（A题方案评审）
