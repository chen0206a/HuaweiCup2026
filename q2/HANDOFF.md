# Q2 Handoff

## Status
已完成方案评审，正式求解器和性能实验尚未开始。

## Problem Interpretation
场景B：每核一个Task，同核保留缓存；跨核连接按评估器插入COPY对并等待500周期。

## Current Candidate Method
以核心亲和和负载为中心，同时保留多子图对核内顺序的控制。
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
继承Q1数据结构和评价封装，分别验证候选的真实spill和执行图合法性。

## Last Updated
2026-09-23（A题方案评审）
