# Project Status

## Topic Selection
用户目前选定A题，见DECISIONS.md D004。六题比较已完成并保留历史记录。

## Q1
状态：Baseline V0 已实现并跑通官方 evaluator；V1 未开始。
V0：`src/q1/`（非COPY Op DAG + Tensor索引、工作量连续分块、关键路径列表调度、独立 validate_plan、官方 evaluator 封装与缓存）。
单元测试：`tests/test_q1_v0.py`，15 项全部通过（含真实 case 的官方 evaluator smoke test）。
真实结果：`results/q1_v0/baseline_v0.csv`（6 例 × K=2..5，24 组全部成功）、`results/q1_v0/singlecore_baseline.csv`（官方单核分母）。
V0 结论：加速比 0.97×–4.98×，且随 K 非单调；多核收益不稳定。
候选V1：把核选择从"最小EFT"改为全局感知，并修正连续等工作量分块造成的伪屏障；再考虑安全多层粗化/细化、含搬运的代理和预算约束自适应局部搜索。
实现规格：`q1/DESIGN_v1.md`，已细化接口、公式、伪代码和文献适用边界，尚未实现。

## Q2
状态：完成场景B源码核查，未实现求解器。
候选方法：核心亲和分配与子图边界/顺序联合调整。
风险：真实spill和全局执行图合法性。

## Q3
状态：完成FIFO缓存事件规则核查，未实现求解器。
候选方法：从问题2候选出发，依据真实缓存事件调整合法顺序与分核。
风险：填充时刻、FIFO插入、共享缓存带宽及硬件/算法收益混淆。

## Q4
A题无第四问；现有目录保留，不作必答任务。

## Paper and Experiments
论文：初始化模板，尚无可写入的正式优化结果。
正式实验：暂无真实结果；原初始化示例不计为正式实验。
工程检查：一个人工图通信核验；三个用例的单核耗时探测，其中最大例达到20秒超时上限。
输出：`notes/a_plan_review_20260923.md`、`notes/a_review_checks/`。

## Last Update
2026-09-23：A题选题确认、外部方案评审、可执行路线设计。
