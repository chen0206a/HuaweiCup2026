# Project Status

## Topic Selection
用户目前选定A题，见DECISIONS.md D004。六题比较已完成并保留历史记录。

## Q1
状态：Baseline V0 已实现并跑通官方 evaluator；V1-lite 的结构感知分块和全局感知分核已实现，代表用例正在消融验证。
V0：`src/q1/`（非COPY Op DAG + Tensor索引、工作量连续分块、关键路径列表调度、独立 validate_plan、官方 evaluator 封装与缓存）。
单元测试：`tests/test_q1_v0.py`，15 项全部通过（含真实 case 的官方 evaluator smoke test）。
真实结果：`results/q1_v0/baseline_v0.csv`（6 例 × K=2..5，24 组全部成功）、`results/q1_v0/singlecore_baseline.csv`（官方单核分母）。
V0 结论：加速比 0.97×–4.98×，且随 K 非单调；多核收益不稳定。
V1-lite：`src/q1/partition_v1.py`、`schedule_v1.py`、`experiments_v1.py`，保留 V0 入口。新增 3 项测试通过，V0 原 15 项继续通过。初步官方结果见 `q1/V1_LITE_RESULTS.md`：分块模块在四个异常用例的代表核数上显著改善；新分核模块暂未显示稳定独立收益。
后续候选多层粗化/细化与局部搜索仍未实现，只有在 V1-lite 的更广泛验证后再讨论。

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
正式实验：V0 的 24 组与 V1-lite 代表用例消融均为真实官方评估；V1-lite 尚非全 100 例最终成绩。
工程检查：一个人工图通信核验；三个用例的单核耗时探测，其中最大例达到20秒超时上限。
输出：`notes/a_plan_review_20260923.md`、`notes/a_review_checks/`。

## Last Update
2026-09-23：A题选题确认、外部方案评审、可执行路线设计。
