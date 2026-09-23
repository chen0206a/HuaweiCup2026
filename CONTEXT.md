# Current Context

## Competition
题目：A — 通用神经网络处理器下的多核调度问题（用户目前已选定，D004）。
当前日期：2026-09-23
阶段：问题一 V1-lite 实现与代表用例消融。

## Current Main Direction
候选主线：Tensor通信计数、安全多层分块/细化、含搬运的代理、关键路径列表调度与预算约束自适应局部搜索。GNN及真正ALNS不列为第一版依赖。
队伍条件：4050 级显卡，代码主要借助 GPT 编写。
评审记录：`notes/a_plan_review_20260923.md`；Q1实现规格：`q1/DESIGN_v1.md`（候选，未实现/实测）。
六题比较作为历史保留：`notes/topic_selection_20260923.md`。

## Question Status
Q1：Baseline V0 与 V1-lite 已实现。V1-lite 仅涉及结构感知分块与全局感知分核；代表用例官方结果见 `q1/V1_LITE_RESULTS.md`，未完成全 100 例。
Q2：确认按核心边界计通信，子图顺序仍影响核内调度，待实现。
Q3：确认按COPY_IN发射/完成事件分析FIFO，待实现。
Q4：A题无此问，保留初始化目录。

## Confirmed Results and Validation
Q1 Baseline V0 结果（全部来自未修改的官方 problem_1 evaluator，固定 config）：
`results/q1_v0/baseline_v0.csv` 6 例 × K=2..5 共 24 组，全部 success；单核分母见 `results/q1_v0/singlecore_baseline.csv`。
代表例加速比：case_036 最高 4.98×，case_093 2.96×，case_025 2.00×，case_074 1.03×，case_026 0.97×，case_014 0.97×（后两例低于单核）。
V0 代理低估官方真实时长，代理只作筛选、不作结论。V1-lite 中张量级预 spill 边界 COPY 计数已与 case_093、case_014 的官方结果核对一致。
V1-lite 的代表核数中，结构感知分块显著改善 case_093、case_026、case_025、case_014；单独全局感知分核暂未显示稳定收益。见 `q1/V1_LITE_AUDIT.md` 与 `q1/V1_LITE_RESULTS.md`。
官方源码哈希已复核未改动（`src/q1/check_official.py`）。这些是真实基线，不是论文最终成绩。
更早的一个人工分叉图语义检查与三例单核计时探测见 `notes/a_review_checks/`。
全局 validation_protocol 尚未修改或确认。

## Current Biggest Risks
- 逐边累计通信会误计共享张量；A与B对跨边界扇出写回的计数不同。
- 基础DAG合法不保证核内Pipe/内存依赖与跨核COPY组合图无环。
- 最大图的四候选评价约需数百秒，全量实验需明确墙钟预算。

## Current Immediate Tasks
1. 补足 V1-lite 代表用例其余核数，核对评估调用数与运行时间。
2. 判断是否保留无稳定收益的 V1-S，再固定参数运行 100 例。
3. 完成论文要求的 1–5 核结果与逐用例附录；问题 2、3 另行实现。

## Last Updated
2026-09-23：Codex，A题方案评审。
