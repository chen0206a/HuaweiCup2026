# Current Context

## Competition
题目：A — 通用神经网络处理器下的多核调度问题（用户目前已选定，D004）。
当前日期：2026-09-23
阶段：问题一 V1-P 快速验证完成，按用户要求收口；下一阶段等待 Q2 明确启动。

## Current Main Direction
当前 Q1 主线：Tensor 通信计数、结构感知连续分块、关键路径列表调度与官方 evaluator 等预算筛选；V1-P 已冻结。历史候选方案见 `q1/DESIGN_v1.md`，不在本阶段实施。
队伍条件：4050 级显卡，代码主要借助 GPT 编写。
评审记录：`notes/a_plan_review_20260923.md`；Q1 历史设计草案：`q1/DESIGN_v1.md`。
六题比较作为历史保留：`notes/topic_selection_20260923.md`。

## Question Status
Q1：Baseline V0 与 V1-lite 已实现。当前冻结 V1-P（结构感知分块）；12 例×K=2..5 的官方对照见 `q1/Q1_FREEZE_REPORT.md` 与 `results/q1_v1_validation/`。V1-S/组合版没有纳入本轮主实验，V1-P 未运行全 100 例。
Q2：确认按核心边界计通信，子图顺序仍影响核内调度，待实现。
Q3：确认按COPY_IN发射/完成事件分析FIFO，待实现。
Q4：A题无此问，保留初始化目录。

## Confirmed Results and Validation
Q1 Baseline V0 结果（全部来自未修改的官方 problem_1 evaluator，固定 config）：
`results/q1_v0/baseline_v0.csv` 6 例 × K=2..5 共 24 组，全部 success；单核分母见 `results/q1_v0/singlecore_baseline.csv`。
代表例加速比：case_036 最高 4.98×，case_093 2.96×，case_025 2.00×，case_074 1.03×，case_026 0.97×，case_014 0.97×（后两例低于单核）。
V0 代理低估官方真实时长，代理只作筛选、不作结论。V1-lite 中张量级预 spill 边界 COPY 计数已与 case_093、case_014 的官方结果核对一致。
V1-lite 的代表核数中，结构感知分块显著改善 case_093、case_026、case_025、case_014；单独全局感知分核暂未显示稳定收益。见 `q1/V1_LITE_AUDIT.md` 与 `q1/V1_LITE_RESULTS.md`。
V1-P 独立快速验证：12 个未用于 V1-lite 开发的 case，48 个 case×K 配对；官方 Makespan 平均改善 20.90%，中位改善 7.50%，87.50% 不退化，严重退化 3/48，涉及 2 个 case。最差 case_002 K=2 退化 30.58%。结论只覆盖限时抽样的 12 例；用户要求停止 Q1 优化，不进入全量研究级验证。
官方源码哈希已复核未改动（`src/q1/check_official.py`）。这些是真实基线，不是论文最终成绩。
更早的一个人工分叉图语义检查与三例单核计时探测见 `notes/a_review_checks/`。
全局 validation_protocol 尚未修改或确认。

## Current Biggest Risks
- 逐边累计通信会误计共享张量；A与B对跨边界扇出写回的计数不同。
- 基础DAG合法不保证核内Pipe/内存依赖与跨核COPY组合图无环。
- 最大图的四候选评价约需数百秒，全量实验需明确墙钟预算。

## Current Immediate Tasks
1. Q1 当前实现与验证报告已冻结；等待后续统一批量实验与论文阶段处理终局曲线。
2. 下一阶段优先 Q2 跑通并优化；需由用户另行启动。

## Last Updated
2026-09-23：Q1 V1-P 12 例快速验证及收口。
