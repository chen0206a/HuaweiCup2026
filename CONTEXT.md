# Current Context

## Competition
题目：A — 通用神经网络处理器下的多核调度问题（用户目前已选定，D004）。
当前日期：2026-09-23
阶段：方案评审与实施设计，正式求解器尚未实现。

## Current Main Direction
候选主线：Tensor通信计数、安全多层分块/细化、含搬运的代理、关键路径列表调度与预算约束自适应局部搜索。GNN及真正ALNS不列为第一版依赖。
队伍条件：4050 级显卡，代码主要借助 GPT 编写。
评审记录：`notes/a_plan_review_20260923.md`；Q1实现规格：`q1/DESIGN_v1.md`（候选，未实现/实测）。
六题比较作为历史保留：`notes/topic_selection_20260923.md`。

## Question Status
Q1：完成候选算法设计，待评价入口和基线。
Q2：确认按核心边界计通信，子图顺序仍影响核内调度，待实现。
Q3：确认按COPY_IN发射/完成事件分析FIFO，待实现。
Q4：A题无此问，保留初始化目录。

## Confirmed Results and Validation
没有正式优化实验或多核加速比。已完成一个人工分叉图的评估语义检查、三例单核计时探测，见 `notes/a_review_checks/`；这些不是论文性能结果。
最终性能须使用未修改的题目评估器与固定config。拟议评价设计在方案文档内，全局validation_protocol尚未修改或确认。

## Current Biggest Risks
- 逐边累计通信会误计共享张量；A与B对跨边界扇出写回的计数不同。
- 基础DAG合法不保证核内Pipe/内存依赖与跨核COPY组合图无环。
- 最大图在20秒单核探测上限内未完成，需实测评价成本后制定预算。

## Current Immediate Tasks
1. 建立官方评价封装、异常记录、有效保底方案和单核基准。
2. 实现Q1等工作量分块基线，再加入张量通信计数及有限邻域搜索。
3. 实测代表规模后确认实验预算和验证设计，再开展正式实验。

## Last Updated
2026-09-23：Codex，A题方案评审。
