# Q1 Handoff

## Status
Baseline V0 已实现、测试、并用官方 evaluator 跑出真实基线。V1 尚未开始。

## Problem Interpretation
场景A：每子图一个Task，所有跨子图通信经DDR。
官方 `task_release_time` 的激活规则是若干时序约束取 max：
该核前一Task结束+100（核为空时不加），跨核前驱Task结束+1000。不是"边数×常数"。

## Current Method (V0, 已实现)
`src/q1/`：Tensor索引 + 非COPY Op DAG（沿COPY做可达闭包，与官方一致）→ 确定性Kahn拓扑序
→ 按 PIPE_M/PIPE_V cycles 做连续工作量分块（候选粒度 K/2K/4K/8K）→ 子图代价 max(M,V,内部CP)
→ 关键路径列表调度（按最小EFT选核）→ 独立 validate_plan → 官方 problem_1 evaluator。
入口：`src/q1/solver.py`（单例）、`src/q1/batch_run.py`（批量）。测试：`tests/test_q1_v0.py`。

## Inputs and Outputs
输入：原始计算图、固定config、核数。输出：`{node_to_subgraph, core_schedules}`（只含这两个键）
写到 `results/q1_v0/plans/<case>_k<K>_multicore_res.json`，以及官方评估结果 CSV。

## Key Results
`results/q1_v0/baseline_v0.csv`：6 例（2小2中1大 + 最大例）× K=2..5，24 组全部成功。
加速比（官方单核分母 `singlecore_baseline.csv`）：4.98× / 2.96× / 2.00× / 1.03× / 0.97× / 0.97×。
代理 makespan / 官方实测 = 1.00–1.26。

## Known Risks
1. 列表调度按最小EFT选核时，跨核前驱的1000 远贵于同核切换的100，
   导致块大量堆到同一核：case_074/014 上 8 个块全落 core 0，几乎无加速。
2. 连续等工作量分块是否切在分支边界上高度敏感：case_093 G=3 三块互相独立（2.96×），
   G=4 四块成链（1.00×）。加速比随 K 非单调，基线不可直接当"多核性能"结论。
3. 大图评价成本高：case_014 单次多核评价约 15–40s，单核约 327s。
4. 分块新增 COPY 可能很大（case_014 达 85 MB）却换不到收益。

## Do Not Change Without Review
全局决定、验证协议，以及原始评估器和固定config。官方评估器哈希见
`notes/a_review_checks/evaluator_hashes.json`，可用 `python src/q1/check_official.py` 复核。

## Next Recommended Actions
V1 优先修两件事，其余不动：
1. 核选择：不要只用即时最小 EFT；先做"整核不空闲"的分配，或按关键路径/负载全局比较，
   消除跨核1000 与同核100 造成的单核塌缩。
2. 分块：连续等工作量分块会切进分支内部制造伪屏障；改为按分支/共享输入结构切，
   或引入 DESIGN_v1 的安全相邻合并——但必须先证明真实收益。

## Last Updated
2026-09-23（Baseline V0 实现、单元测试与官方 evaluator 基线结果）
