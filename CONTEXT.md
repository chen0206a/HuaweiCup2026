# Current Context

日期：2026-09-24。用户当前明确选定 E 题（D005）。E 代码、结果与实验位于 `E2026/`。

## Current progress

Q2 已完成数据审计、B0/B1、连续缺失基准、多种子稳定性、重建路线验证、B4′单种子诊断、B5-P pooling、B5-F1 fusion、B5-L1 loss balance screening、B5-H0 只读 head consistency diagnostic，以及 B5-P2 seed42/43/44 复现。
B0-WCE 仍是参考。B5-P seed42 P1/P2 robust 分别 0.744320/0.745847；P2 paired robust deltas 为 seed42 +0.005600、seed43 +0.003791、seed44 −0.000090，均值 +0.003100 ± 0.002907（sample SD），2/3 为正，存在 seed 敏感性。仅保留为候选，不宣称稳定三 seed 增益。B5-F1 robust 0.738850，停止。B5-L1 保留 lambda_reg=1.0，lambda2 仅+0.000808，不细扫。
B5-H0 基于历史 B0-WCE seed42，在 clean + 固定54场景、仅validation上分析两头。tau=0 agreement clean/missing 为82.69%/81.62%；both-wrong约30.3%，reg-only-correct约5.1%/5.5%，oracle union上限约69.6%/69.7%。类别判断为部分互补但共享错误明显；Neutral真值回归轴分布较宽，符号映射无法恢复Neutral。详细结果和逐样本输出见 `E2026/outputs/metrics/b5_h0_head_diagnostic.md`、对应JSON与预测CSV/JSONL；正式记录 `experiments/exp_011_b5_h0_head_diagnostic/`。
Q1/Q3 的 E 题完成情况未核实；不要将A题历史成果混入。

## Immediate next steps

B5-P2 seed43/44 及三 seed 汇总完成，详见 `E2026/outputs/metrics/b5_p2_multiseed_report.md` 和 `experiments/exp_012_b5_p2_multiseed/`。当前没有自动启动下一阶段的授权；不要自动做 z-score 或模块组合。attachment2 test未索引/评估，attachment3未访问。早期 `b0_metrics.json` 有一次test评估，因此不要声称全项目test从未读取。冻结54场景协议未变。

## Historical A work

A题快照保存于 `archive/A_CONTEXT_before_E_20260923.md` 和 `archive/A_STATUS_before_E_20260923.md`。根目录 `q1/`、`q2/`、`q3/` 属于A的历史交接，不可混为E。
