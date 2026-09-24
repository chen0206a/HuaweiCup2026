# Current Context

日期：2026-09-24。用户当前明确选定 E 题（D005）。E 代码、结果与实验位于 `E2026/`。

## Current progress

Q2 已完成数据审计、B0/B1、连续缺失基准、三种子稳定性、重建路线验证、B4′单种子门控筛选、B5-P pooling、B5-F1 fusion、B5-L1 loss balance seed42 screening。
B0-WCE 主干仍为当前可靠参照。B5-P seed42 P1/P2 robust 分别为 0.744320/0.745847，高于 B0 的 0.740248，但需要多种子确认；B5-F1 robust 0.738850，fusion 线按预设规则停止。
B5-L1 lambda 1.0 从头重训精确复现历史 seed42 clean score 0.741331 / robust 0.740248。lambda .25/.5/1/2 的 robust 分数为 0.729305/0.737328/0.740248/0.741056，故 lambda2 相对 reference 仅 +0.000808，低于 +0.002 screening 门槛；其它候选未提升。保留 lambda_reg=1.0，不继续细扫，也不补多 seed。
结果：`E2026/outputs/metrics/b5_l1_lambda_report.md`、`b5_l1_lambda_metrics.json`；正式记录 `experiments/exp_010_b5_l1_lambda/`。
Q1/Q3 的 E 题完成情况本轮尚未核实，不能将 A 题旧成果混入。

## Immediate next steps

暂无用户授权的下一实验；不要进入 Focal、Label Smoothing、B4′后续或其它 B5 分支。附件2 test 与附件3在 B5-L1 未加载。早期 `b0_metrics.json` 有一次 test 评估，因此不要声称全项目 test 从未读取。冻结54场景基准未改。

## Historical A work

原A快照保存于 `archive/A_CONTEXT_before_E_20260923.md` 和 `archive/A_STATUS_before_E_20260923.md`。根目录 `q1/`、`q2/`、`q3/` 是 A 的历史交接，代码与结果保留；不要当成 E 的进度。
