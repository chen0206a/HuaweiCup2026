# Project Status

2026-09-24：当前 E 题工作目录为 `E2026/`。

- E Q2：B5-L1 完成 seed42 screening。原 B0-WCE 的 `lambda_reg=1.0`，四组初始权重、训练顺序、训练类权重一致；lambda1.0 精确复现历史参考。lambda2 robust +0.000808，未达到 +0.002 阈值；保留 lambda1.0 并停止该网格。
- 既有筛选：B5-P P1/P2 为正但尚未多种子确认；B5-F1 为负并停止；B4′后续暂停；重建路线停止。
- 本轮 outputs：`E2026/outputs/metrics/b5_l1_lambda_report.md` 与对应 JSON；正式实验 `experiments/exp_010_b5_l1_lambda/`。
- 未加载 attachment2 test 或 attachment3；不进入 Focal、label smoothing 或其他实验分支。
- Q1/Q3 当前完成度未核实；A题历史数据和文档仍保留。
