# Project Status

2026-09-24：当前 E 题工作目录为 `E2026/`。

- E Q2：已有可靠基线与缺失评测、多种子重建负结果；B4′后续暂停。B5-P pooling 残差 seed42 筛选为正但原生 vision 全零子集退化；B5-F1 低秩 fusion seed42 为负，按预设规则停止。
- E Q1/Q3：本轮未核实完成情况，不等同于未开展。
- 本轮完成：核验 B5-P 冻结 B0 的 Dropout 为 eval；F1 初始化等价、训练、冻结 benchmark、缓存一致性、两种 checkpoint 与 interaction 贡献诊断。
- 本轮未进行：B4′后续、B5-P 多种子、F1 seed43/44、组合、终局test评估、附件3预测或验证协议更改。
- 当前结果：`E2026/outputs/metrics/b5_fusion_report.md`；正式记录见 `E2026/experiments/exp_009_b5_fusion/`。
- A题历史状态：`archive/A_STATUS_before_E_20260923.md`；根目录qX交接仍属于A。
