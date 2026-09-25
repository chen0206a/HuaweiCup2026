# Q2 剩余实验状态（2026-09-25 更新）

ATTACHMENT3_FINAL_INFERENCE = PASS

PUBLIC_BASELINE_TFN = COMPLETE_3_SEEDS

PUBLIC_BASELINE_MULT = COMPLETE_3_SEEDS

PUBLIC_BASELINE_MISA_OR_SELFMM = MISA_COMPLETE_3_SEEDS

BASELINE_3SEED_COMPLETE = YES

BASELINE_MISSING_BENCHMARK_COMPLETE = YES

上一轮仅完成本地 Attachment3 无标签推理。本轮另按用户指示在新服务器 RTX 3090 上部署 Attachment2 和适配代码，TFN/MulT/MISA 各训练 seed 42/43/44，并在冻结 clean + 54 缺失场景上评估。全部九次完成；汇总见 `outputs/final/q2/public_baselines/Baseline_Experiment_Report.md`。

附件3正式结果：`outputs/final/q2/attachment3/attachment3_predictions.csv`（30行）及审计、摘要、接口回归与交付检查文件。没有标签性能指标；Q2 checkpoint 不变。公开 baseline 未使用附件3。

本地推理开始时 Git HEAD：`0788d28abf1d584db82554351af1f343bbb30df6`；最终当前 commit 请以 `git rev-parse HEAD` 为准。仓库既有的未跟踪论文/绘图资产未改动。

本轮服务器 baseline 已结束；结果和 checkpoint 已回传本地，服务器继续占用并非复现所必需。
