# 公开 Baseline 缺失场景比较审计

## 数据来源与完整性

- TFN/MulT/MISA 逐场景结果来自 `E2026/experiments/q2/public_baselines/<model>/metrics_seed{42,43,44}.json`，已逐条核验，不从论文表格反推。
- 每模型 3 seed × 54 场景 = 162 条；总计 486 条唯一缺失场景记录。所有场景 ID 与冻结 benchmark 完全一致。
- 每个结果文件的 benchmark SHA256 与冻结值一致：`3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`；每个文件均标记 `smoke=false`，clean 行存在，指标数值有限。
- 九个公开 baseline checkpoint 的文件大小及 SHA256 均与 `baseline_checkpoint_manifest.json` 一致。本文模型 P2 的 54 场景值来自已经锁定的 Q2 完整场景表。
- 指标统计：每 seed 先对 54 场景作等权平均；之后按三个 seed 计算均值与样本 SD（ddof=1）。有效重复单位是 seed（n=3），不是 162 个 seed × scenario。
- 数据边界：Attachment2 train/valid 已有结果；没有使用 Attachment2 test、Attachment3 或 Attachment4；不重新训练或推理。

## 缺失场景比较（均值 ± 样本 SD）

| 方法 | Accuracy↑ | Macro-F1↑ | MAE↓ | Pearson↑ | R（辅助） |
|---|---:|---:|---:|---:|---:|
| TFN | 0.5767 ± 0.0197 | 0.5582 ± 0.0127 | 0.6451 ± 0.0101 | 0.5784 ± 0.0070 | 0.7139 ± 0.0039 |
| MulT | 0.6000 ± 0.0209 | 0.5749 ± 0.0230 | 0.6309 ± 0.0189 | 0.6232 ± 0.0057 | 0.7253 ± 0.0061 |
| MISA | 0.6079 ± 0.0095 | 0.5958 ± 0.0080 | 0.9247 ± 0.1025 | 0.6198 ± 0.0077 | 0.7168 ± 0.0019 |
| 本文模型（P2） | 0.6340 ± 0.0049 | 0.6163 ± 0.0012 | 0.6091 ± 0.0060 | 0.6414 ± 0.0025 | 0.7448 ± 0.0015 |

## 解释边界

公开 TFN/MulT/MISA checkpoint 按 clean validation 选择；已锁定的 P2 checkpoint 按 robust score 选择。由于选模准则不一致，本表只能称为**描述性缺失场景比较**，不用于声称本文模型严格优于公开模型或显著超过全部 baseline。公开 baseline 是基于对齐特征的架构适配实现，不能称作原论文公开数字的复现。

## 文件

- `q2_public_baseline_missing_scenarios_per_seed.csv`：486 条公开 baseline 原始场景结果。
- `q2_public_baseline_missing_per_seed.csv`：每个模型/seed 的 54 场景平均。
- `q2_public_baseline_missing_comparison.csv`：三 seed 均值/SD 与展示字段。
- `q2_public_baseline_missing_comparison.tex`：可人工审阅的 LaTeX 表格片段；未自动插入论文。
- `q2_all_models_clean_per_seed.csv`：已有 clean validation 逐 seed 对照。
