# Figure 9 数据审计：随机初始化稳定性与误差分析

## 范围与输出

图文件位于 `E2026/outputs/final/q2/figures/`。本次只新增图9的 PNG/PDF、绘图脚本与本说明，没有修改论文正文、表格或任何实验结果。检查当前 Q2 图目录时未发现已有 Figure 9 占位文件，因此按请求文件名生成了可供正文引用的图文件。

## 输入数据与来源

| 图面板 | 实际使用文件 | 数据含义 |
|---|---|---|
| (A) | `E2026/outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig6_paired_robust_score.csv` | B0-WCE 与 B5-P2 对应随机种子的验证集 robust score 及配对差 |
| (B) | `E2026/outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig6_clean_confusion_matrices.csv` | 三个种子、两种模型在 Attachment2 clean validation 上的真实类别×预测类别计数；图中取锁定主预测器 B5-P2 seed42 |
| (C) | `E2026/outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig6_clean_vision_all_zero.csv` | 原生 vision-all-zero 子集的逐种子 Accuracy、Macro-F1、MAE、Pearson；图中展示前三项 |

CSV SHA256：

- 配对鲁棒分数：`a1b05510ae6bec31178a6a137614059ad25bd255dec7c13f5d4c50ed52b870e6`
- 混淆矩阵计数：`b3c82158b5fca47cd30f0b77c641d85b7f4ddff9717177d53abf17cef095b13c`
- 原生视觉全零子集：`a96566aa984fcee08028a9042be89489e8e5c24b4685a2ee19e12728a18a9972`

上述绘图数据随 Q2 plotting handoff 提供，源结果快照包括 `source_results/b5_p2_multiseed_summary.json`、`source_results/b5_p2_multiseed_metrics.json` 与 `source_results/b0_weighted_ce_score_selection_metrics.json`。模型与 checkpoint 锁定记录为 `E2026/outputs/final/q2/q2_model_lock.md` 和 `q2_checkpoint_manifest.json`。

## 面板定义

### (A) 三次随机初始化配对结果

随机种子 42、43、44 分别配对连接 B0-WCE 与 B5-P2 的 robust score。逐种子 P2−B0 为 `+0.00559956`、`+0.00379148`、`−0.00008968`。三种子均值差为 `+0.00310045`，样本标准差（`ddof=1`）为 `0.00290689`。模型均值±样本标准差分别为 B0 `0.74167670 ± 0.00145806`、P2 `0.74477715 ± 0.00149212`。图内未使用置信区间、显著性标记或 p 值。

### (B) 验证集混淆矩阵

采用锁定主预测器 B5-P2 seed42 的 clean validation 预测，验证样本数 728。类别顺序为消极、中性、积极。计数矩阵（行是真实类别，列为预测类别）为：

```text
[[150, 27,  29],
 [ 34, 80,  70],
 [ 45, 51, 242]]
```

每行归一化后与计数一同标注；行支持数分别为 206、184、338。对角线准确率与锁定 clean 验证 Accuracy `0.64835165` 一致。此面板是单个锁定主预测器的诊断矩阵，不是三种子集成结果。

### (C) 原生视觉整段全零子集

子集样本数为 15。图中按同一随机种子并列连接 B0-WCE 与 B5-P2，只显示已有 Accuracy、Macro-F1 与 Pearson。实际数值如下：

| 种子 | 模型 | Accuracy | Macro-F1 | Pearson |
|---:|---|---:|---:|---:|
| 42 | B0-WCE | 0.533333 | 0.555556 | 0.337119 |
| 42 | B5-P2 | 0.466667 | 0.497280 | 0.243761 |
| 43 | B0-WCE | 0.533333 | 0.555556 | 0.369924 |
| 43 | B5-P2 | 0.466667 | 0.497280 | 0.400894 |
| 44 | B0-WCE | 0.466667 | 0.444444 | 0.290520 |
| 44 | B5-P2 | 0.466667 | 0.444444 | 0.291885 |

图中不呈现 MAE。该小子集只作为失效模式诊断，不外推为总体性能结论。

## 数据与实验边界

- 所有展示指标和计数均来自已保存的 Attachment2 validation 实验结果或其既有绘图汇总文件；没有重新计算模型预测。
- 使用了三个已锁定初始化 seed 42/43/44；A、C 中点线按相同 seed 配对。B 选用锁定的 P2 seed42 主预测器。
- 没有新训练、调参或 checkpoint 修改；没有读取 Attachment2 test，也没有使用 Attachment3/4。
- 图中的低饱和蓝色表示基线模型，橙色表示本文模型；混淆矩阵采用浅蓝序列色。

## 输出校验

- `fig09_q2_seed_pairing_error_analysis.png`：2109×1276 像素，PNG DPI 元数据约 300×300。
- `fig09_q2_seed_pairing_error_analysis.pdf`：单页矢量 PDF，版面约 178.6×108.0 mm。
- `plot_fig09_q2_seed_pairing_error_analysis.py` 可从以上既有 CSV 重新生成图，不加载模型或特征。
