# 分模态缺失比例曲线审计

## 来源与样本单位

- 场景级输入：`E2026/outputs/final/q2/q2_plotting_handoff_v2/plot_data/scenario_details_complete.csv`；完整场景表 330 行。
- P2 单模态缺失行：135（seed × modality × rho × location = 3 × 3 × 5 × 3）。每行对应一次 Attachment2 valid 条件评估，不是一次训练。
- Benchmark SHA256：`3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`。位置列表为 early/middle/late。
- 每个 seed × 模态 × rho 内先对三个位置作等权平均；每条曲线点是三 seed 均值，带状区域为 seed 间样本标准差（ddof=1，n=3）。不把位置或场景当独立重复。
- 完整输入虚线为 Q2 锁定模型 clean valid 的三 seed 均值。

## 输出数据

- `q2_modality_missing_scenarios_p2.csv`：135 条位置级原始场景值。
- `q2_modality_missing_per_seed.csv`：每个 seed × 模态 × rho 的位置平均值。
- `q2_modality_missing_degradation_data.csv`：绘图均值、样本 SD 和 seed42/43/44 明细。
- 四个 panel 分别为 Accuracy、Macro-F1、MAE、Pearson；MAE 越低越好，其余越高越好。

## 端点变化（rho=0.5 减 rho=0.1）

| 指标 | 文本缺失 | 音频缺失 | 视觉缺失 |
|---|---:|---:|---:|
| accuracy | -0.0292 | -0.0002 | -0.0038 |
| macro_f1 | -0.0230 | -0.0002 | -0.0039 |
| mae | +0.0133 | +0.0004 | -0.0002 |
| pearson | -0.0289 | -0.0001 | -0.0010 |

## 图注草稿

图X 不同模态缺失比例下的性能退化。每个ρ下先对 early、middle、late 三个位置等权平均，再计算三次随机初始化的均值；阴影表示三 seed 间样本标准差。虚线表示完整输入验证集的三 seed 均值。结果为 Attachment2 validation 上的描述性比较，不应解释为现实因果效应。
