# Figure 5 热图说明

## 输入

- `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig5_modality_by_location_per_seed_complete.csv`（SHA256 `73f7fcf2537d54a10fc7a83cbd60a682644c5f597cc5fa9a3137e21472a98f6e`）
- `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig5_modality_by_location_mean_sd.csv`（SHA256 `f5f0b651d5c215d7ae20c418da360ce5b3cf23c70e7d1bfdd4754f298b58a8db`）
- `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig5_double_modality_location_per_seed.csv`（SHA256 `7e802aa327dfe0587956791f1d7c3f0514ec49f89ef58d04c62ae556f1fd339f`）
- `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig5_double_modality_location_mean_sd.csv`（SHA256 `cdfb444b985f8c3ea3611f0782f6ca184d1ca2cdd2b6fe5ae1408974831fad17`）

逐种子表和均值／样本标准差表已交叉复核：每组为 2 模型 × 3 种子 × 3 模态组合 × 3 位置；单模态单元格在每个种子内平均 5 个预设缺失比例场景，双模态单元格在每个种子内对应 1 个场景。图中不把场景当作独立重复。

## 数值与色标

- 宏平均 F1、皮尔逊相关、准确率：`本文模型三种子均值 − 基线模型三种子均值`。
- 绝对误差改善：`基线模型 MAE 三种子均值 − 本文模型 MAE 三种子均值`。
- 全部 8 个热图共用以零为中心的色标 `[-0.012, +0.012]`；冷色为负、暖色为正。每格显示带符号的三位小数；显示为 `±0.000` 时实际绝对值可能小于 `0.0005`。
- 图中包含准确率面板，以呈现分类指标之间的取舍。未以颜色或数值标记统计显著性。

按 18 个模态×位置组合计，正／负单元格数：宏平均 F1 17/1，皮尔逊相关 18/0，绝对误差改善 16/2，准确率 6/12。这些是验证集三种子均值的方向统计，不代表每种子或每场景均改善。

## 图注草稿

图5 不同缺失位置与模态组合下本文模型相对基线模型的性能变化。上排为单模态缺失，下排为双模态缺失；色块表示附件2验证集三随机种子均值之差，绝对误差按“基线−本文”计算，因此正值均表示本文模型更优。收益依赖指标及模态／位置组合：宏平均 F1 和相关性多数为正，但准确率在若干组合下降；总体改善幅度较小且存在种子敏感性，不表示所有缺失场景稳定提升。
