# 图8 中文版说明

## 输入与复现

- 唯一数值来源：`E2026/outputs/q3/heaf_validation_metrics.json`，状态 `HEAF_VALIDATION_PASSED`。
- 协议和解释限制：`E2026/configs/final/q3_heaf_validation.yaml`、`E2026/outputs/q3/heaf_validation_report.md`。
- 绘图脚本：`E2026/outputs/final/q3/figures/scripts/figure8_q3_faithfulness_zh.py`。从 `E2026/` 运行 `python outputs/final/q3/figures/scripts/figure8_q3_faithfulness_zh.py`。
- 旧英文图保存于 `archive/figure8_q3_faithfulness_v1.{png,pdf,svg}`；原始指标和中间 CSV 未修改。

## 子图含义

- **（A）主导模态样本数统计：** Attachment2 全部 728 条 validation 样本的分类/回归主模态计数。文本 656/620、视觉 65/104、音频 7/4；实色与斜线分别表示分类与回归。
- **（B）删除实验曲线：** 396 条 audit 样本中，删除高贡献窗口或随机等长窗口后，冻结预测器的分类边际平均下降。横轴为 10%、20%、30%、40% 删除比例；阴影是按 126 个 `video_id` 分组、1,000 次 bootstrap 得到的各方法 95% 区间。
- **（C）平均边际差值：** 同一删除比例下“高贡献窗口 − 随机等长窗口”的平均分类边际下降差，阴影和误差线使用该配对差的分组 bootstrap 95% 区间。

## 相比旧版

图内文字改为中文；（A）与（B）互换编号，使主导模态统计先于删除曲线；保留三幅子图和全部原始数值。（A）使用固定方案 C：文本 `#4477AA`、视觉 `#EECC66`、音频 `#CC6677`；（B）蓝/玫红分别编码两种删除方法，（C）蓝色编码方法差值。两幅折线图使用 2.4 点线宽、4.5 点实心圆和 `alpha=0.18` 的真实 bootstrap 区间。背景、字体和辅助线与方案 C 版 Figure 9 保持协调。输出为 183 mm 宽、300 dpi PNG，以及 PDF/SVG 矢量图。

**建议图注：** 图8 基于 Attachment2 验证集的解释有效性与主导模态统计。（A）728 条样本中分类与回归任务的主导模态计数。（B）audit 子集上删除高贡献窗口和随机等长窗口后的分类边际平均下降；阴影表示按视频分组 bootstrap 的 95% 区间。（C）两种删除方式的平均边际下降差及其 95% 区间。高贡献窗口由候选窗口中最大影响值选出，因此其相对随机窗口的优势带有选择效应；本图反映模型扰动一致性，不构成现实因果或预测正确性的证明。
