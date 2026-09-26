# 图10–12去重导出

数据未变化。全部读取既有绘图CSV/已保存完整输入参考值，不重新计算实验指标。

| 正文图号 | 原图文件 | 原绘图源 | 本轮处理 |
|---|---|---|---|
| 图10 | fig11_q2_modality_missing_degradation.pdf | 论文整理/paper_revision_work/revision_sources/04_Q2_NEW_FIGURE_DATA/q2_missing_pattern_public_baseline_bundle/package_q2_missing_pattern_results.py | 删除“不同模态缺失比例下的性能退化”总标题，保留图例、四子图、阴影和虚线，tight裁边 |
| 图11 | fig12_q2_modality_location_absolute_degradation.pdf | 论文整理/paper_revision_work/paper/scripts/build_revision_figures.py | 删除底部正负含义、固定比例、三种子配对均值及MAE方向两行；规则移入caption，tight裁边 |
| 图12 | fig13_q2_p2_minus_b0_modality_location_gain.pdf | 论文整理/paper_revision_work/paper/scripts/plot_fig13_q2_modality_location_gain.py | 删除底部正值含义/三种子均值差及指标差值定义两行；规则移入caption，tight裁边 |

指定的论文润色工作区/04_数据与绘图源中找到历史绘图源；其中热图方向和当前版本不同，因此采用上述与当前PDF相匹配的后续绘图源。没有执行其统计聚合或评价部分，仅保留绘图函数并读取已保存数值。

本轮可复现导出脚本位于 figure_sources/export_figure10.py、export_figure11.py、export_figure12.py；绘图数值副本位于 figure_data/。颜色、曲线、阴影、各面板数值、坐标范围和色标均沿用原源；只是删除重复文字并裁边，图件保持矢量PDF，另保留300dpi PNG及SVG。

原图来自 outputs/paper_frontmatter_merged/figures/q2/；新图位于本目录 figures/q2/。新旧PDF中保留的全部小数数值逐项核对相同。
