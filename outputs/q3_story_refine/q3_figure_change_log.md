# Q3 图件文字修订

所有改图均在本轮输出目录完成；原始图、脚本、PPTX、指标和解释数据保持原样。采用既有布局、配色、曲线及视频上下文画面，从源文件重新导出PDF，不在原PDF上覆盖文字。

| 正文图号 | 原图 | 绘图源 | 修改 |
|---|---|---|---|
| 图16 | `论文润色工作区/E2026_论文整理包/01_LaTeX论文工程/paper/figures/q3/fig14_q3_heaf_framework.pdf` | `论文整理/paper_revision_work/paper/figures/q3/fig14_q3_heaf_framework_source.pptx` | HEAF解释分析→预测解释分析；证据核验→输入来源回溯；来源标签改为原文字符级/未对齐特征行级；原核验状态改为实际输出“原文片段/特征行”；音频→语音。 |
| 图17 | 同目录 `fig15_q3_faithfulness_validation.pdf` | `E2026/outputs/final/q3/figures/scripts/figure8_q3_faithfulness_zh.py` | 分类边际→分类对数优势；高贡献窗口→高影响位置；随机等长窗口→随机位置，以对应删除比例曲线的实际删除集合；音频→语音；主模态→主导模态。 |
| 图18 | 同目录 `fig16_q3_case_explanations.pdf` | `E2026/outputs/final/q3/figures/scripts/figure9_q3_case_studies_v2.py` | 对数赔率→对数优势；时间遮挡→局部遮挡；已验证文本证据→原文字符级；负面/正面→消极/积极；音频→语音。保留样本02槽位1–6对应未对齐行36–41及上下文画面标注。 |

修改副本位于 `figure_sources/`，新图位于 `figures/q3/`。正文保留原LaTeX label以维持引用，仅显示名称和图注改写。

## 导出及数据核对

- 图16：`export_framework.ps1` 先执行Office预检，再通过PowerPoint原生导出。系统缺少query.exe，初次预检的会话状态为Unknown；使用Windows WTS接口查询真实会话状态后，预检确认Active、Default桌面、非sandbox用户且COM可用。结果保存为 `qa/office_preflight.json`。
- 流程图PPTX重新打开为1页；全部形状的位置与尺寸保持不变；文本中的全部数字及嵌入媒体逐项保持不变。删除核验词后填入实际来源输出，避免保留空白状态节点。
- 图17/18：复制既有Python脚本，仅调整标签及输出路径；从锁定指标JSON和解释JSONL读取原始值，不调用预测器。
- 比较修改前后渲染对象中的每条线、柱体、置信带/误差线集合及图片数组，逐项相等。数值快照为 `qa/fig15_q3_faithfulness_validation_plot_values.json` 和 `qa/fig16_q3_case_explanations_plot_values.json`。
- 图17的样本单位、均值、95%视频分组bootstrap区间均沿用原统计；图18仅展示样本14与02的既有贡献和连续曲线，没有新统计检验。
- 导出PDF矢量母版和300dpi PNG，另保留图17/18可编辑文字SVG。三图均完成图面及嵌入正文后的视觉检查。

## 文字修订QA

- 反包装与术语：三图均无HEAF、已核验/未核验、对数赔率、分类边际、关键时间等旧表述。
- 代码与数据：保留原配色、字体设置、图幅、坐标、误差范围和样本；本轮为文字修订，不重设绘图模板。
- 图面逻辑：主导模态、局部特征区间和来源回溯对应正文；视频画面持续标为上下文示意。
- 渲染：检查标题、轴标签、数据标签、图例及来源文字，无遮挡或裁切。案例曲线纵轴缩写为“对数优势下降”，类别目标由正文与面板标题说明。

本轮图中实验数值未变化，无待导出图件。
