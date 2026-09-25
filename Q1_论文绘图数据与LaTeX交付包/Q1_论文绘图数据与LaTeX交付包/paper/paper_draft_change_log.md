# 全文初稿变更记录

## 2026-09-25 插图更新

- 将用户提供的 Q1 架构 PDF 替换正文总览图；在 Q1 最大时间重叠对齐方法之后新增模态时间对齐示意图、正文引用及图注。新图原 PPTX 的具体音视频帧率与本文实际抽取设置不一致，排版版将其改为示意文字，保留可编辑 PPTX 与 PDF；不将热图视为实验结果。
- 将用户提供的 Q2 架构 PDF 和缺失比例四指标 PNG 替换对应占位，并补充图注中的聚合方式、三次随机初始化样本标准差和 MAE 指标方向。Q2 三次初始化与误差分析图仍为占位。
- 重新运行 `build_paper.ps1`，全文为 20 页；无未定义引用、未定义文献、缺字或 overfull hbox。下方 19 页及旧图号记录属于更新前的历史构建。

## 本轮范围

- 以原 Q1 LaTeX 工程继续写作，未运行模型、未改动原始数据或实验结果。修改前的纸稿工程备份位于同级目录 `paper_source_backup_before_full_draft_20260924/`。
- 保留 Q1 方法、公式和四张数据表，仅清理实验管理措辞并补充辅助五折评价的适用边界。原 Q1 两张消融图移至附录 A1、A2；正文 Q1 图1、图2保持不变。
- 完成问题重述、假设、符号、Q2、Q3、综合评价及结论。Q2 正文图号为3—6，Q3为7—9。
- Q2 图3、4、6是标明 TODO 的占位；图5和 Q3 图7—9使用现有 PDF。题目、摘要、关键词仍为占位。
- `build_paper.ps1` 在本机缺少 Perl 时改用 XeLaTeX—BibTeX—XeLaTeX×2；原有表格生成与日志检查保留。

## 数值与边界依据

- Q1 数值来自 `../outputs/q1_final_local/00_manifest/q1_paper_numbers.json` 及 `../outputs/q1_final_local/02_paper_tables/`，构建脚本生成 `q1_numbers.tex` 和表格源文件。
- Q2 三次初始化性能和配对差来自 `D:/华为杯/E2026/outputs/metrics/b5_p2_multiseed_summary.json`；模型结构和训练配置依据 `D:/华为杯/E2026/src/`、`configs/b0.yaml` 与相应实验记录。所有预测性能均为附件2验证集结果。
- Q3 模态归因、解释干预和初始化稳定性来自 `D:/华为杯/E2026/experiments/q3/exp_001_heaf_validation/metrics.json`；文本证据映射依据 `D:/华为杯/E2026/outputs/q3/q3_text_row_identity_report.md`；附件4计数依据 `D:/华为杯/E2026/outputs/q3/final/attachment4_summary.json`。附件4无标签，正文没有用它计算预测性能。
- Q2 的综合鲁棒分数明示为项目内部验证指标。Q3 所选窗口对随机窗口的优势含选择效应；音频及视觉原始秒数或帧号未核验。上述限制在正文保留。

## 构建与版面检查

- `powershell -ExecutionPolicy Bypass -File build_paper.ps1` 成功生成 `build/main.pdf`；共19页，A4。
- XeLaTeX 日志未发现 undefined control sequence、未定义引用/文献、缺字或 overfull hbox。图号、表号和章节编号已核对。
- 已渲染全部19页并检查缩略总览，另检查 Q1/Q2 衔接、Q2 图表、Q3 方法/结果与案例页。现有图件显示正常。图9占独立浮动页，终稿排版时可再压缩留白。

## 未完成

见 `paper_todo.md`。本稿是内容完整、可编译的初稿，不是最终投稿版。
