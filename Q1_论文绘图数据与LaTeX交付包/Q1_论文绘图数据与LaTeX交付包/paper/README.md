# LaTeX 论文工程

本工程使用 MiKTeX、XeLaTeX 与 BibTeX，主文件为 main.tex。页面尺寸、页边距、字体、标题、行距与页码设置依据 2026 年第 23 届竞赛格式规范及论文模板；题目、摘要与关键词仍是明确标注的占位内容。

## 冻结数据与表格

Q1 数量、模型和张量常量的唯一清单为：

../outputs/q1_final_local/00_manifest/q1_paper_numbers.json

构建脚本直接从该清单生成 q1_numbers.tex。四张正文表由 scripts/generate_frozen_tables.py 从 ../outputs/q1_final_local/02_paper_tables/ 的冻结 CSV 读取并排版；脚本只格式化原始字段，不重算任何实验指标。各 CSV 的 SHA-256 记录在 generated/q1_table_sources.json。

正文现包含 Q1、Q2、Q3、模型评价和结论。正文图按统一顺序编号：图1为Q1总览，图2—4为Q1三模态特征提取占位，图5为Q1时间对齐，图6—9为Q2，图10—12为Q3。Q1消融图与样本追踪图移至附录A1—A3。Q2三次初始化与误差分析图仍为占位。题目、摘要、关键词尚待最终定稿。未改写实验结果表格。

本轮新增的 Q1 模态对齐图保留可编辑 PPTX 与排版用 PDF。图内原先标注的音视频帧率与论文实际特征抽取设置不一致，排版版改为时间粒度示意，正文图注明确说明热图与采样间隔不代表实际特征值或采样率。

## 构建

在本目录运行 PowerShell 命令：powershell -ExecutionPolicy Bypass -File build_paper.ps1

需要 XeLaTeX、BibTeX 和 Python。若同时有 latexmk 与 Perl，脚本调用 latexmk；否则自动执行 XeLaTeX、BibTeX、XeLaTeX 两遍。最终 PDF 生成在 build/main.pdf，LaTeX 中间文件保存在 build/。稿件变更与尚待完成事项见 paper_draft_change_log.md 和 paper_todo.md。
