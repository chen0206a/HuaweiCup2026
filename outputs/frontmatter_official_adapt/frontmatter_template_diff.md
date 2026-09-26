# 前置部分官方版式适配

## 版本与边界
基于 `outputs/layout_refine_round1/paper_layout_refine_round1.tex` 制作独立副本，保留 ctexart；没有改动原论文工程。Q1/Q2/Q3正文、生成表格、图件及冻结摘要文件逐文件字节比较一致，详见 source_preservation_audit.json。全文45个物理页：封面1页、摘要2页、目录2页，随后正文、参考文献、附录连续编号至44。

## 逐项对照
| 项目 | 当前版本 | 参考及修改后效果 | 与官方附件3的对应 |
|---|---|---|---|
| 封面 | 已填写的官方Word导出封面 | 参考Suwren：直接插入原PDF第一页；武汉轻工大学、26104960057、路晨/宋文杰/徐丽；不重绘赛事标题和四个Logo | 使用本地官方Word已导出页，A4、无页码；仅填写身份信息 |
| 摘要页头 | 仅论文标题及摘要标题 | 从本地official_reference/template.pdf第二页截取三行固定赛事字形，作为矢量背景；标题与冻结摘要、关键词不变 | 页头原字形及页面坐标保持；不包含模板的空白题目横线 |
| 目录 | 自动目录，默认缩进与点引线 | 参考Suwren、rudykon三级目录：统一小四宋体，逐级缩进，点引线、页码右对齐；保留原来Q3前的目录分页 | 官方未提供目录逐级尺寸；不照搬第三方五号字号，以附件2小四宋体优先 |
| 页码 | 摘要起1、居中页脚 | 延续Suwren连续阿拉伯页码方式；摘要1–2、目录3–4，正文从5开始 | 封面无页码；其余居中连续，不增加页眉 |
| 页边距 | 正文上2.4/下2.2/左右2.25cm | 摘要及目录上3/下1.75/左右2.25cm，对照官方Word节属性；正文restoregeometry恢复原布局 | 官方Word约上85.1、下49.65、左63.8、右63.7pt；正文保留当前实现，避免整体重排 |
| 参考文献后台 | 手工thebibliography | Suwren settings/gmcm.bst + BibTeX、数字引用；现有references.bib16篇，按首次引用排序，无新增文献 | 采用请求允许的gmcm顺序编码后台；该bst为第三方适配，不能将其等同官方认证或完整GB/T7714实现 |
| 附录编号 | 手工附录A/表A1 | 借鉴Guangchen自动字母编号逻辑，使用标准appendix与ctex编号；附录A自动生成、表A1保留原100条 | 附录与表格内容不变；附录B仅注释预留，不显示空标题 |

## 预留结构
第1、2、6、7节和附录B仅保留注释输入位置，没有生成占位正文或目录条目。目录从第3节开始。参考文献位于正文之后、附录之前。

## 检查结果
- 最终XeLaTeX/BibTeX编译成功，45页；无overfull，无未定义交叉引用或文献引用。
- 已检查封面、摘要、两页目录、参考文献与附录首页渲染。附录A表A1数据文件逐字节不变。
- 宋体、黑体、Times New Roman继续调用系统字体；未下载或复制第三方字体。已有图内字形未改。
- Word COM预检提示桌面会话状态Unknown，因此本轮没有重新启动Word编辑；使用本地此前已完成的官方Word封面导出PDF，并核对页面真实身份信息。未声称本轮重新导出Word。
- 编译日志仍有系统宋体粗体回退提示（源文件原设置），不影响正文宋体要求；MiKTeX系统版本提示不影响编译结果。

## 参考项目
- https://github.com/Suwren/2026_23rdCPGMCM_LaTeX ：固定PDF封面/摘要头、目录与gmcm.bst。
- https://github.com/GuangchenJ/Huawei_Cup_2026 ：页面和附录实现参考，其旧gmcm.bst在当前BibTeX测试遇到条目处理问题，未采用。
- https://github.com/rudykon/GMCM2026-LaTeX-Template ：三级目录及附录辅助参考。

本地官方附件2格式规范和附件3Word模板为最高依据；第三方代码仅用于适配。参考文本代码保留在template_reference，未迁移任何第三方类文件到正文。

## 复编译
在本目录运行：xelatex paper_frontmatter_official_adapt.tex；bibtex paper_frontmatter_official_adapt；xelatex两次。所有输入文件随副本保留。
