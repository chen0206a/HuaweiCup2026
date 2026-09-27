# 当前排版版本

主文件：paper_layout_balance_optimized.tex

编译结果：paper_layout_balance_optimized.pdf（77页，附录29页）。旧版本未覆盖。

在本目录运行两遍：

```powershell
xelatex -interaction=nonstopmode -halt-on-error paper_layout_balance_optimized.tex
xelatex -interaction=nonstopmode -halt-on-error paper_layout_balance_optimized.tex
```

需要Windows系统宋体、黑体、Times New Roman、Consolas，以及XeLaTeX和源文件所列宏包。保持sections、generated、figures、frontmatter、appendices目录及根目录引用文件的相对位置。

排版调整见layout_balance_audit.md；内容保护比对见layout_content_integrity.json；AI声明变化见ai_declaration_change_log.md。所有实验、图表数据及摘要未修改。
