# Q1证据更新版独立预览

主输出q1_rewrite_preview_v2.pdf（15页）；源码入口q1_rewrite_preview_v2.tex；正文sections/05_q1.tex。依赖generated/、figures/q1/、q1_numbers.tex及bibliography/。保持正式工程格式；不自动合并正文。

说明文件：q1_requirement_mapping_v2.md、q1_repro_audit_v2.md、q1_rewrite_change_log_v2.md。机器检查记录位于qa/。原v1保留不覆盖。

## 编译

在本目录运行XeLaTeX、BibTeX、XeLaTeX两次；不需要读取特征文件即可编译已核实正文：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build q1_rewrite_preview_v2.tex
bibtex build/q1_rewrite_preview_v2
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build q1_rewrite_preview_v2.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build q1_rewrite_preview_v2.tex
```

build/保存实际编译产物，顶层PDF为其副本。字体和格式依赖与正式前言一致。revise_and_build.py记录从v1复制和修订的过程；编译已交付源文件直接使用上述命令即可。

## 文件核对与来源

原包D:/java录屏/q1_completion_20260926.zip解压至evidence/completion/，未修改包内文件。verify_completion.py检查包清单、最终FP32/mask/ID hash、数组和来源对应；final_qa.py检查论文数据/引用/保护hash。核对脚本需要恢复包及v1工作区，不是模型运行脚本。

最终文件的三个hash与包内manifest一致，详见q1_repro_audit_v2.md。zip、1024文件hash清单、实际统计完整记录在qa/completion_verification.json。原始媒体和模型权重不随本预览复制；最终特征与完整来源文件在本地evidence/completion/供核查。

正文不写长hash或未记录依赖小版本。PyTorch2.14.0+cu130与Transformers4.57.3有记录，GPU Python/OS未记载，因此不使用建议中的Ubuntu24.04/Python3.12.3。本轮没有新实验、训练或特征生成。
