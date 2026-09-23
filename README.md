# 华为杯中国研究生数学建模竞赛工作区

用于 2026 年竞赛的轻量协作、实验追踪和论文整理。赛题、模型和实验结论均留空待真实材料填入。

## 仓库结构

- `problem/`：赛题和附件清单
- `shared/`：假设、符号、数据字典、验证协议和方法笔记
- `q1/`–`q4/`：小问说明与交接
- `src/`、`scripts/`：建模代码与轻量工具
- `experiments/`：正式实验配置、指标和简短记录
- `results/`、`figures/`、`tables/`：汇总产物
- `paper/`：XeLaTeX 论文模板及自动生成内容
- `archive/`：确需保留的旧材料

## 新账号接手

先读 `AGENTS.md`、`CONTEXT.md`、`STATUS.md`、`DECISIONS.md` 和对应的 `qX/HANDOFF.md`，然后只打开完成分配任务所需的文件。不要默认扫描整个仓库。

## 本地命令

从仓库根目录运行：

```bash
python scripts/check_project.py
python scripts/summarize_experiments.py
python scripts/export_latex_tables.py
```

论文需使用 XeLaTeX。若已安装 latexmk：

```bash
cd paper
latexmk -xelatex main.tex
```

Python 汇总器使用标准库；读取 YAML 配置时优先使用 PyYAML（见 `requirements.txt`），未安装时会尽量提取简单标量字段。

## Git 协作

开始前检查 `git status`，适用时安全地获取远程更新。完成有意义的阶段成果后检查 `git diff`，只提交相关文件，并使用清晰的提交说明。不要覆盖其他小问已确认的内容。`data/raw/`、密钥、本地环境和编译缓存默认不上传。

## 比赛当天启动流程

1. `git pull`
2. 放入赛题和数据
3. 更新 `problem/problem.md`
4. 更新 `CONTEXT.md`
5. 三个账号分别独立阅读完整赛题
6. 总控确认各小问依赖关系
7. 更新 `DECISIONS.md`
8. 更新 `shared/validation_protocol.md`
9. 给 Q1/Q2/Q3 分工
10. 开始并行实验

每次接手任务时：

```text
Read:
AGENTS.md
CONTEXT.md
STATUS.md
DECISIONS.md
对应 qX/HANDOFF.md

Then read only the files necessary for your assigned task.
Do not scan the whole repository unless there is a concrete reason.
```

## 默认账号职责

- **Account A — Overall / Modeling Lead：**全题理解、总体路线、validation protocol、Q1 或核心建模、全局决策和一致性检查。
- **Account B — Experiment Lead：**Python、基线、特征工程、批量实验、消融、鲁棒性和实验汇总。
- **Account C — Paper / Secondary Modeling Lead：**其他小问、LaTeX、图表、论文整合和全文检查。

职责可根据赛题和进度调整。
