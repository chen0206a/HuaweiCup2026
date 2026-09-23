# 工程语义与计时探测

本目录不属于正式优化实验，没有多核求解器性能结论。

- fanout_check.json：人工输入图、方案与原始结果。b=120的内部张量向两个远端目标发送；A新增360字节，B新增480字节。
- timing_probe.json：三个规模的官方单核一次性计时探测。20秒是本次子进程上限，不是题目限制。超时例没有Makespan结果。
- evaluator_hashes.json：原始评估代码与配置的SHA-256。

将原始A题附件解压到独立位置（该位置应有code和data目录），从仓库根运行：

```powershell
python notes/a_review_checks/reproduce_fanout.py --package-root <原始附件根目录>
python notes/a_review_checks/probe_worker.py <原始附件根目录>/data/case_019.json --package-root <原始附件根目录>
```

计时脚本统计评价函数耗时，不含读图；脚本本身不强制20秒超时，原始超时由外部子进程管理器实施。语义检查的等待参数对应所记录的固定config。
