# Project Status

2026-09-24：当前 E 题工作目录为 `E2026/`。

- E Q2：完成 B5-H0 只读诊断。历史 B0-WCE seed42 checkpoint 未修改；valid clean + 54场景，benchmark SHA256不变。
- Head一致性：clean/missing agreement 82.69%/81.62%；both-wrong约30.3%；reg-only-correct约5.1%/5.5%。Oracle union上限约69.6%/69.7%，仅诊断。结论：部分互补但共享错误明显，尤其Neutral不能由tau=0回归符号映射恢复。
- 输出：`E2026/outputs/metrics/b5_h0_head_diagnostic.{json,md}`、`b5_h0_predictions.{csv,jsonl}`；正式记录 `experiments/exp_011_b5_h0_head_diagnostic/`。
- 未训练模型、未修改checkpoint、未索引/评估attachment2 test、未访问attachment3；不自动进入H1。
- Q1/Q3完成情况本轮未核实；A题历史资料保留。
