# Experiments

正式实验目录格式：`exp_XXX_name/`，包含 `config.yaml`、`metrics.json`、`notes.md`，必要时包含 `figures/`。仅登记可复现且有明确问题的正式实验，临时调试放在被忽略的 `tmp/`。

根目录运行 `python scripts/summarize_experiments.py` 更新结果汇总。
