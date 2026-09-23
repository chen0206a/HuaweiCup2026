# Q1

本目录用于问题 1 的说明和任务特有材料。开始工作前阅读根目录协作文件及 `HANDOFF.md`。

## V1-lite 复现

官方附件放在 Git 忽略的 `data/raw/A题附件/`，内含 `code/`、`data/`、`docs/`；也可设置 `HUAWEI_CUP_ATTACH` 指向已有附件目录。固定配置不修改。

从仓库根目录运行：

```powershell
python tests/test_q1_v0.py
python tests/test_q1_v1.py
python -m src.q1.experiments_v1 --cases case_093 case_026 --cores 2 3 4 5 --algorithm all --budget 4
```

`--algorithm` 支持 `v0`、`v1_partition`、`v1_schedule`、`v1` 和 `all`。每个版本最多有 4 个候选进入官方评估；候选总数、实际评估次数、缓存命中和耗时分别记录。结果保存在 `results/q1_v1_lite/`。算法与实验判断见 `V1_LITE_AUDIT.md` 和 `V1_LITE_RESULTS.md`。
