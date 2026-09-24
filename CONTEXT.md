# Current Context

日期：2026-09-24。用户当前明确选定 E 题（D005）。E 代码、结果与实验位于 `E2026/`。

## Current progress

Q2 已完成数据审计、B0/B1、连续缺失基准、三种子稳定性、重建失败验证、B4′单种子门控筛选，以及 B5-P 单种子 pooling 消融。
当前稳定基线 B0-WCE：三种子 robust score 0.741677±0.001458；B4′仅seed42为0.743974，对同seed B0增加0.003726。均为内部验证分数，非官方竞赛总分。
证据：`E2026/outputs/metrics/b21_multiseed_summary.json`、`b4_reliability_metrics.json`、`b5_pooling_metrics.json`。
当前交接与候选计划：`E2026/Q2_PROGRESS_REVIEW.md`。
Q1/Q3 的 E 题完成情况本轮尚未核实，不能将 A 题旧成果混入。

## Immediate next steps

用户暂停 B4′ 后续实验。B5-P seed42：P0/P1/P2 robust 分别 0.740248/0.744320/0.745847；B0 参数冻结且完全相等。P2 的15个原生 vision 全零验证样本四指标均退化；P1 的该子集回归退化。当前只完成 seed42 筛选，未补 seed43/44，也未进入 fusion。
B0已有LayerNorm；原生零不等于人工缺失；B4′质量感知尚未成立。历史B0 clean选轮与B4′robust选轮需分别披露并补公平对照。
冻结54场景基准不变。早期b0_metrics.json已有test评估，不能称全项目test从未读取；后续B2–B5-P明确未使用test/附件3。共享validation_protocol仍是旧待定文档，本轮未修改；E既有规则见其README和benchmark定义，后续需经全局审阅同步。
算力：B5-P 在 RTX3090 24GB 服务器运行，并通过 CUDA sanity test。

## Historical A work

原A快照保存于 `archive/A_CONTEXT_before_E_20260923.md` 和 `archive/A_STATUS_before_E_20260923.md`。
根目录 `q1/`、`q2/`、`q3/` 仍是 A 的历史交接，代码与结果保留；不要当成 E 的进度。
