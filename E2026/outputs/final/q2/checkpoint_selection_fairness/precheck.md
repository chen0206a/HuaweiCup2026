# Q2 checkpoint 选择口径预检（2026-09-25）

状态：**无法用目前可访问的历史权重完成 B0/P2 × CleanSelect/RobustSelect 的完整 2×2 对照**。本次只读盘点本地文件，没有重训、推理或改写锁定结果。原训练服务器 `cn-north-b.ssh.damodel.com:49826` 当前拒绝连接，因此远端是否另存逐 epoch 权重尚未核验。

所有路径以下表所示，均相对于 `E2026/outputs/checkpoints/`。本地冻结场景定义 `outputs/metrics/b2_benchmark_definition.json` 的 SHA256 复核为 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`。

| 模型 | seed | 本地合格权重数 | 已保存权重与 SHA256 | 历史 epoch 记录 | 两种选择能否恢复 |
|---|---:|---:|---|---|---|
| B0-WCE | 42 | 1 | `b0_weighted_ce_best_selection_score.pt` · `20231a1436f1908f9271f420d66d209bd9e4f5e0b4c7bb9757820fa0f7e6e80a` | 15 epoch；clean 最优 3 | 仅 CleanSelect |
| B0-WCE | 43 | 1 | `b21_b0_seed_43_best_selection_score.pt` · `cf51892f79f135faf5e38ff338363552b52aa68b5616c99ea6489ca73f52a3de` | 15 epoch；clean 最优 3 | 仅 CleanSelect |
| B0-WCE | 44 | 1 | `b21_b0_seed_44_best_selection_score.pt` · `45d60973f3e1b0312fbc88e9f45cbe0a594b75288c59a27fbd781699aeeccc4a` | 16 epoch；clean 最优 4 | 仅 CleanSelect |
| B5-P2 | 42 | 2 | `b5_pooling_p2_best_clean_score.pt` · `4a4ab039f3e239961bf68e7afea6a187932f00b14481a8bc757df61ac05644a0`；`b5_pooling_p2_best_robust_score.pt` · `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff` | 27 epoch；clean 最优 20，robust 最优 15 | 两种均可 |
| B5-P2 | 43 | 1 | `b5_p2_multiseed_seed43_best_robust_score.pt` · `e4f814a5a986a37e217794a80fe5fb7e65c595b213844381076b63810b939e89` | 24 epoch；两规则均选 12 | 同一权重可用于两种规则 |
| B5-P2 | 44 | 1 | `b5_p2_multiseed_seed44_best_robust_score.pt` · `cbd486e03a44f86311c5a61896846e529ce4d91f53b8a47d31b645fd08ea5d81` | 13 epoch；两规则均选 1 | 同一权重可用于两种规则 |

B0 的三份历史日志为 `outputs/metrics/b0_weighted_ce_score_selection_history.json`、`b21_b0_seed_43_history.json`、`b21_b0_seed_44_history.json`。它们逐 epoch 只记录 clean validation loss、四指标及 selection score `S`，没有 54 缺失场景指标或 robust score `R`。本地也没有这三条训练轨迹的逐 epoch 权重；`src/training/train.py` 保存 best-valid-loss 和 best-selection，而非每个 epoch。即使找回 best-valid-loss 权重，它也不能代表在所有 epoch 中按 `R` 选出的最优权重。`b2_b0_*` 属于 BlockMask 增强训练，`b3_b0_*` 属于重构实验，均不是本次原始 B0 候选。

P2 三条逐 epoch 日志位于 `outputs/metrics/b5_pooling_history.json`（seed42）及 `b5_p2_multiseed_history.json`（seed43/44），包含 clean、54 场景均值及 `R`；seed43/44 的 best-clean 与 best-robust epoch 相同，所以无需另一个物理文件。seed42 两者不同，且两个权重均在本地。现有文件足以做 **统一 CleanSelect** 的三 seed 配对比较，但不足以填写 B0 + RobustSelect 一列，因此现在不能形成要求的完整对照，也不能从现有历史总分定量归因“结构收益”与“选模收益”。

完成完整 2×2 对照的优先补救是找回 B0 seed42/43/44 原训练轨迹中每个 epoch 的权重，然后在冻结 54 场景上逐 epoch 复评并按既有 `R` 选取。若这些权重确已丢失，则需要在原配置、数据顺序和 seed 下重跑 B0 三条训练轨迹，并同时保存 clean/robust 两个最优权重；这属于重训，需先单独确认，且必须验证新 CleanSelect 是否复现历史 checkpoint。P2 由各 seed 的历史 CleanSelect B0 初始化，后续解释跨模型差异时还需注明这一依赖。

本预检按任务停止条件结束；没有生成 2×2 指标表，也没有使用 Attachment2 test、Attachment3 或 Attachment4。
