# E2026 Q2 缺失规律与公开 Baseline 数据预检

**本地数据完整性检查先于绘图与制表。** 本报告根据已有本地结果文件生成；未训练、未运行推理、未读取 test、未读取 Attachment3/4，也未修改正文。

## A. 分模态缺失率曲线

- 场景来源：`E2026/outputs/final/q2/q2_plotting_handoff_v2/plot_data/scenario_details_complete.csv`（330 行，2 模型 × 3 seed × 55 条件）。
- P2 单模态场景记录：**135/135**；唯一组合：**135/135**。
- Seed：42/43/44；模态：文本/音频/视觉；ρ：0.1–0.5；位置：early/middle/late，全部齐全。
- Accuracy、Macro-F1、MAE、Pearson 数值全部有限；每条记录 `sample_count=728`。
- 冻结 benchmark SHA256：`3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`；Q2 模型锁、benchmark manifest 与 benchmark definition 一致。
- P2 seed42/43/44 checkpoint 与 Q2 checkpoint manifest 校验通过；模型锁引用的多 seed 汇总结果 SHA256 也匹配。
- 与已完成 Figure 4 逐 seed 聚合文件核对：每个 seed × 模态 × ρ 均为三个位置等权平均，45/45 项四项指标一致（绝对容差 1e-12）。
- 完整输入参考值来自 `q2_model_lock.json` 的 P2 clean validation 三 seed 原始值，且与场景表 clean 行一致。

**CURVE_DATA_COMPLETE = YES**

## B. TFN / MulT / MISA 公开 Baseline 缺失场景

- 数据来源：每个模型/seed 的既有 `E2026/experiments/q2/public_baselines/<model>/metrics_seed<seed>.json`，其中含 clean 与逐场景结果；另有锁定摘要及 checkpoint manifest。
- 每模型缺失记录：TFN **162/162**；MulT **162/162**；MISA **162/162**。
- Seed 42/43/44 各有 54/54 场景；三个模型的 scenario ID 集合与冻结 benchmark 的 54 项完全一致。
- 九份结果的 benchmark SHA、clean 行、finite 指标和非 smoke 状态均通过；九个 checkpoint 文件大小与 SHA256 均匹配 manifest。
- 全量缺失明细共 **486/486** 条唯一模型 × seed × 场景记录；Accuracy、Macro-F1、MAE、Pearson、selection score 均有限。
- 不纳入 Attachment2 test、Attachment3 或 Attachment4。已存在的实验报告将这些 public baseline 定义为 Attachment2 train/valid 的适配训练与 valid 评估结果。

**BASELINE_MISSING_DATA_COMPLETE = YES**

## C. 门槛结论

两项数据门槛均满足，可以继续绘制分模态退化曲线并生成描述性缺失场景对比表。Baseline checkpoint 按 clean validation 选择，锁定 P2 按 robust score 选择；比较表必须标为**描述性比较**，不能据此声称严格公平的鲁棒性排名或显著优越。
