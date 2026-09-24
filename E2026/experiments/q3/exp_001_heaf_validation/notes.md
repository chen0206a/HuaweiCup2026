# Q3-1 HEAF validation

日期：2026-09-24。状态：**HEAF_VALIDATION_PASSED**。本实验只对已冻结 B5-P2 做推理与输入扰动，没有训练任何模型。正式配置为 `config.yaml`（与 `../../../configs/final/q3_heaf_validation.yaml` 同步），机器结果为 `metrics.json`（与 `../../../outputs/q3/heaf_validation_metrics.json` 同步）；解释见 `../../../outputs/q3/heaf_validation_report.md`。

从 Git 仓库根目录复现：`python E2026/scripts/run_q3_heaf_validation.py`。运行器先核验 checkpoint SHA256 与 Q2 clean 指标，停止门槛失败时不执行后续阶段。

## 为什么运行

在附件4解封前验证 Q3-0 的 8 联盟 exact Shapley、两任务独立主模态、配对交互、连续时间窗和扰动忠实性。仅使用 Attachment2 valid。官方 `aligned_50.pkl` 是单体 pickle，读取 valid 会反序列化容器，但代码只索引与评价 valid，未使用 train/test 样本，也未解析附件3/4。

## 锁定预测器与复现门槛

主预测器为 P2 seed42 best-robust checkpoint，SHA256 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`。严格加载 state_dict，mean_attention、eval、no_grad、float32、无标准化。728 条 valid clean 得到 Accuracy 0.6483516484、Macro-F1 0.6234347967、MAE 0.6082603505、Pearson 0.6461703738；与 Q2 锁定四指标的最大绝对差为 `4.01e-9`，低于 `1e-6` 停止阈值。

## 设计组、审计组和窗口锁定

按 `sha256("20260924|video_id")` 首字节最低位分组：design 332 条/113 video_id，audit 396 条/126 video_id，video_id 无交叉。全量 ID 在 `validation_ids.json`。design 组三个候选 ρ=0.10/0.20/0.30 的平均 top−随机同长度 class-margin drop 为 0.268837/0.366684/0.440487；按预定最大值规则锁定 **ρ=0.30、stride=1**，随机 32 次、seed=20260924。没有用 audit 组选 ρ。

## 归因与忠实性

728 条样本的 8 联盟均输出有限值。Shapley 效率最大绝对误差：分类 `1.78e-15`，回归 `4.44e-16`；人工遮挡只清零有效位置，padding、原生零标记及源张量检查覆盖全部 728 条并通过。分类/回归主模态一致率 83.93%。配对交互的有符号统计在 metrics；负值只称 negative interaction。

Audit 组同长度单区间删除：class-margin top−random 均值 **+0.400735**，按 video_id 分组 bootstrap 95% 区间 `[0.363538, 0.437243]`；confidence top−random **+0.077718**，区间 `[0.071170, 0.084282]`。10/20/30/40% 删除曲线的 class-margin top−random 均值分别为 +0.108296/+0.237758/+0.418076/+0.465465；10% 曲线仍有 11.36% 样本出现负 top margin drop。代表性正例和失败例来自 audit 原始诊断文件，不含原始媒体或附件4信息。

**解释限度：** 最重要区间按同一组窗口的最大 margin drop 选出，因此与这些窗口的随机抽样相比，margin 优势部分由选择规则保证。这个检验是实现与扰动一致性检查，不能单独作为外部忠实性或因果证明。10%–40% 删除曲线使用与选窗长度不同的删除预算，提供额外检查；其中个别样本失败，须在论文中保留。

## Seed 稳定性与下一步

固定 audit ID 和 ρ 后，seed43/44 分类主模态与 seed42 一致率 86.87%/83.08%，回归主模态一致率 77.78%/78.28%；平均同模态区间 IoU 0.579/0.527。分类预测类别一致率均为 89.65%，不同类别样本的分类 Shapley 排名比较须谨慎。详细相关、配对记录和 32 次对照保存在 `seed43/44_*` JSONL。

Q3-1 达到预设门槛，但**还没有原始媒体定位证明**。下一步仅在另行启动 Q3-2 后检查附件4无标签输入接口与槽位→原始媒体映射；不得用其分布改变冻结的 checkpoint、ρ 或删除协议。本轮不生成最终解释卡、不媒体回看、不解封附件4。
