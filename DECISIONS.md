# Confirmed Decisions

> 只记录已确认且全员需要遵守的事项。普通讨论放在相应任务记录中。

## D001 - Repository and collaboration structure
决定：按当前仓库目录组织四个小问；使用 `CONTEXT.md` 作为当前快照、`DECISIONS.md` 作为全局决定、各 `qX/HANDOFF.md` 作为小问交接。
原因：三个独立账号需要共享简洁、可追溯的项目记忆。
影响范围：全仓库协作。
日期：2026-09-23

## D002 - Raw data handling
决定：`data/raw/` 默认不纳入 Git；只跟踪 `.gitkeep` 和数据说明。
原因：原始竞赛数据可能很大或受分发限制。
影响范围：数据管理和远程仓库。
日期：2026-09-23

## D003 - Validation protocol pending problem review
决定：目前不预设数据划分、指标或随机种子；赛题和数据结构确认后再填写统一验证方案。
原因：赛题尚未提供，提前选定会制造未经验证的假设。
影响范围：所有预测、分类和机器学习实验。
日期：2026-09-23

## D004 - Current topic selection: A
决定：根据用户明确指示，目前选定 A 题“通用神经网络处理器下的多核调度问题”。
原因：用户在六题比较后选择 A，并要求评审外部方案、提出可行路线。
影响范围：当前聚焦 A 的三个问题；q4 保留但不属于本题必答范围。算法仍是待实测验证的候选，不代表已确认性能结论。
方案记录：`notes/a_plan_review_20260923.md`。
日期：2026-09-23

## D005 - Current topic selection: E (supersedes D004 for active work)
决定：根据用户明确指示，当前选定 E 题“复杂场景下多模态情感识别的数学建模与算法设计”。
工作范围：本轮核对进度与下一步方案；E 项目现位于 E2026，A代码及交接保留为历史，不混用问题编号。
进度依据：E2026/outputs/metrics 中的真实结果；复核与待测建议见 E2026/Q2_PROGRESS_REVIEW.md。
本决定不批准或修改新的验证协议，不将候选模型视为已有效。沿用既有E冻结基准，协议同步需另行审阅。
日期：2026-09-23。

## D006 - E Q2 final main model lock

决定：按用户明确指示，E Q2 的论文 baseline 固定为 B0-WCE，最终主模型固定为 B5-P2（冻结 B0 的轻量 attention residual pooling）。停止 Q2 后续模型探索与调参，不组合其他模块。沿用 attachment2 validation clean + 54 缺失场景的现有冻结 benchmark。
依据：`E2026/outputs/metrics/b5_p2_multiseed_summary.json` 和 `E2026/outputs/final/q2/q2_model_lock.json`。三 seed paired robust delta 为 `+0.003100 ± 0.002907`（sample SD），2/3 为正、seed44 近乎持平；对外表述必须保留 seed 敏感性。
影响范围：E Q2 最终模型、实验汇总与论文表述；不改变其他题目或共享验证协议。日期：2026-09-24。

## D007 - E Q3 HEAF method lock before Attachment4 audit

决定：按用户明确指示，在解析附件4之前固定 B5-P2 seed42、aligned_50、8 联盟 exact Shapley、分类固定类别 log-odds、原始回归输出、三对 interaction、ρ=0.30/stride=1 连续遮挡、top 区间规则、10/20/30/40% 删除曲线、32 次随机对照及 seed 20260924，解释卡 schema v0.2.0。附件4不得用于重选这些参数。
依据：`E2026/outputs/q3/heaf_validation_metrics.json`、`E2026/configs/final/q3_heaf.yaml`、`E2026/outputs/q3/q3_method_lock.{md,json}`。影响范围：E Q3；不改变 D006 的 Q2 模型与历史结果。日期：2026-09-24。

## D008 - E Q3-2.5 provenance and token-grounding status

决定：Attachment4 当前只能确认与 MMSA/Self-MM aligned 数据结构高度吻合，不能确认来源。固定 `bert-base-uncased` tokenizer 对 `text_bert` 20/20 完整数组复现，准许将 token slot 映射到 raw_text character span；P2 768-D `text` feature row 到 token slot、audio/vision row 到时间均保持 `UNVERIFIED`，外层证据字段仍不得填值。Q3-2.5 总体维持 `BLOCKED_BY_GROUNDING`。
依据：`E2026/outputs/q3/q3_alignment_provenance.json`、`E2026/outputs/q3/q3_alignment_provenance_report.md`、`E2026/outputs/q3/provenance_sources.md`。影响范围：E Q3 证据 grounding；不改变 HEAF、P2、rho/stride 或 Q2 历史结果。日期：2026-09-24。
