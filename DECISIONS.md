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

## D009 - E Q3-2.6 text feature row identity

决定：Attachment4 aligned-50 的 text feature row 与 `text_bert` token slot 同索引语义判为 **VERIFIED**。`evidence_grounding.py` 可沿该映射和固定 tokenizer offsets 返回原文片段；只含 special token 的区间不返回片段。Audio/Vision 继续 `UNVERIFIED`，Q3 总状态为 `PARTIAL_GROUNDING_READY`。
依据：`E2026/outputs/q3/q3_text_row_identity.json` 中 20 样本、604 有效行的 source-supported final-layer 重建和 full cosine matrix：604/604 对角 argmax，mean diagonal cosine 0.999999994、RMSE 8.3977e-7、最大绝对差 3.0041e-5；真实样本 grounding smoke test 通过。公开代码未锁定 Attachment4 历史使用的确切权重 revision，因此不宣称该历史权重来源已固定。
影响范围：只更新 E Q3 文本证据定位状态和 API；不修改 HEAF/P2、Q2 历史结果或音频/视觉映射。未运行全量 Attachment4 解释。日期：2026-09-24。

## D010 - E Q3-2.7 A/V semantic-position gate

决定：公开 aligned-feature 接口与预处理源码不足以证明 Attachment4 的 audio/vision 第 i 行与已验证 text/token 第 i 槽具有相同语义位置。保持 AUDIO/VISION=`UNVERIFIED`；在任何 MP4 音频提取、forced alignment、视频解码或 PTS 推断之前停止。总状态继续 `PARTIAL_GROUNDING_READY`。
依据：Attachment4 没有 align 记录、slot timestamps、frame/PTS/source IDs 或 producer/hash provenance；公开 MMSA 只记录兼容格式，MMSA-FET 的 token expansion 属于另一明确 pipeline，CMU SDK 的 alignment 需要具名 reference sequence 和 timestamp intervals。审计：`E2026/outputs/q3/q3_av_grounding_audit.{md,json}`。
影响范围：只锁定本轮 grounding 审计结论；不修改 P2、HEAF、Shapley、rho、faithfulness 或最终预测协议。日期：2026-09-24。

## D011 - E Q3-3 Attachment4 final inference complete

决定：在 D007 锁定的 HEAF 协议和 B5-P2 seed42 checkpoint 上完成 Attachment4 aligned_50 20 个样本的最终无标签解释推理。checkpoint SHA256 前后均为 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`；不修改模型、解释协议或 A/V grounding 状态。Text evidence 仅在 VERIFIED 的 slot/token/raw_text 路径上填写；Audio/Vision 保持 UNVERIFIED，媒体时间/frame 为 null。输出状态 `Q3_FINAL_INFERENCE_COMPLETE`。
依据：`E2026/outputs/q3/final/attachment4_delivery_check.md`、`attachment4_summary.json` 及 20 条 schema-validated JSONL。20/20 输入/配对通过，Shapley efficiency 最大误差低于 `9e-16`。分类主模态 Text/Vision/Audio=`19/1/0`，回归=`18/2/0`，主模态一致 `17/20`；Grounding verified/unverified=`19/1`。无真实标签或性能指标，未使用 Attachment2 test / Attachment3，未训练或调参。
影响范围：仅完成并记录 E Q3 Attachment4 最终推理与解释交付；不改变 Q1/Q2、P2、HEAF、rho 或 faithfulness 协议。日期：2026-09-24。

## D012 - E Q2 Attachment3 text interface and final inference

决定：附件3对齐版30文件仅含 `text_bert/audio/vision`，先复用 Q3-2.6 已验证的 `bert-base-uncased` 有效前缀、最后层逐token重构路径。固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594`、权重SHA256 `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`；在20个同时含官方 `text` 的附件4无标签样本上先验证604/604同索引，特征最大差 `3.0041e-5`、RMSE `8.3977e-7`，锁定Q2模型logits/回归最大差 `2.4438e-6`/`1.1176e-6`，均通过预先固定容差。随后仅以该固定适配器生成附件3文本特征，完成30/30最终无标签推理。
依据：`E2026/outputs/final/q2/attachment3/attachment3_text_interface_audit.{md,json}`、`attachment3_summary.json`、`attachment3_delivery_check.md`。Q2 checkpoint、audio、vision、padding mask及回归后处理不变；未用附件3分布调参、未读标签、未用附件2 test。历史预计算BERT的原始revision仍未独立留档，结论限于现有数值接口近似一致。影响范围：E Q2附件3最终推理接口与交付；不改变D006主模型或验证协议。日期：2026-09-25。

## D013 - E Q2 public baseline results locked after MISA sanity check

决定：MISA 回归分支 sanity 检查通过；标签目标尺度与其他模型一致，回归头无激活并输出 `[B]`，回归及辅助目标梯度均非零，独立评价复算一致。保留 TFN/MulT/MISA 已有三 seed baseline 指标，不因 MISA 较高 MAE 重训或调参。将 clean validation 对照和 54 场景描述性结果写入 Q2 正文；因公开方法按 clean validation 选 checkpoint 而本文模型沿用固定缺失场景验证表现确定的 checkpoint，缺失表不用于宣称公平鲁棒排名。建议停止新增公开 baseline 训练。
依据：`E2026/outputs/final/q2/public_baselines/misa_regression_sanity_check.md`、`baseline_paper_integration_check.md`、`Baseline_Experiment_Report.md`。影响范围：E Q2 baseline 结果与论文正文；不改变 D006 主模型、冻结 benchmark、Attachment2 test 使用边界或其他题目。日期：2026-09-25。
