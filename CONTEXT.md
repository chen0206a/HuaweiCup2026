# Current Context

日期：2026-09-25。当前选题为 E（D005），E 代码与结果位于 `E2026/`。根目录 `q1/`、`q2/`、`q3/` 属于历史 A 题，不可与 E 的问题编号混用。

## E Q3

Q3-2.6 已通过 20 个 Attachment4 aligned 样本审计：公开 MMSA/Self-MM/MMSA-FET 代码都直接返回 BERT 最后层 token states；固定 `bert-base-uncased` 候选对 604 个有效文本槽的同索引 cosine argmax 为 604/604，均值 cosine `0.999999994`，RMSE `8.40e-7`，最大绝对差 `3.01e-5`。因此 `text feature row i ↔ text_bert token slot i` 判为 VERIFIED，正式 grounding API 已可用 tokenizer offsets 输出 raw-text span。历史生成未固定具体 BERT 权重 revision，作为来源限制保留。Text=VERIFIED，Audio/Vision=UNVERIFIED，总体 `PARTIAL_GROUNDING_READY`。诊断与复现见 `E2026/outputs/q3/q3_text_row_identity_report.md` 和 `E2026/experiments/q3/exp_0026_text_row_identity/`。本阶段未跑 HEAF 全量解释。

Q3-2 已先锁定 HEAF 方法（`E2026/configs/final/q3_heaf.yaml`、`E2026/outputs/q3/q3_method_lock.{md,json}`），然后只审计 Attachment4 对齐版。20 pkl 与 20 MP4 的 ID 精确对应，接口合法，但音频/视频没有逐槽时间映射，仍 UNVERIFIED；不允许据视频时长均分 50 槽或照搬 Q1 窗口。早期整体 `BLOCKED_BY_GROUNDING` 记录见 `E2026/outputs/q3/q3_grounding_audit.{md,json}`，经 Q3-2.6 后当前为 `PARTIAL_GROUNDING_READY`。

Q3-2.7 审核公开 MMSA、MMSA-FET 与 CMU SDK 对齐接口后，确认这些资料只说明候选格式/方法，没有 Attachment4 文件级 provenance、A/V row→token row 关系或 word-to-WordPiece 展开规则。入口 gate 未通过，已按要求在媒体音频提取/forced alignment/PTS 抽取前停止。TEXT=VERIFIED，AUDIO/VISION=UNVERIFIED，overall=`PARTIAL_GROUNDING_READY`。见 `E2026/outputs/q3/q3_av_grounding_audit.{md,json}` 与 `E2026/experiments/q3/exp_0027_av_grounding/`。

Q3-3 已在锁定协议下完成 Attachment4 aligned_50 的 20/20 全量 frozen P2 + HEAF 推理。checkpoint 前后 SHA256 一致；输入/MP4 ID、形状、finite、mask、Shapley efficiency、schema 与文本 span QA 全通过。无标签指标、训练、调参、A2 test 或 Attachment3 使用。分类主模态 Text/Vision/Audio 为 19/1/0，回归为 18/2/0，分类与回归主模态一致 17/20；原始 grounding verified 19、unverified 1，A/V 时间/帧字段均 null。状态 `Q3_FINAL_INFERENCE_COMPLETE`。交付见 `E2026/outputs/q3/final/attachment4_delivery_check.md`、CSV/JSONL/summary；执行脚本与正式记录见 `E2026/scripts/run_q3_attachment4_final.py`、`E2026/experiments/q3/exp_003_attachment4_final_heaf/`。不据此声称预测正确性或现实因果解释。

Q3 Figure 8/9 已统一重绘为低饱和中文论文风：浅雾蓝/浅米黄/浅粉模态色，黑色细边与点纹柱体，深蓝遮挡曲线和真实 grouped-bootstrap 95% 半透明置信带。Figure 8 保留 Attachment2 validation 三幅统计与全部数值；Figure 9 v2 保留 Attachment4 样本 14/02 和固定相对位置的原视频场景截图，原始文本证据仍按英文原文引用。视觉槽位→视频时间仍 UNVERIFIED，截图不表示关键帧。新旧 PNG/PDF/SVG、脚本、截图清单、图注和归档见 `E2026/outputs/final/q3/figures/`；原始预测、解释与验证结果未改变。

Q3 Figure 7 v2 从用户提供的 Q2 架构图 PPTX 副本改制，保留四个虚线框的整体布局，内容改为输入、情感预测、HEAF 与解释输出。原文、音频波形与视频场景统一取自配对样本 09，避开 Figure 9 的 14/02 案例及 Q2 原图中的画面；截图和波形仅示意原始输入，不表示 HEAF 定位。可编辑 PPTX、PNG/PDF/SVG、来源清单与脚本见 `E2026/outputs/final/q3/figures/`。

Q3-1 HEAF 在 Attachment2 valid 完成冻结预测器复算、8 联盟 exact Shapley、连续窗口和忠实性验证；结果为 `E2026/outputs/q3/heaf_validation_metrics.json`、报告与 `E2026/experiments/q3/exp_001_heaf_validation/`。主模型仍为 B5-P2 seed42，checkpoint 哈希 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`；valid clean 与 Q2 锁定指标最大差 `4.01e-9`。按 video_id 分成 design 332/audit 396，锁定 ρ=0.30、stride=1。Audit 同长度 top−random class-margin 均值 +0.400735，95% video-group bootstrap 区间 `[0.363538,0.437243]`；此比较因 top 由同窗口最大值选出而有选择优势，10% 删除曲线仍有失败样本。seed43/44 仅作稳定性审计。Q3-3 随后在 Attachment4 输出最终解释卡。

## E Q2

用户已正式锁定 **B0-WCE 为 baseline，B5-P2 attention residual pooling 为最终主模型**。不再进行 Q2 模型探索、调参或组合。冻结验证协议为 attachment2 validation clean + 54 个连续缺失场景，benchmark seed 20260923，SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`。

B0 三 seed robust `0.741677 ± 0.001458`；P2 `0.744777 ± 0.001492`；paired delta `+0.003100 ± 0.002907`（sample SD）。seed42/43 为正，seed44 基本持平，结果存在 seed 敏感性。依据：`E2026/outputs/metrics/b5_p2_multiseed_summary.json`。最终配置、数据与 checkpoint manifest、实验索引见 `E2026/configs/final/`、`E2026/data/manifests/`、`E2026/outputs/final/q2/`。

本地 attachment2 原始文件实际位于 `E2026/data/raw/`；历史服务器配置使用 `data/raw/attachment2/`，该差异已记录，原始大文件未移动。六个 B0/P2 主 checkpoint 均在本地并完成哈希索引。Attachment3 对齐版30文件仅提供 `text_bert/audio/vision`；Q3 固定 BERT 路径在附件4的20个已知 `text` 样本上通过604行特征及锁定 Q2 模型输出的数值回归后，本地完成30/30无标签最终推理。预测CSV与审计见 `E2026/outputs/final/q2/attachment3/`，无标签性能指标；Q2 checkpoint 与验证协议未变。Attachment2 test 没有参与模型选择或本次适配。

B5-P1 只有 seed42 screening；B4′ 仍为单 seed 诊断；fusion、reconstruction、loss 调整和文本 z-score 路线均未进入最终模型。详细原因以既有实验结果和 `E2026/outputs/final/q2/q2_experiment_index.json` 为准。Q1审查见下节；Q3 的 E 题建模进度未核实，本轮没有启动 Q3。

## E Q1/Q2 审查（2026-09-24）

审查报告：`notes/E_Q1_Q2_REVIEW_20260924.md`，复算证据：`notes/e_q1_q2_audit_evidence_20260924.json`。Q1实际交付目录为未跟踪的`Q1_方法流程与实验结果_交付包/`，包含方法与汇总，但不含全量特征、100条timeline和部分复现依赖。Q1方法及辅助实验可收口，当前包不能作为完整提交包；25/25辅助CV折存在同video_id跨训练/测试，结论限于诊断。Q1附件1与Q2附件2/3的数据分工符合官方要求，不需跨问特征衔接。

Q2六checkpoint/benchmark哈希及冻结B0张量核验通过，主要汇总数值复算一致；附件2官方三split的video_id无交集。P2缺失绝对综合分数较高，但相对clean的降幅大于B0，不能声称退化更小或各指标全面提高。已有同clean选模记录支持平均paired robust增量0.003046，无新训练。此段记录的是2026-09-24审查时状态；附件3预测CSV及推理入口现已按上节补齐，D006和验证协议仍未改变。

## Historical A work

A 题快照保存于 `archive/A_CONTEXT_before_E_20260923.md` 和 `archive/A_STATUS_before_E_20260923.md`。
