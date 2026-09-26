# Current Context

日期：2026-09-25。当前选题为 E（D005），E 代码与结果位于 `E2026/`。根目录 `q1/`、`q2/`、`q3/` 属于历史 A 题，不可与 E 的问题编号混用。

## E Q1

2026-09-26：用户提供 `D:/java录屏/q1_completion_20260926.zip`，实际恢复并核验最终三模态FP32矩阵各(100,50,768)、bool掩码各(100,50)、100唯一ID与manifest行序、15000条来源。有效位4121/4955/5000，原生与逐样本/汇总向量逐值对应；包内时间验证记录支持100样本共同零点和23241帧PTS一致性。独立证据更新版Q1预览见 `outputs/q1_rewrite_preview_v2/`，15页。最新叙事重构见 `outputs/q1_story_rewrite/`，12页（含参考文献），保留10个公式、7张原图和100条表2记录；正式论文未合并、实验结果未改。详细边界见 `q1_repro_audit_v2.md`：本轮未重新解码媒体，GPU Python/OS未记录，不虚构版本。此前“全量特征未定位”仅为补包前阶段状态。

## E Q3

2026-09-27：以用户提供的 `D:/java录屏/07_q3.tex` 和 Q3 PDF 为本轮输入，完成独立终稿叙事修订 `outputs/q3_story_refine/`（12页）。精确公式、7张表和20条附件4数值保持原样；核实0.10/0.20/0.30比例选择及Spearman定义，三图仅改文字后从源重新导出，74项检查通过。未训练、未改预测/解释数据、未合并正式论文；修改记录见目录内两个change log。

2026-09-25 Attachment4 原始证据映射 PRECHECK：两版各 20 pkl/20 MP4，80/80 文件哈希与清单一致。全部非零 aligned 音频/视觉行分别有 564/564、534/534 个同样本未对齐数组的唯一精确数值匹配；sample02 视觉槽位 `[1,7)` 对应未对齐视觉行 36–41，但没有未对齐行到原始 MP4 frame/PTS 的记录。文本 19/19 解释片段仍可由原文字符区间核对。TEXT=`VERIFIED`，AUDIO/VISION 原始时间=`UNVERIFIED`，sample02 原视频关键帧映射=`NO`；未抽帧或切音频。详见 `E2026/outputs/q3/q3_raw_evidence_mapping_precheck.md`。

Q3-2.6 已通过 20 个 Attachment4 aligned 样本审计：公开 MMSA/Self-MM/MMSA-FET 代码都直接返回 BERT 最后层 token states；固定 `bert-base-uncased` 候选对 604 个有效文本槽的同索引 cosine argmax 为 604/604，均值 cosine `0.999999994`，RMSE `8.40e-7`，最大绝对差 `3.01e-5`。因此 `text feature row i ↔ text_bert token slot i` 判为 VERIFIED，正式 grounding API 已可用 tokenizer offsets 输出 raw-text span。历史生成未固定具体 BERT 权重 revision，作为来源限制保留。Text=VERIFIED，Audio/Vision=UNVERIFIED，总体 `PARTIAL_GROUNDING_READY`。诊断与复现见 `E2026/outputs/q3/q3_text_row_identity_report.md` 和 `E2026/experiments/q3/exp_0026_text_row_identity/`。本阶段未跑 HEAF 全量解释。

Q3-2 已先锁定 HEAF 方法（`E2026/configs/final/q3_heaf.yaml`、`E2026/outputs/q3/q3_method_lock.{md,json}`），然后只审计 Attachment4 对齐版。20 pkl 与 20 MP4 的 ID 精确对应，接口合法，但音频/视频没有逐槽时间映射，仍 UNVERIFIED；不允许据视频时长均分 50 槽或照搬 Q1 窗口。早期整体 `BLOCKED_BY_GROUNDING` 记录见 `E2026/outputs/q3/q3_grounding_audit.{md,json}`，经 Q3-2.6 后当前为 `PARTIAL_GROUNDING_READY`。

Q3-2.7 审核公开 MMSA、MMSA-FET 与 CMU SDK 对齐接口后，确认这些资料只说明候选格式/方法，没有 Attachment4 文件级 provenance、A/V row→token row 关系或 word-to-WordPiece 展开规则。入口 gate 未通过，已按要求在媒体音频提取/forced alignment/PTS 抽取前停止。TEXT=VERIFIED，AUDIO/VISION=UNVERIFIED，overall=`PARTIAL_GROUNDING_READY`。见 `E2026/outputs/q3/q3_av_grounding_audit.{md,json}` 与 `E2026/experiments/q3/exp_0027_av_grounding/`。

Q3-3 已在锁定协议下完成 Attachment4 aligned_50 的 20/20 全量 frozen P2 + HEAF 推理。checkpoint 前后 SHA256 一致；输入/MP4 ID、形状、finite、mask、Shapley efficiency、schema 与文本 span QA 全通过。无标签指标、训练、调参、A2 test 或 Attachment3 使用。分类主模态 Text/Vision/Audio 为 19/1/0，回归为 18/2/0，分类与回归主模态一致 17/20；原始 grounding verified 19、unverified 1，A/V 时间/帧字段均 null。状态 `Q3_FINAL_INFERENCE_COMPLETE`。交付见 `E2026/outputs/q3/final/attachment4_delivery_check.md`、CSV/JSONL/summary；执行脚本与正式记录见 `E2026/scripts/run_q3_attachment4_final.py`、`E2026/experiments/q3/exp_003_attachment4_final_heaf/`。不据此声称预测正确性或现实因果解释。

Q3 Figure 8/9 已统一重绘为低饱和中文论文风：浅雾蓝/浅米黄/浅粉模态色，黑色细边与点纹柱体，深蓝遮挡曲线和真实 grouped-bootstrap 95% 半透明置信带。Figure 8 保留 Attachment2 validation 三幅统计与全部数值；Figure 9 v2 保留 Attachment4 样本 14/02 和固定相对位置的原视频场景截图，原始文本证据仍按英文原文引用。视觉槽位→视频时间仍 UNVERIFIED，截图不表示关键帧。新旧 PNG/PDF/SVG、脚本、截图清单、图注和归档见 `E2026/outputs/final/q3/figures/`；原始预测、解释与验证结果未改变。

Q3 Figure 7 v2 从用户提供的 Q2 架构图 PPTX 副本改制，保留四个虚线框的整体布局，内容改为输入、情感预测、HEAF 与解释输出。原文、音频波形与视频场景统一取自配对样本 09，避开 Figure 9 的 14/02 案例及 Q2 原图中的画面；截图和波形仅示意原始输入，不表示 HEAF 定位。可编辑 PPTX、PNG/PDF/SVG、来源清单与脚本见 `E2026/outputs/final/q3/figures/`。

Q3-1 HEAF 在 Attachment2 valid 完成冻结预测器复算、8 联盟 exact Shapley、连续窗口和忠实性验证；结果为 `E2026/outputs/q3/heaf_validation_metrics.json`、报告与 `E2026/experiments/q3/exp_001_heaf_validation/`。主模型仍为 B5-P2 seed42，checkpoint 哈希 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`；valid clean 与 Q2 锁定指标最大差 `4.01e-9`。按 video_id 分成 design 332/audit 396，锁定 ρ=0.30、stride=1。Audit 同长度 top−random class-margin 均值 +0.400735，95% video-group bootstrap 区间 `[0.363538,0.437243]`；此比较因 top 由同窗口最大值选出而有选择优势，10% 删除曲线仍有失败样本。seed43/44 仅作稳定性审计。Q3-3 随后在 Attachment4 输出最终解释卡。

## E Q2

2026-09-27：Q2在独立叙事稿 `outputs/q2_story_rewrite/` 基础上完成终稿精修 `outputs/q2_final_refine/`（正文14页、参考文献1页；图8–14、表8–13不变）。明确S_clean结构/公开方法比较与R最终配置诊断的不同用途；按既有代码修正加权交叉熵归一化和线性打分器两处纸面公式，删除重复均值公式；核对11篇原文献并纠正错配；澄清604槽位检查来自20条无标签附件4及text_bert重构接口。7图、数值表、30条附件3预测、模型代码与正式论文均保持；57项检查通过，无overfull或未定义引用。无训练、调参、预测重跑或正式合并，状态 `Q2_FINAL_REFINE_COMPLETE`。证据和修改说明见该目录的reference/technical audit及change log。同日完成求解思路增强版 outputs/q2_richer_narrative/（15页）：2.1改为任务条件/基线选择/改进动机三段，补齐方法与实验阶段承接，表11/12均值排名按所列模型核实；18公式、全部图表数据及编号保持，未覆盖旧稿或正式论文。

2026-09-26 E Q2 expanded baseline：完成12个公开架构适配（TFN/LMF/MFN/MulT/MISA/Self-MM/MMIM/MAG-BERT/TFR-Net/MissModal/M3S/MMIN）与B0/P2共14模型、42/43/44三seed的统一CleanSelect比较。新增27次训练完成；clean与冻结54场景各42条seedwise结果、14条汇总。P2 clean四指标排名1/1/2/2，missing排名4/1/1/2（Acc/F1/MAE/Pearson）；不声称全面最优。missing-aware训练单独标记，所有模型为aligned-50机制适配。结果、来源、checkpoint清单与QA见 `E2026/outputs/final/q2/public_baselines_expanded/`；未修改论文、未用test或附件3/4选模。

用户已正式锁定 **B0-WCE 为 baseline，B5-P2 attention residual pooling 为最终主模型**。不再进行 Q2 模型探索、调参或组合。冻结验证协议为 attachment2 validation clean + 54 个连续缺失场景，benchmark seed 20260923，SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`。

B0 三 seed robust `0.741677 ± 0.001458`；P2 `0.744777 ± 0.001492`；paired delta `+0.003100 ± 0.002907`（sample SD）。seed42/43 为正，seed44 基本持平，结果存在 seed 敏感性。依据：`E2026/outputs/metrics/b5_p2_multiseed_summary.json`。最终配置、数据与 checkpoint manifest、实验索引见 `E2026/configs/final/`、`E2026/data/manifests/`、`E2026/outputs/final/q2/`。

公开架构对比已在新服务器使用 Attachment2 train/valid 独立重训 TFN、MulT、MISA，各 seed 42/43/44，并完成同一 54 场景缺失评估。robust 三 seed 均值±样本SD 分别为 `0.7139±0.0039`、`0.7253±0.0061`、`0.7168±0.0019`；详情和四项原始指标见 `E2026/outputs/final/q2/public_baselines/Baseline_Experiment_Report.md`。三者为 aligned-50 架构适配，不是原仓库原封运行；TFN 参考仓库无许可证，未复制源码。新 baseline 按 clean-valid score 选 checkpoint，历史 P2 按 robust score 选，缺失结果的横向差距有选择口径限制。未改锁定 Q2 模型，未用 Attachment2 test、Attachment3/4 或外部情感数据。

2026-09-25 完成 MISA regression sanity：训练/验证标签均直接来自同一原始 `regression_labels` 尺度，无归一化/反变换，回归头 shape、梯度与独立 MAE/Pearson 复算通过，未发现实现错误。三 seed MISA 回归绝对误差较高但 Pearson 保持；原 baseline 表锁定，不重训。Q2 正文已加入 clean 四指标表和 54 缺失场景描述表（明确 checkpoint selection 差异），Table 1/2 均在 `paper/main.pdf` 第3页。核验报告见 `E2026/outputs/final/q2/public_baselines/baseline_paper_integration_check.md`；Q2 外的论文 TODO 未改。

本地 attachment2 原始文件实际位于 `E2026/data/raw/`；历史服务器配置使用 `data/raw/attachment2/`，该差异已记录，原始大文件未移动。六个 B0/P2 主 checkpoint 均在本地并完成哈希索引。Attachment3 对齐版30文件仅提供 `text_bert/audio/vision`；Q3 固定 BERT 路径在附件4的20个已知 `text` 样本上通过604行特征及锁定 Q2 模型输出的数值回归后，本地完成30/30无标签最终推理。预测CSV与审计见 `E2026/outputs/final/q2/attachment3/`，无标签性能指标；Q2 checkpoint 与验证协议未变。Attachment2 test 没有参与模型选择或本次适配。

B5-P1 只有 seed42 screening；B4′ 仍为单 seed 诊断；fusion、reconstruction、loss 调整和文本 z-score 路线均未进入最终模型。详细原因以既有实验结果和 `E2026/outputs/final/q2/q2_experiment_index.json` 为准。Q1审查见下节；Q3 的 E 题建模进度未核实，本轮没有启动 Q3。

## E Q1/Q2 审查（2026-09-24）

审查报告：`notes/E_Q1_Q2_REVIEW_20260924.md`，复算证据：`notes/e_q1_q2_audit_evidence_20260924.json`。Q1实际交付目录为未跟踪的`Q1_方法流程与实验结果_交付包/`，包含方法与汇总，但不含全量特征、100条timeline和部分复现依赖。Q1方法及辅助实验可收口，当前包不能作为完整提交包；25/25辅助CV折存在同video_id跨训练/测试，结论限于诊断。Q1附件1与Q2附件2/3的数据分工符合官方要求，不需跨问特征衔接。

Q2六checkpoint/benchmark哈希及冻结B0张量核验通过，主要汇总数值复算一致；附件2官方三split的video_id无交集。P2缺失绝对综合分数较高，但相对clean的降幅大于B0，不能声称退化更小或各指标全面提高。已有同clean选模记录支持平均paired robust增量0.003046，无新训练。此段记录的是2026-09-24审查时状态；附件3预测CSV及推理入口现已按上节补齐，D006和验证协议仍未改变。

## Historical A work

A 题快照保存于 `archive/A_CONTEXT_before_E_20260923.md` 和 `archive/A_STATUS_before_E_20260923.md`。
