# Project Status

- 2026-09-27 E Q3：上下文衔接修订完成，交付 `outputs/q3_context_final/` 的TeX、12页PDF及仅记录五处改动的change log。3.1改为三个自然段，3.2.2、3.5、3.7、3.8补充承接；其余正文、4公式、7表、3图和所有实验结果不变。20项核对及全页视觉检查通过，未合并正式论文。

- 2026-09-27 E Q3：独立终稿叙事优化完成，交付 `outputs/q3_story_refine/` 的TeX、12页PDF及正文/图件修改日志。移除重复流程与HEAF包装，保留4公式、7表、20条预测解释及两个典型案例；三图由原源仅修改标签重新导出，原始实验数字未变。74项核对及全页视觉检查通过；未合并正式论文。

2026-09-25：当前 E 题工作目录为 `E2026/`。以下较早条目保留其阶段历史状态。

- E Q3 Attachment4 原始证据映射 PRECHECK：80/80 本地文件哈希复核通过；非零 aligned 音频/视觉槽位均可在同一样本未对齐特征中找到唯一完全相等的行（564/564、534/534）。仍缺未对齐行到原音频时间、原视频 frame/PTS 的来源映射；sample02 `[1,7)` 只能确认视觉特征行 36–41，不能生成已验证关键帧。Text=`VERIFIED`，Audio/Vision=`UNVERIFIED`。见 `E2026/outputs/q3/q3_raw_evidence_mapping_precheck.md`；未修改论文或解释结果。

- E Q2 public baseline 收口：MISA regression sanity=`PASS`，未发现标签尺度、输出形状、梯度或评价实现错误；TFN/MulT/MISA 现有三 seed 结果锁定。论文 Q2 新增 clean 结果 Table 1 与 54-scenario 描述性 Table 2，均位于 `paper/main.pdf` 第3页。无进一步 baseline training 建议。完整检查见 `E2026/outputs/final/q2/public_baselines/baseline_paper_integration_check.md`。

- E Q2 公开架构对比：新服务器 RTX 3090 上完成 TFN、MulT、MISA 的 Attachment2 aligned-50 train/valid 三 seed（42/43/44）重训及冻结 clean + 54 缺失场景评估，9/9 checkpoint/指标 QA 通过。robust 均值±样本SD：TFN `0.7139±0.0039`、MulT `0.7253±0.0061`、MISA `0.7168±0.0019`；历史锁定 P2 `0.7448±0.0015`。P2 历史 checkpoint 按 robust score 选择，新 baseline 按 clean score 选择，缺失分数对比存在选择口径优势，报告已注明。结果、来源和适配清单见 `E2026/outputs/final/q2/public_baselines/`、`E2026/experiments/q2/public_baselines/`；未用 Attachment2 test 或 Attachment3/4 调参。

- E Q2 Attachment3：本地对齐版30个文件 SHA256 全匹配，但均缺预计算 `text`。复用 Q3 固定 `bert-base-uncased` revision 在附件4的20个已知样本复算604/604行同索引、特征RMSE `8.3977e-7`、锁定Q2模型logits/回归最大差 `2.4438e-6`/`1.1176e-6` 后，文本接口门槛 PASS。锁定checkpoint不变，30/30无标签最终推理及CSV重读/概率/ID QA通过；预测类计数 Negative/Neutral/Positive=`7/11/12`，仅为无标签分布。交付与正式记录见 `E2026/outputs/final/q2/attachment3/`、`E2026/experiments/q2/exp_015_attachment3_final_inference/`。

- E Q3-3：Attachment4 aligned_50 20/20 frozen B5-P2 seed42 + 锁定 HEAF 全量推理完成，状态 `Q3_FINAL_INFERENCE_COMPLETE`。checkpoint hash 前后不变；输入、ID、finite/mask、Shapley efficiency、schema、raw_text span 和 A/V null 字段 QA 通过。Text/Vision/Audio 分类主模态 19/1/0，回归 18/2/0，主模态一致 17/20；grounding verified 19、unverified 1。没有标签性能指标。交付 `E2026/outputs/q3/final/`；见 `attachment4_delivery_check.md`。

- E Q3 Figures 8/9：两图已统一重绘为中文低饱和论文风，采用浅蓝/浅黄/浅粉模态色、黑边点纹柱体与深蓝曲线。Figure 8 保留 Attachment2 valid 三幅原始统计和真实分组 bootstrap 95% 区间；Figure 9 v2 仍仅含 Attachment4 样本14/02及原视频场景截图（不代表 HEAF 关键帧）。PNG 为 300 dpi，PDF/SVG 为矢量，旧版均归档。A/V 原始时间 grounding 仍未验证；原始指标和解释结果不变。文件、图注与 QA 见 `E2026/outputs/final/q3/figures/`。

- E Q3 Figure 7 v2：基于用户提供的 Q2 架构图副本，保留四框总体布局，改为情感预测与 HEAF 方法内容；真实原文、MP4 场景和音轨波形统一取附件4样本09，避免与 Q2 原画面及 Figure 9 的14/02案例重复。视频/音频素材仅作输入示例，A/V 原始时间定位仍未验证。输出可编辑 PPTX、300 dpi PNG、PDF/SVG、来源清单与复现脚本于 `E2026/outputs/final/q3/figures/`，不修改实验结果。

- E Q3-2.6：BERT 最后层重建与公开预处理代码支持 Attachment4 text 行对应 `text_bert` 同索引 token；604/604 cosine 行最大值在对角，均值 cosine 0.999999994，RMSE 8.40e-7，最大绝对差 3.01e-5。Text grounding 已接入 tokenizer offsets 并对真实样本做 API smoke test。Text=`VERIFIED`，audio/vision=`UNVERIFIED`，当前=`PARTIAL_GROUNDING_READY`。无 HEAF 推理、无训练。详见 `E2026/outputs/q3/q3_text_row_identity_report.md` 与 `E2026/experiments/q3/exp_0026_text_row_identity/`。

- E Q3-2.7：公开接口与预处理资料不能把 Attachment4 的 A/V 第 i 行绑定到已验证 text/token 第 i 槽。按 gate 要求，在提取 MP4 音频、forced alignment、解码视频/PTS 之前停止。TEXT=`VERIFIED`，AUDIO/VISION=`UNVERIFIED`，overall=`PARTIAL_GROUNDING_READY`。未运行 HEAF。见 `E2026/outputs/q3/q3_av_grounding_audit.{md,json}`。

- E Q3-2（阶段结果）：HEAF 方法已在访问 Attachment4 内容前锁定。对齐版 20 特征/20 视频精确 ID 匹配，全部 feature shape、finite、padding 与 inventory hash 检查通过。当时缺逐槽原始证据映射，阶段结论为 `BLOCKED_BY_GROUNDING`；Q3-2.6 后当前状态已升级为 `PARTIAL_GROUNDING_READY`。见 `E2026/outputs/q3/q3_grounding_audit.{md,json}`。

- E Q3-2.5（阶段结果）：固定版本 `bert-base-uncased` fast tokenizer 对 Attachment4 的 `text_bert` IDs/mask/type IDs 20/20 完全复现，token slot 到 raw_text character span 已 VERIFIED；当时 P2 `text[50,768]` 行同索引语义仍 UNVERIFIED，故该阶段维持 `BLOCKED_BY_GROUNDING`。Q3-2.6 已解决文本行身份；A/V 时间行仍未能 join 到 SDK 源 ID。见 `E2026/outputs/q3/q3_alignment_provenance_report.md`、`provenance_sources.md`。

- E Q3-1：HEAF validation 已完成，状态 `HEAF_VALIDATION_PASSED`；冻结 P2 seed42 clean 复算通过，728 条 valid 的 exact Shapley 效率误差约 `1e-15`，按 video_id 的 design/audit 为 332/396 条，ρ=0.30 在 design 组选定并锁定。Audit 同长度 top−random margin/置信度均值为 +0.400735/+0.077718；10%–40% 删除曲线、分组 bootstrap、seed43/44 稳定性与失败例见 `E2026/outputs/q3/`。区间选择比较存在内生选择优势，不应当作外部因果证明。未训练、未访问 attachment2 test 样本、未解析附件3/4，未做媒体回看。

- E Q1/Q2：完成官方要求、方法与结果审查，见`notes/E_Q1_Q2_REVIEW_20260924.md`。Q1方法/辅助实验可收口，当前摘录交付包缺全量特征和timeline；本地测试10通过、1因缺100条timeline失败。Q2相关测试28通过、2因无CUDA跳过；六checkpoint及benchmark哈希、冻结B0张量、已有主要指标复算通过。不建议继续模型探索，优先补全量交付、正文图表和最终附件3推理入口；本轮未训练、未解封附件3。两问使用不同官方附件是正确安排，无需跨问特征对接。

- E Q2：用户正式锁定 B0-WCE 为 baseline，B5-P2 attention residual pooling 为最终主模型，停止后续 Q2 模型探索。六个 seed42/43/44 的 B0/P2 主 checkpoint 已在本地验证；attachment2 aligned-50 本地副本的 split、形状、标签与冻结审计一致。已生成最终 configs、data/benchmark/checkpoint manifests 与实验索引：`E2026/outputs/final/q2/`。冻结 benchmark 哈希匹配；attachment3 保持 SEALED，attachment2 test 未用于本阶段选模。原始本地文件在 `E2026/data/raw/`，与历史服务器 `data/raw/attachment2/` 路径不同，未移动大文件。

- E Q2：B5-N1 seed42 text-only train-stat feature-wise z-score finished. Historical N0 B0-WCE checkpoint reproduced exactly on clean + frozen 54-scenario validation. N1 trained the complete original B0 from matching seed42 initialization; best-clean and best-robust both selected epoch 1. N1 robust 0.734474 vs N0 0.740248 (Δ −0.005774); accuracy and overall score fell despite higher Neutral recall/F1. Stop normalization route; no P2 combination or additional normalization runs.
- Text scaler fit on 83,672 valid train timesteps for 768 text dimensions, epsilon 1e-6; zero/nearly-zero variance dimensions 0/0. Float32 normalized train feature means max abs 6.2e-8, nondegenerate std max deviation from 1 6.8e-8. Synthetic missing is overwritten to exact zero after normalization; unchanged benchmark SHA256.
- CPU/CUDA forward/backward, initial-state identity, N0 reproduction, and both checkpoint round-trips passed. Three scaler tests pass. The aligned pickle is monolithic and deserialized as a container, but test is not indexed into a Dataset, evaluated, or used; attachment3 not accessed.
- Outputs and formal record: `E2026/outputs/metrics/b5_n1_text_zscore_report.md`, metrics/scaler-state JSON, `experiments/exp_013_b5_n1_text_zscore/`.

- E Q2：B5-P2 attention residual pooling seed43/44 paired replication completed, reusing seed42. Seed-specific B0 checkpoints were used; B0 tensors remained frozen and passed exact post-training equality checks. Validation was clean + fixed 54 missing scenarios only; benchmark SHA256 unchanged.
- Paired robust deltas P2−B0: seed42 +0.005600, seed43 +0.003791, seed44 −0.000090; mean +0.003100 ± 0.002907 sample SD. Two of three seeds are positive, with seed44 effectively neutral. Treat as seed-sensitive candidate, not stable universal improvement.
- Results: `E2026/outputs/metrics/b5_p2_multiseed_report.md`, machine-readable `b5_p2_multiseed_summary.json`, formal record `experiments/exp_012_b5_p2_multiseed/`.
- No z-score or module combination started. Test/attachment3 excluded.

- E Q2：完成 B5-H0 只读诊断。历史 B0-WCE seed42 checkpoint 未修改；valid clean + 54场景，benchmark SHA256不变。
- Head一致性：clean/missing agreement 82.69%/81.62%；both-wrong约30.3%；reg-only-correct约5.1%/5.5%。Oracle union上限约69.6%/69.7%，仅诊断。结论：部分互补但共享错误明显，尤其Neutral不能由tau=0回归符号映射恢复。
- 输出：`E2026/outputs/metrics/b5_h0_head_diagnostic.{json,md}`、`b5_h0_predictions.{csv,jsonl}`；正式记录 `experiments/exp_011_b5_h0_head_diagnostic/`。
- 未训练模型、未修改checkpoint、未索引/评估attachment2 test、未访问attachment3；不自动进入H1。
- Q1/Q3完成情况本轮未核实；A题历史资料保留。

- 2026-09-26 E Q2 expanded baseline：完成12个公开架构适配（TFN/LMF/MFN/MulT/MISA/Self-MM/MMIM/MAG-BERT/TFR-Net/MissModal/M3S/MMIN）与B0/P2共14模型、42/43/44三seed的统一CleanSelect比较。新增27次训练完成；clean与冻结54场景各42条seedwise结果、14条汇总。P2 clean四指标排名1/1/2/2，missing排名4/1/1/2（Acc/F1/MAE/Pearson）；不声称全面最优。missing-aware训练单独标记，所有模型为aligned-50机制适配。结果、来源、checkpoint清单与QA见 `E2026/outputs/final/q2/public_baselines_expanded/`；未修改论文、未用test或附件3/4选模。
