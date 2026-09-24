# Q3-2.5 Alignment Provenance Recovery

日期：2026-09-24。状态：`BLOCKED_BY_GROUNDING`。公开来源审计及 20 个 Attachment4 aligned sample 的无标签 tokenizer 重放说明见 `../../../outputs/q3/q3_alignment_provenance_report.md`；逐代码出处见 `../../../outputs/q3/provenance_sources.md`；机器统计在本目录 `metrics.json`。

Attachment4 的 keys、50 长度和 `(768,74,35)` shape 与 MMSA/Self-MM 的 MOSEI `aligned_50.pkl` 高度吻合，但 sample IDs 是 `01`–`20`，对象是 20 个单样本 pickle。未找到把这些文件 hash/ID 绑定到公开整包、生成 commit 或原始 `video_id$_$clip_id` 的证据。结论是候选格式吻合，provenance 未确认。

固定 `bert-base-uncased` fast tokenizer revision `86b5e0934494bd15c9632b12f734a8a67f723594`（tokenizer JSON SHA256 `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98`），lowercase、添加 `[CLS]/[SEP]`、右截断到 50、右侧补 ID 0。逐样本 input IDs、attention mask、token type IDs（含 padding）20/20 精确匹配。左截断 18/20、无 special token 0/20、case-sensitive 0/20、无截断且不补齐 0/20（长度/掩码不符）。`text_bert` token index 到 raw_text offset 可验证，特殊 token 和 padding 被排除。

P2 输入的是 `text[50,768]`。公开候选实现不能确认 Attachment4 的 text feature row index 与 `text_bert` index 同义；这点被明确标记为 `FEATURE_ROW_MAPPING_UNVERIFIED`。因此 token span 只作为核验诊断，`evidence_grounding` 的 P2 `text_fragment` 保持 null。Audio 74 / vision 35 与 COVAREP/FACET 常见 family 相符，但没有附件级 extractor provenance、原始来源 ID 或 per-slot intervals；两者保持 `UNVERIFIED`。

CMU-MultimodalSDK 的 CSD 层保存 feature intervals，MMSA-FET 展示了 word interval 平均后按 BERT `word_ids` 扩展并保留 align 记录的可行路径。这些是候选来源而不是 Attachment4 的来源证明。当前 generic IDs 无法 join 到 MOSEI `video_id$_$clip_id`；本轮未下载大体量 feature/label 文件，仅对仓库代码和公开 schema 做审计。

Smoke checks 在两个样本上调用集成后的 `ground_interval()`；token 映射返回 verified，而 P2 evidence 字段继续 null。没有加载 predictor、使用标签、运行全量解释或改动 HEAF。

复现：从仓库根目录运行 `python E2026/scripts/run_q3_provenance_audit.py`。运行依赖 Python packages `tokenizers` 和 `huggingface_hub`，仅按固定 commit 拉取约 1 MB tokenizer artifact；Attachment4 文件应已登记在现存 inventory。

后续若要解除阻塞，需要官方 source-key/hash manifest 或经单独评审的、能证明所用 feature rows 对应关系的 provenance。没有实现 A/V 重建，也没有重选解释算法。
