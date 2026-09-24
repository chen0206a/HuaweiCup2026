# Q3-2 Attachment4 grounding and interface audit

日期：2026-09-24。**结论：`BLOCKED_BY_GROUNDING`；当前局部证据等级 `UNVERIFIED`。** 在解析任何 Attachment4 特征前，已将 Q3-1 协议写入并提交 `configs/final/q3_heaf.yaml`、`outputs/q3/q3_method_lock.{md,json}`（锁定提交 `3d23b9e`）。本轮只读 Attachment4 对齐版 pkl/MP4 的接口和文件级、媒体时间元数据，没有加载 P2 或生成预测。

从仓库根目录复核：`python E2026/scripts/run_q3_grounding_audit.py`。机器结果 `metrics.json` 与 `../../../outputs/q3/q3_grounding_audit.json` 一致，说明见 `../../../outputs/q3/q3_grounding_audit.md`。

20 个单样本 pkl 均具 `id/raw_text/text/audio/vision/text_bert`，无标签或 timestamp/alignment/interval/word/frame/length 字段。特征维度、有限值、`text_bert[1]` 二值有效前缀检查通过；有效长度 13–50。20 个 ID 与同名 MP4 精确一一对应，缺失、重复、歧义均为 0。对齐版 40 个文件的当前大小和 SHA256 与既有 inventory 相符。

整段 `raw_text` 可读，`text_bert` 可恢复整数 token ID 与有效长度，但无 tokenizer vocabulary、token offset 或逐词时间。20 段 MP4 在 OpenCV 可读，提供名义 FPS/帧数/估计时长与解码器报告的帧位置；**16/20 个文件的报告帧数与顺序解码帧数不一致**，名义时长尤其不能直接作为定位依据。未发现槽位到视频/音频时间的映射。两个样本的合成审计区间按 `evidence_grounding.ground_interval` 返回 `unverified`，文本片段、音频时段、帧时间全部为 null。没有用 `duration/50`、字符比例切片或 Q1 的统一窗口规则。

候选仅两条：优先寻回官方特征提取/对齐 provenance，以同一 ID、hash 和媒体核验，力争 VERIFIED；若不存在，再另行评审基于提供视频与冻结 tokenizer 的重建式对齐，独立估计误差，最多作为 APPROXIMATE。此阶段不实施候选，也不运行 Attachment4 全量 HEAF。只有证据映射通过进一步审查后才能进入最终解释推理；不得重调已锁的预测器或解释超参数。
