# Q3 Attachment4 原始证据映射可行性核查（PRECHECK）

**结论（2026-09-25）：**

```text
TEXT_RAW_MAPPING = VERIFIED
AUDIO_RAW_MAPPING = UNVERIFIED
VISION_RAW_MAPPING = UNVERIFIED
SAMPLE02_VISUAL_MAPPING = NO
```

本次只读核查本地 Attachment4 的对齐特征、未对齐特征、媒体文件清单及锁定的 Q3 解释结果。没有重训、修改 HEAF、读取标签、改论文、按槽位推断秒数、解码关键帧或导出音频。`UNVERIFIED` 指**特征槽位到原始媒体时间/帧**尚无可靠映射；下述特征数组之间的逐行数值身份是更浅一层的发现，不能替代原始媒体定位。

## A. 本地资产与样本身份

- 来源目录：项目外同一工作区的官方 `附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/`，分 `对齐版本/` 和 `未对齐版本/`。定位依据为 [`attachment4_inventory.json`](../../data/manifests/q3/attachment4_inventory.json) 中的绝对路径。
- 两个版本各有 `01`–`20` 的 **20 个 pkl + 20 个 MP4**；每个版本 pkl ID 与同名 MP4 一一对应，两个版本的 MP4 按 ID 的 SHA256 也 20/20 相同。清单内 80 个文件现均存在，重新计算的 SHA256 和大小 80/80 匹配。
- 对齐 pkl 的 20/20 字段集合相同：`audio, id, raw_text, text, text_bert, vision`。未对齐 pkl 的 20/20 字段集合相同，并额外包含 `audio_lengths, vision_lengths`。所有样本在两个版本中的 `id`、`text`、`text_bert`、`raw_text` 一致；没有标签字段。
- 对齐版 sample `02`：`text_bert` 为 `[3,50]`，有效前缀 21；`audio` 为 `[50,74]`，`vision` 为 `[50,35]`。未对齐版同一样本的音频/视觉数组分别是 `[500,74]`、`[500,35]`，声明长度分别为 66、50。`02.mp4` 在两个版本中字节哈希相同：`6b651e908aefd86959f85b78f945f7d5a5d9e767eb4c095926e1df2ed9629ff9`。
- 官方附件目录除上述 pkl/MP4 外无代码、CSV、JSON、音频单文件、时间戳或对齐记录。对 `D:/华为杯/E2026/`、`paper/`、`D:/华为杯/`（包含项目历史输出与 `.cache`）的文件名和相关代码检索，未找到可把 Attachment4 槽位绑定到 raw MP4 的 `timestamp/PTS/frame_index/audio_time/source_map/align_result` 文件或官方特征生成脚本。已有图件缓存中的视频截图清单明确标为 `context_only`，不代表关键槽位定位，见 [`figure9_video_frame_manifest.json`](../final/q3/figures/data/figure9_video_frame_manifest.json)。

## B. Text：VERIFIED

现有 [`q3_text_row_identity_report.md`](q3_text_row_identity_report.md) 已验证 `text[i] ↔ text_bert token slot i`：20 个样本、604 个有效行中，同索引 cosine argmax 为 604/604，均值 cosine 为 0.999999994。原文定位沿 [`text_grounding.py`](../../src/q3/text_grounding.py) 与固定 tokenizer offsets 实现；本次没有重新编码文本。

一致性复核：锁定的 [`attachment4_explanations.jsonl`](final/attachment4_explanations.jsonl) 中 19 个文本主导案例，其 `mapping_note` 记录的 `raw_text_character_span` 均能从同 ID 对齐 pkl 的 `raw_text` 精确截取出已保存的 `text_fragment`（19/19）。sample `14` 的关键文本槽位为半开区间 `[1,12)`，原文字符区间 `[0,38)`，片段为 `He is the co-founder of Rossen and Vet`，与锁定解释一致。sample `02` 的关键模态是 vision，其文本原文定位不用于替代视觉时间定位。

## C. Audio：原始时间 UNVERIFIED

未对齐版保存 500 行、74 维音频特征及 `audio_lengths`，但没有每行的 waveform sample、起止时间、帧率、采样率、音轨偏移或提取器版本。`74-D COVAREP` 与常见 MOSEI 配置相符，**附件级提取器身份仍未证实**。

对齐/未对齐数组逐元素比对的新发现：20 个样本的 **564/564 个非零 aligned audio 行**，各能在同一样本 `[500,74]` 未对齐数组中找到唯一、完全相等的行；源行索引随 aligned 槽位非递减。sample `02` 的槽位 1–6（0-based）分别匹配未对齐音频行 52–57。这验证了已保存特征值之间的行身份，**没有验证这些源行对应哪个音频时段**；重复使用同一源行也可能存在，不能把单个槽位解释成单一时间点。

因此目前不存在 `aligned slot → unaligned row → waveform sample/time interval` 的完整链。没有经过核验的时间区间，故不输出 `q3_audio_slot_mapping.csv` 或 `key_audio_segment.wav`。

## D. Vision：原始帧/PTS UNVERIFIED

未对齐版保存 500 行、35 维视觉特征及 `vision_lengths`，但没有源 MP4 frame index、PTS、抽帧规则、帧特征提取器版本、聚合窗口或插值记录。`35-D FACET` 是基于常见 MOSEI 特征维度的候选解释，**不是这些文件的已验证来源**。

20 个样本的 **534/534 个非零 aligned vision 行**，各能在对应完整 `[500,35]` 未对齐数组中找到唯一、完全相等的行；源行索引非递减。该观察表明这些已保存的非零输出行在数值上是未对齐特征行的复制，不能由此断定生成算法是固定帧采样、word-level 对齐、最近邻、重复填充或其他机制，也不能给源行赋予 MP4 帧号。

特别是未对齐 `vision_lengths` 不可直接当作有效帧数使用：sample `13` 声明长度 1，但完整数组最后一个非零行的结束索引是 18；sample `16` 声明长度 1，最后一个非零行的结束索引是 44。sample `13` 的 aligned vision 有效前缀全零，sample `16` 却有 11 个非零 aligned vision 行，均可在 500 行完整数组中精确匹配。长度字段的含义或质量需要原生成方说明。

### sample 02 的关键区间

锁定解释卡中 classification primary modality 为 `vision`，关键区间是 **0-based 半开 `[1,7)`，即槽位 1–6**；不能写作“第 1–7 个槽位”。sample `02` 的这六个视觉向量恰好逐元素匹配未对齐视觉特征行 36–41：

| aligned slot（0-based） | 匹配的未对齐 vision 行（0-based） | 匹配的未对齐 audio 行（0-based） | 原 MP4 frame/PTS |
|---:|---:|---:|---|
| 1 | 36 | 52 | 未核验 |
| 2 | 37 | 53 | 未核验 |
| 3 | 38 | 54 | 未核验 |
| 4 | 39 | 55 | 未核验 |
| 5 | 40 | 56 | 未核验 |
| 6 | 41 | 57 | 未核验 |

sample `02` 的有效前缀第 0 与第 20 槽在音频和视觉均为零向量。表中源行是**未对齐数值特征的行索引，不是 MP4 帧号或秒数**。缺少 `unaligned vision row → MP4 frame index/PTS` 的映射层；所以 `SAMPLE02_VISUAL_MAPPING = NO`，不生成 `sample02_visual_slot_mapping.csv` 或 `sample02_slot01.png` 等所谓关键帧文件。

已有上下文截图由 [`figure9_video_frame_manifest.json`](../final/q3/figures/data/figure9_video_frame_manifest.json) 的固定相对位置规则产生；其 sample `02` MP4 报告帧数 108、实际解码帧数 101，两者不同。这些截图不能倒推槽位到帧的关系。

## E. 已有预处理实现与 50 槽生成机制

项目内有 [`run_q3_provenance_audit.py`](../../scripts/run_q3_provenance_audit.py)、[`attachment4_audit.py`](../../src/q3/attachment4_audit.py) 和 [`provenance_sources.md`](provenance_sources.md)，它们执行接口/来源审计，**不是生成 Attachment4 特征的脚本**。Q1 的提取器服务附件1，与本题 Attachment4 的 74/35 维来源不同，不能套用。

现有 [`provenance_sources.md`](provenance_sources.md) 已按固定版本检查公开候选：MMSA 的 `data_loader.py` 读取现成 aligned pickle；Self-MM 的 `data/DataPre.py` 是通用生成例程，特征维度不符；MMSA-FET 的 `dataset.py:extract_align`、`aligner/default.py` 使用词时间区间并保留 `align` 记录，但其示例特征与本附件不同；CMU-MultimodalSDK 的 `dataset.py:align` 需要具名 timestamped CSD 来源序列。没有任何一个来源与本地 40 个 Attachment4 pkl 的哈希、原始源 ID、生成配置或版本建立对应。

目前可验证的 50 槽事实仅包括：`text_bert` 的 50 槽 token/padding 布局、文本行身份、A/V 非零 aligned 行到未对齐特征行的精确数值对应。**不能确认** A/V 的 50 槽由何种官方抽帧、词级对齐、截断、最近邻、池化或插值规则生成。即便存在通用公开对齐算法，也缺少把这些特定 MP4 和特征哈希接入该算法的生成记录，不能可重复地重建原始媒体时间映射。

## F. 状态、最大可验证粒度与停止点

| 模态 | 已验证的最深层级 | 缺失的下一层 | 原始证据状态 |
|---|---|---|---|
| Text | feature slot → BERT token → raw text character span | 无（本次范围内） | `VERIFIED` |
| Audio | 非零 aligned slot → 唯一 unaligned feature row | unaligned row → waveform sample/time interval | `UNVERIFIED` |
| Vision | 非零 aligned slot → 唯一 unaligned feature row | unaligned row → MP4 frame index/PTS；sample 13 的零向量亦无数值行身份 | `UNVERIFIED` |

本地 PRECHECK 未找到可验证的原始时间链，故没有进入视觉帧/PTS 解码映射、音频切段或三模态案例包阶段。若后续取得与这些文件哈希关联的官方提取配置、逐源行时间戳/PTS、源视频 ID 和 50 槽选择规则，可沿此处已核验的 aligned→unaligned 数值对应继续验证；在此之前保持原始时间字段为空。

**复核方法：**对 [`attachment4_inventory.json`](../../data/manifests/q3/attachment4_inventory.json) 的 80 个文件重新计算 SHA256；逐个读取两版同 ID pkl，确认字段、形状和 `raw_text/text/text_bert` 一致；对有效前缀内每个非零 aligned A/V 行，在同 ID 的完整 500 行未对齐数组中执行逐元素精确相等比较，要求命中数恰为 1 并记录 0-based 行索引；检查同 ID 锁定解释卡的 sample `02` 区间及 19 个文本案例的原文字符切片。未使用标签、模型推理或按媒体时长换算槽位。
