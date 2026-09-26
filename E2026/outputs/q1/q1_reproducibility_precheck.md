# E2026 Q1 复现性与实现一致性 PRECHECK

日期：2026-09-25。范围：本地代码、已保存的清单/案例记录、当前 Q1 论文源文件及编译 PDF。仅核查，不运行编码器、不重新提取特征、不修改论文或图件。路径均相对于 `D:/华为杯/`。

## 状态

| 检查项 | 状态 | 判定依据与边界 |
|---|---|---|
| `TEXT_IMPLEMENTATION` | `PARTIAL` | 正式提取器及参数可核对；其输入 `text_feature_sources.json` 和生成该文件的规则/代码未随当前包保存，因此官方词到 Whisper 词的最终构造无法端到端复核。不能把同文件内旧版 `WhisperRoBERTaExtractor` 的 `SequenceMatcher` 规则当成正式生成规则。 |
| `AUDIO_IMPLEMENTATION` | `PARTIAL` | WAV 命令、WavLM 输出及样本支持区间有明确代码和案例记录；实际 WavLM processor 配置/版本、音轨起点偏移的应用情况没有完整记录。 |
| `VISION_IMPLEMENTATION` | `PARTIAL` | 4 fps 的 PTS 最近帧选择、视觉输出及时间支持有代码和案例记录；SigLIP2 processor 的 resize/crop/normalize 实际配置未随包保存，不能仅凭模型名断言输入一定为 224×224。 |
| `ALIGNMENT_TIEBREAK` | `VERIFIED` | `align_to_bins_hard` 使用正交叠、最大交叠、最近区间中心、最早原生索引的词典序；没有硬对齐 epsilon。 |
| `FINAL_FEATURE_FORMAT` | `NO` | 写入脚本和旧审核记录定义了格式，但当前本地交付包没有正式 FP32/FP16 NPZ、mask NPZ、feature manifest 或逐样本原始矩阵；本轮不能实际读取并验证它们。 |
| `READ_INTERFACE` | `NO` | 有打包脚本内的 `np.load` 和完整性验证脚本，但当前没有可运行的独立最终特征读取入口；被读取的正式大文件也不在包中。 |
| `MODEL_REVISIONS` | `PARTIAL` | RoBERTa/WavLM/SigLIP2 的模型 commit 已列出；Whisper turbo revision、GPU Python/NumPy 版本及 processor 细节缺失。 |
| `CASE_TRACE` | `PARTIAL` | 已保存的 `main_case.json` 与 `main_case_trace.csv` 对关键索引/时间完全一致；原逐样本 native JSON、aligned JSON/NPZ 和 MP4 不在本地交付包，不能再次从源媒体独立验证。 |

## 1. 正式处理链与文本实现

正式配置为 `Q1_方法流程与实验结果_交付包/configs/q1_full.yaml`；入口为 `scripts/q1_full_feature_run.py`，其文本 backend 是 `precomputed_contextual`。`src/q1/extractors/text.py::PrecomputedTextSourceExtractor` 加载 `FacebookAI/roberta-base`，`AutoTokenizer.from_pretrained(..., use_fast=True, add_prefix_space=True)` 与 `AutoModel.from_pretrained(..., add_pooling_layer=False)` 使用同一 resolved model path。`model_versions.csv` 记录模型 commit `e2da8e2f811d1448a5b465c236feacd80ffbac7b`，但没有独立 tokenizer commit 字段。

提取器把一条 clip 的所有**带时间戳的 `feature_records[*].text` 片段作为一个预分词列表**传给 tokenizer，参数 `is_split_into_words=True`、`truncation=True`、`max_length=510`。取 `last_hidden_state[0]`；对同一 `word_id` 的 subword hidden vector 取算术平均，得到一条 768 维原生文本特征。特殊 token 的 `word_id=None`，不会进入任何片段的平均。超出截断长度的片段被丢弃并产生 warning。它不是逐词单独跑一次 RoBERTa，也不是把 50 个公共时间窗当成 RoBERTa token 位置。

正式提取器**读取** `outputs/q1_text_final/text_feature_sources/text_feature_sources.json`，依据记录中的 `text_feature_source` 选择官方文本或当前媒体 ASR，保留 official/ASR 文本、原文字符 span、ASR 词索引、时间来源。当前包没有这个 JSON，也未找到生成 `feature_records` 的脚本；因此官方词如何逐词匹配 Whisper、未匹配官方词如何舍弃/改用 ASR，不能从正式链确认。`src/q1/extractors/text.py` 前半部旧类的 `SequenceMatcher` 匹配不能移植为正式实现结论。`scripts/build_q1_text_final.py` 的 `tokens_score`、subsequence/coverage、decoder 证据是**样本级文本来源审计/选择**，并未构建缺失的 `feature_records` manifest。旧审计报告显示 secondary `whisper-large-v3` 在 16 条异常样本上被比较但最终入选 0，正式主时间戳来源仍为 `whisper-large-v3-turbo`。

`src/q1/timestamp_fallback.py::resolve_word_interval` 明确要求词区间有限、`0≤start<end≤duration`、相对前一个**原始词区间**两端不倒退；词与词重叠允许。无效或非单调时仅回退到有效的所属 Whisper segment 区间，segment 仍无效则 unresolved；不插值、不跨片段搬运。`scripts/build_q1_text_final.py::timestamp_quality` 统计原始 ASR fallback；旧结果 `outputs/q1_text_final/report.md` 为 46。`outputs/q1_final_local/00_manifest/q1_paper_numbers.json` 将 21 指向正式 validation report 的 `selected_text_feature_segment_fallback_count`；`scripts/build_q1_final_local.py::build_samples_and_features` 还按最终 text native records 中的 `timestamp_source == segment_fallback` 与 source audit 核对。故 46 是原始 ASR 词级回退数，21 是最终入选文本特征记录中的回退数，两者分母/处理阶段不同；由于原始 source audit/native records 缺席，本轮只能核对保存的汇总来源链，不能重新计数 21 条。

## 2. 音频实现

`scripts/prepare_q1_cpu.py::prepare_one` 使用 ffmpeg `-map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le` 将 MP4 第一音轨转为 16 kHz 单声道 PCM WAV；随后用 `wave` 核验采样率与通道数。`src/q1/extractors/audio.py::WavLMExtractor` 由 `soundfile.read(..., dtype="float32")` 读取 WAV，调用 `AutoFeatureExtractor` 和 `AutoModel`，模型为 `microsoft/wavlm-base-plus`，commit `4c66d4806a428f2e922ccfa1a962776e232d487b`。使用 `last_hidden_state[0]`，转 float32；输出 768 维由保存的验证摘要和打包脚本的维数校验支持，提取器本身没有写死 `768`。

输出 row 的样本支持由模型配置 `conv_kernel`/`conv_stride` 计算有效 hop 与 receptive field，定义 `[row*hop, min(N,row*hop+receptive))`，再除以 16000 转成相对秒数，并将终点截到 timeline duration。输出 row 数另与模型 `_get_feat_extract_output_lengths` 校验。代码没有显式额外裁剪/补齐，也没有保存 processor 的实际 normalize/padding 配置；WAV 的重采样由 ffmpeg 执行。`prepare_q1_cpu.py` 记录了原音轨 `start_time`，但 `audio.py` 把 WAV 样本零点直接当作 MP4 presentation 零点，未见使用该 offset；这应作为复现假设核实，不能据此断言现有案例时间一定错误。

已保存案例显示 row 542 的 WAV 样本 `[173440,173840)`，对应 `[10.840,10.865)` 秒，与 `main_case.json`、`main_case_trace.csv` 一致。

## 3. 视觉实现

`src/q1/extractors/vision.py::select_frames_by_pts` 用 `np.arange(0,duration,1/4)` 生成理论采样时刻，对 `prepare_q1_cpu.py::probe` 通过 ffprobe `best_effort_timestamp_time` 得到的真实帧 PTS 作 nearest-neighbor 选择；距离恰相同时取左侧/较早帧，并去掉相邻重复选择。故 VFR 按真实 PTS 选帧，不按固定帧索引间隔。实际图像再由 OpenCV 按 frame index 解码、BGR→RGB、转 PIL Image；代码未逐帧重新核对 OpenCV 与 ffprobe 的 PTS 对位。

`SigLIP2Extractor` 加载 `AutoProcessor`、`AutoModel`，模型 `google/siglip2-base-patch16-224`，commit `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`。调用 `get_image_features`；返回值有 `pooler_output` 则取它，否则取返回 tensor，转 float32。输出维数 768 由保存的验证摘要/打包脚本校验。当前本地没有正式 SigLIP2 `preprocessor_config.json` 或 processor dump；`outputs/q1_visual_encoder_ablation/model_inspection.json` 属于 **DINOv3 对照**，不能据此声称 SigLIP2 的 resize/crop/normalization 细节或 224×224 实际输入已验证。

视觉 row 的时间支持由相邻**被选中的实际 PTS 的中点**划分；首边界 0，尾边界媒体 duration。案例 row 43 对应源 frame index 322、PTS `10.733333 s`，支持区间约 `[10.6166665,10.8666665)`，与两份保存的案例记录一致。

## 4. 时间对齐与正式文件契约

`src/q1/temporal.py::align_to_bins_hard` 使用半开区间的长度公式 `max(0,min(e1,e2)-max(s1,s2))`，只保留 `overlap>0`。选中项排序键为 `(-overlap, |native_center-bin_center|, native_index)`：**最大交叠 → 最近中心 → 最早索引**。浮点并列按 Python 原始数值精确相等决定，没有 `isclose`/epsilon。`align_to_bins_method` 的默认 `epsilon=1e-8` 只用于其他加权平均路径，不能写成正式 hard path 的容差。无正交叠时输出 float32 零向量、bool `False`、空 trace；零值自身不是有效性判据。`scripts/q1_full_feature_run.py::align_sample` 将逐样本 `<sample_id>.npz` 写为 `text_values/audio_values/vision_values` 与 `text_valid/audio_valid/vision_valid`；同名 JSON 的 `bins[].modalities` 保存原生 `feature_index`、`source_interval`、`overlap_seconds`、`source`，另有 `sample_id`、源 MP4/hash、time base、`valid_length` 和全 False 的 `padding_mask`。

`scripts/build_q1_final_local.py::build_samples_and_features` 所**定义**的最终密集格式：`01_final_features/aligned_features_fp32.npz`，`np.savez_compressed`，键 `text/audio/vision`，各 `(100,50,768)` float32；`masks.npz` 键 `text_mask/audio_mask/vision_mask`，各 `(100,50)` bool；`sample_ids.csv` 的零基 `index,sample_id` 固定行序；`feature_manifest.json` 记录 sample order、dtype、shape、三编码器及 revision、alignment method、K、源目录、文件 SHA256/bytes。`aligned_features_fp16.npz` 是候选传输副本，非正式 FP32。**源索引和时间不内嵌在密集 NPZ 中**，必须另读逐样本 aligned JSON/native JSON；不应在论文虚构 `text_source_index` 等 NPZ key。

当前两份 Q1 本地交付包均不包含上述 `01_final_features` 大文件，也不含逐样本源 JSON/NPZ；README 明示大矩阵排除。`outputs/q1_final_local/07_audit/final_audit.md` 记录过去运行时 100 IDs、形状、有限值、SHA256 等均通过，但这是**既往验证记录**，不是本轮对实际文件的重新打开。`scripts/verify_q1_final_local.py::main` 是完整性检查读取逻辑；没有面向交付用户的简短 `read_q1_features.py`。当前无法执行“加载正式文件→按 ID 取三模态→读取 mask/source trace”的端到端读入，也不能重新核验 100 样本有限值。论文正式写文件结构前须以实际提交附件补齐并重跑只读核验。

## 5. 版本、案例与论文差异

版本文件：`Q1_方法流程与实验结果_交付包/outputs/q1_final_local/06_reproducibility/model_versions.csv`、`software_versions.txt`；主配置 `configs/q1_full.yaml`；打包命令 `06_reproducibility/commands.md`。记录值：RoBERTa `e2da8e2f811d1448a5b465c236feacd80ffbac7b`；WavLM `4c66d4806a428f2e922ccfa1a962776e232d487b`；SigLIP2 `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`；Whisper turbo revision `unavailable`。GPU Transformers `4.57.3`、PyTorch `2.14.0+cu130`；CPU preparation Python `3.13.9`、NumPy `2.3.5`、FFmpeg/ffprobe `8.0`。GPU Python/NumPy 未记录。仓库根 `requirements.txt` 仅有 PyYAML 约束，并非 Q1 完整环境锁文件。

两份案例记录分别为 `outputs/q1_final_local/04_case_studies/main_case.json` 与 `main_case_trace.csv`。样本 `-a55Q6RWvTA__3`：duration `22.153971 s`；bin **index 24（零基，第 25 个）** 为 `[10.63390608,11.0769855)`。文本 index 34，词 `life`，区间 `[10.9,11.12)`；语音 index 542，`[173440,173840)` WAV samples，`[10.84,10.865)` 秒；视觉 index 43，源帧 322，PTS `10.733333 s`，支持区间约 `[10.6166665,10.8666665)`。两份案例记录吻合，源 MP4 hash 一致。`Q1_论文绘图数据与LaTeX交付包/Q1_论文绘图数据与LaTeX交付包/outputs/q1_final_local/05_appendix/q1_sample_summary_100.csv` 实有 100 行、100 个唯一 ID，案例 duration 与之相同；这只验证汇总表，不代替矩阵审计。

当前论文 `paper/sections/05_q1.tex` 的待修项（本阶段**未修改**）：

1. 第 188、193 行写“第24个统一时间窗”，与零基 `index=24` 的自然语言序号不一致；应表述为“索引24的公共时间窗（第25个时间窗）”。
2. 对齐公式只给 `argmax overlap`，没有写正式代码的最近中心、再最早索引并列规则。无重叠零值/mask 语义与代码一致。
3. 正文表 `generated/table_q1_all_samples.tex` 的“时长/s”读自 `q1_sample_summary_100.csv` 的 `duration`。`scripts/prepare_q1_cpu.py::probe` 定义为 `max(ffprobe format.duration, 最后一帧 PTS + 1/fps)`，是处理采用的媒体时间轴终点；**不是单纯容器 duration，也不是声画共同有效区间**。表头/说明应明确该定义。
4. 当前图 6 PDF 内有 “RoBERTa 词位置 t”、`[CLS]`/`[SEP]` 等文本，容易把公共时间窗误读为 token 坐标；公共坐标实际是媒体等时长 `B0…B49`。现图中的 `[CLS]/[SEP]` 也不是正式 RoBERTa tokenizer 的特殊 token 名称。图 6 应按真实媒体时间、原生区间、公共 50 窗重构；图 2/3/4 由另一任务负责，本轮不碰。
5. 当前 `paper/build/main.pdf` 第 12 页已出现“问题二”标题，图 8 图注在第 13 页，确有跨章节浮动；下一阶段须约束图 8 留在 Q1 典型样本小节内。
6. 文本正文只写“RoBERTa-base 提取语义向量”，尚未交代整 clip 预分词上下文、末层、subword 均值、特殊 token 排除和 510 token 截断；语音/视觉的精确采样和来源结构也可在紧凑复现小节中交代，但 SigLIP2 processor 与正式文件读取证据须先补齐。

## 第二阶段前所需资产

请以最终实际提交位置提供/恢复 `aligned_features_fp32.npz`、`masks.npz`、`sample_ids.csv`、`feature_manifest.json`，以及对应 100 条逐样本 aligned/native trace；另需 `text_feature_sources.json` 或其生成代码/规则与 SigLIP2/WavLM processor 配置。届时仅做只读文件校验并形成最小读取示例，再修正文、图 6 与图 8。当前 PRECHECK 到此停止。
