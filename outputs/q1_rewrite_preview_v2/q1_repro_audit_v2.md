# Q1复现性核查：恢复产物证据更新

## 结论与检查边界

新包已恢复最终100样本特征、mask、行序和完整来源。本轮实际读取并核对，替代旧版“未定位最终文件”的阶段结论。核查入口verify_completion.py；机器结果qa/completion_verification.json。

包来源：D:/java录屏/q1_completion_20260926.zip。SHA256为751b3fbe1dbab84e59a0e9a4dc499daaad671a49711f94bbd45c602dba716453。包内1025个文件，bundle_sha256.csv列出的1024个文件全部重新计算hash及bytes一致（该清单自身不列入）。原始MP4/WAV不在包中，本轮未重新解码媒体；时间验证依据是包内100样本CSV和报告，不将其称为本轮重新执行逐帧解码所得。

## 实际最终文件

| 文件 | 实际键/字段 | 本轮读取结果 |
|---|---|---|
| final/aligned_features_fp32.npz | text、audio、vision | 各(100,50,768)，float32；全部finite |
| final/masks.npz | text_mask、audio_mask、vision_mask | 各(100,50)，bool |
| final/sample_ids.csv | index、sample_id | 100行、100唯一ID；index为0–99；与manifest及旧100样本汇总行序一致 |
| final/feature_manifest.json | sample_order、dtype、shape、encoder、encoder_revision、alignment_method、K、source_directory、generation_script、files | 已读取；FP32/mask/ID三个实际文件hash与manifest一致 |
| trace/source_mapping_15000.csv | 见下列实际字段 | 15000行，15000个唯一样本/窗/模态键，无缺漏 |

有效位：文本4121、语音4955、视觉5000；平均0.8242、0.9910、1.0000。全部无效位的特征为精确零向量；有效位没有全零行。本结论通过mask检查，不以零向量反推mask。

核心文件SHA256：

- aligned_features_fp32.npz：b3e814045187f6de6264bc811e4438c692b7f32ffdbdd76264435728d0b5f9dc
- masks.npz：591207484ad4bc3b6a7e0cf26dc9c7bd6c2982d8d0c80778e22c6e1341a95b65
- sample_ids.csv：3df1414bd01113c936ded30ef3a3005c02712be1c8862ab014be4f5aeece1094

NPZ内部为压缩NPY数组。manifest还记载FP16候选传输副本，它不在本包，不是本轮正文的正式FP32结果。

## 原生、对齐与来源对应

native/text、audio、vision中各100对NPY/JSON实际读取。原生行数1847/38776/3205，均768维FP32且finite。JSON records[feature_index]对应原生向量行。聚合矩阵与trace/alignment_source_npz中的100个逐样本对齐NPZ逐值相同；全部有效对齐向量又与其被选原生行逐值相同。旧汇总各样本的有效窗数、媒体hash、时长均与恢复文件对应。

来源CSV实际字段：

```text
row_index, sample_id, window_index,
window_start_seconds, window_end_seconds, modality, valid,
source_mp4_sha256, native_feature_index,
source_support_start_seconds, source_support_end_seconds,
actual_overlap_seconds, text, text_feature_source, timestamp_source,
transcript_span_json, source_words_json,
wav_sample_start, wav_sample_end,
frame_indices_json, frame_pts_seconds_json
```

逐条核对valid与mask一致，row_index与样本行序一致；有效记录的原生索引、支持区间、交叠与aligned_json相同。文本词、字符区间、时间来源及source_words、WAV范围、帧号及PTS与嵌套来源一致。无效记录保留行、valid=False、空原生索引及零交叠。来源字段分别保存，不假定它们内嵌于密集FP32矩阵。

正式文本来源为text/text_feature_sources.json，包含100样本的official_text、observed_asr_text、feature_records、内容来源及有效性信息；100条raw_asr与timeline均已恢复。报告支持同样本词序匹配、未匹配词省略及所属segment回退；无forced alignment。匹配阈值中的token coverage参考本批100条的下四分位数，报告值0.851190，不能误称预设外部阈值。新包未附整套生成脚本，README指向原工作区build_q1_text_source_manifest.py，本文不编造该脚本未读取的额外细节。所选原生文本时间回退实计7样本、21位置，与旧表一致；报告记载原始ASR回退46词，须与所选特征21位置区分。

## 时间基准及帧对应

audit/timebase_validation_100.csv实际100行，ID覆盖完整；视频流起点、音频流起点、首视频PTS、首解码音频PTS四字段全部0。源MP4/WAV路径及hash保留，WAV无独立媒体PTS，样本0对应首解码音频样本。已有无seek的提取流程因此可按共同相对零点组织。

CSV记录FFprobe/OpenCV帧数逐样本一致，合计23241；最大PTS差4.4444444480262746e-7 s，低于0.001 s阈值，无超过阈值的样本。附带报告记载逐帧不一致0。该核验为补包生成时的独立媒体检查，不冒充历史GPU版本记录或本轮重新解码。

## 模型与处理配置

正式批次为q1_features_100_final；旧q1_features_100已标为superseded。最终聚合由build_q1_final_local.py产生，来源maximum_overlap_hard，且恢复矩阵与正式来源逐值一致。q1_full.yaml为cuda、seed2026、100样本，text precomputed_contextual、max_tokens510，音频16kHz，视觉目标4fps/批8。

实际模型及revision：

- FacebookAI/roberta-base：e2da8e2f811d1448a5b465c236feacd80ffbac7b
- microsoft/wavlm-base-plus：4c66d4806a428f2e922ccfa1a962776e232d487b
- google/siglip2-base-patch16-224：75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
- Whisper large-v3-turbo：模型名已记录，精确revision未记录。

WavLM processor快照：sampling_rate16000，do_normalize=false，return_attention_mask=true，右侧padding/零值；不依据通用默认值推测归一化。SigLIP2快照：224×224缩放、resample2、1/255重标度、均值/标准差各0.5、SiglipImageProcessor；未记录的额外裁剪不猜测。模型processor与特征提取规则分开核查。

GPU实验清单记录PyTorch2.14.0+cu130和Transformers4.57.3。新包明确GPU运行Python/NumPy/OpenCV未记录；补包本机独立核对Python3.11.16/NumPy2.4.6/OpenCV5.0.0不能冒充历史GPU环境。未找到Ubuntu24.04或Python3.12.3的实际记录，因此不写入论文。cu130表示PyTorch构建版本，不独立证明历史运行时CUDA详情。此类非核心小项仅在本文件保留，不写入正文缺口说明。

## 案例与结果保护

sample -a55Q6RWvTA__3的23–25窗九条记录与现有表7一致：公共范围、原生索引、时间支持、交叠、best/life/free、三个WAV范围、315/322/337源帧及10.500/10.733/11.233显示时间均核对。表7源文件逐字节未改。表2全部100行未改，辅助实验表和七张图未改。正式工程136文件hash未变化。没有新训练、调参、特征生成或辅助实验运行。
