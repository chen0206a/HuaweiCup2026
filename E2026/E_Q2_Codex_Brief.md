# 2026 华为杯 E 题 — Q2 Codex 独立开发 Brief

> 角色：你只负责 **问题 2：局部模态信息缺失条件下的鲁棒多模态情感预测**。  
> 主战场是附件 2 + 附件 3。  
> 第一原则：**先建立可靠 baseline 和缺失模拟，再逐层加入 reconstruction / reliability / distillation。不要一开始堆全套论文模块。**

---

## 0. 最高优先级：先读赛题原文

若项目中存在：

- `复杂场景下多模态情感识别的数学建模与算法设计.docx`

请先完整阅读。

任何论文、开源模型、已有代码与赛题冲突时，以赛题原文为准。

---

## 1. Q2 硬性赛题要求

### 1.1 任务

针对 **局部时段模态信息缺失**，构建鲁棒多模态情感预测模型。

这里的缺失是：

> 文本、语音、视觉中一个或多个模态的 **部分连续时间段不可用**，并不是整个模态永久不存在。

模型必须稳定输出：

- 情感极性：Negative / Neutral / Positive；
- 情感强度：连续值。

还必须分析：

- 缺失模态类型；
- 缺失位置；
- 缺失时长 / 缺失率；

对性能的影响规律。

### 1.2 数据

训练与验证：

- 附件 2；
- 优先第一阶段锁定 `aligned_50.pkl`。

附件 2 标准 shape：

```text
text   : (N, 50, 768)
audio  : (N, 50, 74)
vision : (N, 50, 35)
```

`unaligned_50.pkl`：

```text
text   : (N, 50, 768)
audio  : (N, 500, 74)
vision : (N, 500, 35)
```

本项目第一阶段使用 `aligned_50.pkl`，除非后续实验明确证明 unaligned 值得切换。

**一旦选择 aligned 或 unaligned，训练、验证、附件 3 专项测试必须保持同一版本和同一输入接口。**

专项最终推理：

- 附件 3；
- 无标签；
- 特征中存在随机连续全零区间；
- 只能用于最终专项推理和缺失模式结构分析，不能利用不存在的测试标签做调参。

### 1.3 指标

分类：

- Accuracy；
- F1。

回归：

- MAE；
- Pearson correlation。

统一实现并固定口径。

### 1.4 数据使用红线

- 所有模型训练、微调、参数优化、阈值选择只能使用赛题提供的 CMU-MOSEI 系列数据；
- 不得额外引入其他情感数据集；
- 可以使用公开预训练模型、开源工具、开源框架，但必须记录版本和核心参数。

---

## 2. 当前技术路线

目标路线：

```text
aligned_50
    ↓
Modality Projection
    ↓
Temporal Encoder
    ↓
Continuous Block Missing Simulation
    ↓
Missing Feature Reconstruction
    ↓
Reconstruction / Representation Reliability
    ↓
Temporal Dynamic Fusion
    ↓
Classification + Regression
```

工程必须按渐进式 baseline 开发：

```text
B0: Masked Mean Pool + MLP
B1: Temporal Encoder + Basic Fusion
B2: B1 + Continuous Block Masking
B3: B2 + Missing Feature Reconstruction
B4: B3 + Temporal Reliability Dynamic Fusion
B5: optional Distillation / Factorization enhancement
```

每增加一个模块都必须有独立 ablation。

---

## 3. 当前没有 GPU 时先做什么

### 3.1 Data Audit

必须先确认真实文件而不是硬编码题面：

- 顶层 keys；
- train / valid / test 数量；
- 字段 shape / dtype；
- classification labels 编码；
- regression labels 分布；
- ID 唯一性和 split 泄漏；
- NaN / Inf；
- padding；
- 有效长度；
- 全零 timestep；
- aligned / unaligned ID 对应关系。

输出：

- `outputs/data_audit.md`
- `outputs/data_audit.json`

### 3.2 附件 3 Missing Pattern Audit

自动识别：

```text
sample_id
modality
start
end
length
ratio
interval_count
early/middle/late
single/multi-modality
touch_padding?
```

输出：

- `outputs/missing_intervals.csv`
- `outputs/missing_summary.csv`
- `outputs/missing_pattern_report.md`

注意：

- 不要把 padding 当 missing；
- 不要读取或猜测试标签；
- 如果能观察到缺失率/位置分布，可以作为 **mask generator 的无标签结构先验**，必须在报告中如实说明。

### 3.3 Preprocess

- scaler 只允许从 train split 计算；
- valid/test/附件3统一使用 train scaler；
- padding/missing zero 不允许错误计入统计；
- 原始 pkl 只读。

### 3.4 B0 + CPU Smoke Test

B0：

```text
Text  -> masked mean
Audio -> masked mean
Vision-> masked mean
        ↓
      concat
        ↓
     small MLP
      ↙    ↘
  3-class  regression
```

loss：

```text
CE + λ * SmoothL1
```

CPU smoke test 必须确认：

- forward；
- backward；
- checkpoint save/load；
- metrics；
- 无 NaN；
- 32 样本 overfit 测试可选但强烈建议。

---

## 4. Continuous Block Masking

这是 Q2 最贴题的第一核心。

接口建议：

```python
masked_features, availability_mask, metadata = apply_block_mask(...)
```

必须支持：

- 单模态局部连续缺失；
- 双模态局部连续缺失；
- 指定 missing ratio；
- 指定 start / length；
- uniform random；
- empirical distribution（根据附件3无标签结构统计）；
- mask 只发生在 valid region。

训练时必须保留被 mask 前的完整 feature 作为 reconstruction / distillation target。

---

## 5. 主模型建议

### 5.1 Modality Projection

统一到：

```text
hidden_dim = 128  # 第一版
```

例如：

```text
Text 768 -> 128
Audio 74 -> 128
Vision 35 -> 128
```

### 5.2 Temporal Encoder

第一版保持轻量：

```text
2 层 Transformer Encoder
4 heads
hidden=128
dropout=0.1~0.2
```

不要一开始上大模型。

### 5.3 双任务 Head

共享 fused representation：

```text
classification head -> 3 classes
regression head     -> scalar intensity
```

第一版 loss：

```math
L_task = L_CE + λ_reg L_SmoothL1
```

---

## 6. Missing Feature Reconstruction

目标不是恢复波形/视频，而是恢复 latent feature。

例：Audio 局部缺失：

```math
\hat H_A = R_A(H_T, H_V, M_A)
```

只在人工 mask 位置计算：

```math
L_rec =
\frac{1}{|\Omega|}
\sum_{(m,t)\in\Omega}
SmoothL1(\hat h_{m,t}, h^{full}_{m,t})
```

第一版禁止上 diffusion / flow。

先证明简单 latent reconstruction 有收益。

---

## 7. Temporal Reliability Gate

这是当前 Q2 最关键的创新方向。

目标：不是“缺失就一律不用”，而是：

> **观测可靠性 + 重建可靠性 + 当前时间位置的重要性共同决定融合权重。**

定义 availability：

```math
M_{m,t}\in\{0,1\}
```

人工 mask 阶段已知真实 feature，可以计算 reconstruction error：

```math
e_{m,t}
=
\|\hat h_{m,t}-h^{full}_{m,t}\|
```

可生成 reliability target：

```math
r^{target}_{m,t}
=
\exp(-\tau e_{m,t})
```

再学习：

```math
r_{m,t}=f(h_{m,t},\hat h_{m,t},M_{m,t},...)
```

融合权重：

```math
\alpha_{m,t}
=
Softmax_m(score_{m,t} + \log(r_{m,t}+\epsilon))
```

最终：

```math
z_t=\sum_m \alpha_{m,t}h_{m,t}^{effective}
```

其中观测位置优先使用真实 feature，缺失位置可使用 reconstructed feature，并由 reliability 降权。

---

## 8. 重点参考论文与可借模块

### 8.1 DEAR — 第一优先级

本地若存在：

- `2026.findings-acl.1517.pdf`

标题：

**DEAR: Distributional Error-Aware Reliability for Robust Multimodal Sentiment Analysis with Missing Modalities**

核心可借：

- HDCR：不要只做 point-wise MSE，要考虑恢复表示的分布偏移；
- SURE / transfer-difficulty / reliability estimation；
- reliability-driven gate；
- synergistic + robust dual-stream 思想。

我们不直接照搬：

- 它的缺失设置和本题“连续局部时间段缺失”并不完全相同；
- 我们要把 reliability 从 sample/modality 粒度推进到 **time-step 粒度**。

目标差异：

```text
DEAR: sample/modality reliability
我们: temporal local reliability r[m,t]
```

### 8.2 CMAD — 第二优先级

本地：

- `Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf`

代码：

- https://github.com/YetZzzzzz/CMAD

可借：

```text
Complete Teacher
→ Incomplete Student
```

以及：

- CAFD：feature + cross-sample correlation distillation；
- MAR：不同 modality combination 难度自适应加权。

四天内先做 `CMAD-lite`：

```math
L_KD = ||z_student - z_teacher||^2
```

若有效再上 correlation alignment。

### 8.3 EMOE — 第三优先级

本地：

- `Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.pdf`

官方代码：

- https://github.com/fuyyyyy/EMOE

可借：

- Mixture of Modality Experts；
- Router Network；
- sample-specific modality importance；
- router entropy balance；
- Unimodal Distillation。

我们的改造重点：

```text
EMOE: α_m
我们: α_{m,t}
```

从 sample-level 扩展到 temporal local weighting。

### 8.4 FUSE-Net — 作为增强项，不阻塞主线

论文：

**Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis, CVPR 2026**

当前官方 PDF 下载可能不可用，因此不要依赖未取得的原文实现全部细节。

已经确认、可借的思想：

```text
H_m
→ shared / specific / noise
```

HMF 主要约束：

- cross-modal shared contrast；
- shared/specific 应保留情感预测能力；
- noise 应尽量不携带情感信息；
- 可选 dual consistency。

MRC：

- 是“分解后重建原模态 representation 以防过度解耦”，
- **不是本题的 missing-modality reconstruction**。

MDF：

```text
sample importance
× factor prior
× branch attention
→ dynamic fusion
```

四天建议只做 `FUSE-lite`：

```text
shared/specific/noise
+ contrastive/info loss
```

如果 B4 已稳定再加。

### 8.5 简单 Transformer Early Fusion

若本地存在：

- `2505.06110v2.pdf`

只用作 B0/B1 背景参考，不把其 97%+ Acc7 结果视为可信 benchmark。

---

## 9. 最终组合路线（有时间再做）

推荐按增益顺序组合：

```text
B1 Temporal
↓
B2 Block Mask
↓
B3 Reconstruction
↓
B4 Reliability Gate
↓
B4 + CMAD-lite KD
↓
B4 + EMOE-style router balancing
↓
optional FUSE-lite factorization
```

不要一次性全开。

---

## 10. 必做实验

### 10.1 Baseline / Ablation

至少：

```text
B0
B1
B2
B3
B4
```

如加入 KD / factorization，单独加行。

### 10.2 Missing Robustness

至少分析：

缺失模态：

```text
T
A
V
T+A
T+V
A+V
```

缺失位置：

```text
early
middle
late
```

缺失比例：

```text
10%
30%
50%
```

若附件 3 的实际经验分布明显不同，再补一组 empirical profile。

### 10.3 记录

每个实验保存：

```text
config
seed
Accuracy
F1
MAE
Pearson
training time
best epoch
checkpoint
```

建议至少对关键模型多 seed 复跑。

---

## 11. 最终提交要求

论文 Q2 至少要有：

- 建模原理；
- 网络结构；
- loss；
- 训练方案；
- 关键参数；
- 缺失模态类型/缺失率/位置/时长规律分析；
- 消融实验；
- 验证集基础性能；
- 可视化；
- 错误归因；
- 附件3全量预测结果。

附件：

- 核心代码；
- 配置；
- 参数文件；
- 环境说明；
- 附件3预测 CSV；
- 总附件与 Q1/Q3 合计不超过竞赛要求，注意小模型和可复现下载说明。

---

## 12. 验收标准

进入正式 GPU 大规模实验前必须满足：

- [ ] 数据字段真实结构已确认；
- [ ] split 无泄漏；
- [ ] scaler 只来自 train；
- [ ] padding / missing 可区分；
- [ ] 附件3 missing pattern 已审计；
- [ ] B0 CPU smoke test 通过；
- [ ] block mask 单元测试通过；
- [ ] metrics 口径固定；
- [ ] B1 代码 forward/backward 正常；
- [ ] 每个新增模块可独立开关用于 ablation。

---

## 13. 工作纪律

不要：

- 引入额外情感训练数据；
- 用附件3不存在的标签调参；
- 一上来实现 diffusion / hypergraph / flow；
- 同时改五个模块导致无法消融；
- 为“新”而牺牲可复现和稳定性；
- 修改赛题原始数据。

每次提交阶段结果时汇报：

1. 新增/修改文件；
2. 运行命令；
3. 指标；
4. 失败实验；
5. ablation 结论；
6. 下一步只建议 1~2 个最值得做的增量。
