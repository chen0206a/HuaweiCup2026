# E2026 Q3-0：HEAF 方法设计与接口审计

**状态：设计版，2026-09-24。结论：`READY_FOR_Q3_EXPERIMENT`。** 此结论只批准在 Attachment2 validation 上实现、检验解释协议；不代表 Attachment4 已解封或原始证据定位已通过。HEAF = Hierarchical Evidence Attribution Framework（层级式多模态证据归因框架）。Q1/Q2 的模型、历史结果和验证协议保持锁定。

## 1. 任务、依据与五个必答问题

官方 E 题问题3要求同时给出情感极性、情感强度、模态作用程度、主要参考模态和可回看到原始文本、语音或视觉的局部证据。附件4是无标签专项集；训练、验证、专项测试必须使用同一特征版本。本方案固定 Q2 的 B5-P2（B0-WCE + Lightweight Attention Residual Pooling），使用 Attachment2 `aligned_50.pkl` 的 50×(768/74/35) 特征接口。

| 问题 | HEAF 输出 |
|---|---|
| 1. 模型预测了什么？ | 冻结 P2 的三分类 logits、预测类别、该类 softmax 置信度、连续强度预测；类别 0/1/2 = Negative/Neutral/Positive。 |
| 2. T/A/V 各贡献多少？ | 对固定预测类别的 log-odds 与回归值分别计算 8 联盟 exact Shapley，保留有符号原值和绝对值。 |
| 3. 模态协同或冗余？ | 三对模态的 Shapley pair interaction，正值为对该固定目标的协同，负值为冗余或相互抑制；同时保存两个第三模态上下文的原始差分。 |
| 4. 哪个时间片段最影响预测？ | 对主要参考模态在有效前缀内逐起点做连续窗置零，记录类别 logit、log-odds、置信度和强度变化；给出 50 槽位曲线与最重要区间。 |
| 5. 解释忠实吗？ | 在 valid 上比较最重要区间删除与同长度随机区间；另做 top 10/20/30/40% 删除曲线和等量随机删除，报告分类 logit/置信度下降及强度变化，不从注意力权重推断忠实性。 |

以上是**冻结模型对输入置零的归因**，不是对现实情绪生成机制的因果推断，也不保证干预后的样本仍处于自然数据分布。

## 2. 审计来源与接口结论（A、B、C）

检查范围：官方 E 题 `.docx`；`outputs/final/q2/q2_model_lock.md`、`configs/final/q2_b5_p2.yaml`、`outputs/final/q2/q2_checkpoint_manifest.json`；`src/models/{pooling_residual,baseline}.py`、`src/data/{dataset,block_mask,preprocess}.py`；根目录 `notes/E_Q1_Q2_REVIEW_20260924.md` 和配套审计 JSON；`data/manifests/q3/attachment4_inventory.json`。未找到另一个以 Astra 命名的 Q1/Q2 报告。Attachment4 manifest 仅有文件名、大小、哈希和路径：20 对齐 pkl、20 对齐视频、20 未对齐 pkl、20 未对齐视频；本轮未读取其中任何特征或视频内容。

**A. P2 可直接作为冻结预测函数，但当前代码尚无 Q3 解释入口。** `B5PoolingResidual.forward(batch)` 计算三模态有效位置均值、各自 attention pool，经同一投影与残差系数融合，返回 `classification_logits[B,3]` 和 `regression[B]`。前向不读取 availability/native-zero；只使用 `text/audio/vision/padding_mask`。Q3 实验拟采用锁定 Q2 三 seed 中既有 best-robust 分数最高的 seed42 checkpoint `outputs/checkpoints/b5_pooling_p2_best_robust_score.pt`（manifest SHA256 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`）；三 seed 平均指标不构成 ensemble 定义。加载时构造 `mode=mean_attention`，核对 state_dict，调用 `eval()`、禁用梯度。Q2 描述性 final YAML 不能直接交给旧训练入口；另写只读推理配置与适配器。

**B. Exact Shapley 可行。** 三模态共 2³=8 个联盟，每个样本对每个固定 checkpoint 只需 8 次预测，两个任务共用同一批输出；不需要 Monte Carlo 或训练。`forward` 没有 modality mask 开关，但输入包装器可复制张量后只将被排除模态的**有效位置**置为 0，不能改 `padding_mask`。

**C. Temporal occlusion 可行但只定位特征槽位。** 50 是最大长度，实际有效前缀长度 L 由 `text_bert[:,1,:]==1` 得到。仅在 `[0,L)` 内滑动，保留 mask；窗口内特征置零。`src/data/block_mask.py` 已有连续置零和 mask 分离的先例，但其 `apply_blocks` 以 `rho` 和完整 availability 为前提，Q3 应做独立、可批量枚举任意 `[start,end)` 的无损副本接口。P2 对有效行的置换不敏感，特征本身可能带上下文，因此局部置零变化只说明该槽位特征对输出的影响，不能称严格局部时间因果效应。

### Masking / baseline 细则

原始样本记作 `x=(x_T,x_A,x_V)`，共享布尔 `p_t` 仅标识真实有效位置。对联盟 S，`x_m^S=x_m` 若 `m∈S`，否则 `x_m^S=0`；`p` 在 8 个联盟中完全相同。零输入与 Q2 的人工缺失值相容，且 `normalization=none`，无后续 scaler 把零移动成非零。注意均值和 attention 仍包含这些**有效的零**：即使模态全零，其 projection bias、LayerNorm、attention scorer bias 和融合层也可产生非零表示，因此 `f(∅)` 不能被当作数值 0。零向量本就可能是原生观测；另存干预掩码，不能从数值零反推缺失。禁止把 masked modality 的所有位置置为 `padding=False`，否则与训练时的局部缺失和 P2 softmax 语义不一致，且全空 mask 会报错。

整模态置零超出 Q2 主要训练/验证的局部缺失范围，可能产生分布外响应。下一阶段只在 valid 上检查输出有限、稳定性、原生全零视觉样本及不可识别的零效应；对此不作真实因果或自然缺失性能声称。附件4无标签，现有 `Aligned50Dataset` 不能直接用于它；新适配器须独立校验 ID、维度、有限值、padding 来源与前缀性，禁止从全零行猜有效长度。

## 3. 模态精确归因与交互

定义 `F(S)=P2(x^S,p)`，先用完整联盟 TAV 固定 `c*=argmax_c z_c(TAV)`，所有联盟**保持同一个 c***。主分类目标选 log-odds 型 logit margin：

`v_cls(S)=z_{c*}(S)-logsumexp_{j≠c*} z_j(S)=log[p_{c*}(S)/(1-p_{c*}(S))]`。

它保留 logit 尺度，避免概率饱和与任意共同 logit 平移对 raw logit 归因的影响。另存 `z_{c*}(S)` 便于审阅题目要求的 class logit change，并存 `p_{c*}(S)` 计算 confidence drop；置信度始终指**原完整输入预测的类别**，不在遮挡后改选类别。回归目标为 `v_reg(S)=r(S)`，基线是模型在**同一 padding_mask、三模态全零**时的 `r(∅)`，不采用标注均值或强度 0。两套 φ 各有自身单位，不能直接相加比较。

对任意目标 `v` 和模态 m，令其余两个模态为 a,b：

`φ_m = (v({m})-v(∅))/3 + [v({m,a})-v({a})+v({m,b})-v({b})]/6 + [v(TAV)-v({a,b})]/3`。

计算结果必须满足效率恒等式 `Σ_m φ_m = v(TAV)-v(∅)`（浮点容差内）。报告 `phi_text/audio/vision` 有符号数值；绝对值只用于尺度展示，不叫贡献概率。分类主模态：若有正的 `φ_cls`，取最大正值；若全非正，取 `|φ_cls|` 最大者并标记 `selection_basis=absolute_negative_fallback`、`supports_predicted_class=false`；完全并列按 text→audio→vision。回归另给 `φ_reg`，不默默混入主模态选择。

对配对 (i,j)、余下模态 k，先定义上下文差分 `Δ_ij(S)=v(S∪{i,j})-v(S∪{i})-v(S∪{j})+v(S)`，其中 `S∈{∅,{k}}`。三模态 pair Shapley interaction 定为 `I_ij=[Δ_ij(∅)+Δ_ij({k})]/2`，分别对分类 margin 与回归值计算。`I>0` 表示对所选标量目标的正协同，`I<0` 表示负交互（可描述为冗余/抑制，**单靠符号不能区分两者机制**）；接近 0 只表示平均二阶作用小。保存两个 `Δ`，避免上下文抵消被均值掩盖。P2 的 `gamma` 和 attention weight 都不是跨模态交互指标。

## 4. 时间级解释与忠实性（D）

先固定分类主模态 m*。选用**按有效长度 L 的比例窗** `w=max(1,round(ρL))`，候选 ρ={0.10,0.20,0.30} 仅在 Attachment2 valid 的 Q3 设计子集比较；stride 固定 1，起点为 `0…L-w`，末端自然覆盖。选定 ρ、随机对照次数和选择规则必须在附件4解封前写入配置；不要查看附件4长度分布后调整。对每窗 W，在原**完整输入**只把 m* 的 W 置零，计算：

- `Δz_W=z_{c*}(TAV)-z_{c*}(TAV\W)`（class logit drop）；
- `Δmargin_W=v_cls(TAV)-v_cls(TAV\W)`，作为主要支持强度；
- `Δp_W=p_{c*}(TAV)-p_{c*}(TAV\W)`（固定类别 confidence drop）；
- `Δr_W=r(TAV)-r(TAV\W)` 及 `|Δr_W|`（回归方向与变化幅度）。

对每个有效槽位 t，将覆盖 t 的窗口 `Δmargin_W` 取最大值构成支持性 temporal importance curve；另保存窗口全表和覆盖 t 的最大 `|Δr_W|` 回归曲线，不能把未覆盖或 padding 槽位记作有效零分。最重要区间取最大正 `Δmargin_W`，若无正值则取 `|Δmargin_W|` 最大者并标注 suppressive/non-supporting；并列按最早起点。索引统一 **0-based、半开 `[start,end)`**。整个窗口排名在原完整输入上计算，不进行未声明的再优化。

Faithfulness 只在 Attachment2 valid 评价：

1. **单区间对照：** 删除最重要 W；在同一原样本、同一模态、有效前缀内均匀抽取 32 个同长度随机区间，固定 RNG seed 20260924，并包括可能抽到 W 的情况；对比 `Δz`、`Δp`、`Δmargin` 和 `|Δr|`，保存随机均值/分位数及 top−random 差。若 L=w，仅有一个区间，该样本对照不具判别力，单列计数。
2. **删除曲线：** 对比例 q∈{0.10,0.20,0.30,0.40}，令 k=max(1,round(qL))。以原输入窗口曲线对有效槽位降序排序，删除前 k 个；随机对照同一模态的 k 个有效槽位，32 次独立抽取、固定 seed。记录四个 q 的分类 logit/置信度下降和回归有向/绝对变化及 top−random 面积差。此曲线允许不连续位置，与“最重要连续区间”分开解释。
3. 预先按 `video_id` 分组把 valid 分成设计子集和审计子集（固定哈希/种子、记录 ID 清单）；只在设计子集选择 ρ，审计子集只报告，且不回头改 ρ。标签仅用于报告分层误差/预测正确性，不以真标签决定单样本目标类别。54 个 Q2 缺失场景或 attachment2 test 不参与本协议选择。

判据预登记：top 删除的平均 `Δmargin` 和 `Δp` 高于同长度随机均值、top deletion curve 在各 q 的平均 drop 与面积差为正，且报告 bootstrap **按 video_id 分组**区间、负效应比例和原生全零子组。此为扰动忠实性证据，不是解释在真实世界中的因果正确性；若效应不成立，要如实报告，不能用附件4反向选窗口。回归没有固定正负目标，主要以 `|Δr|` 与随机比较，同时保留符号，避免只报告绝对值掩盖方向。Attention 仅作为内部诊断，可在锁定协议后比较其质量分布与 perturbation curve 的秩相关；不得写 `attention = explanation`。

## 5. 原始证据回看与解释卡

输出契约见 `q3_explanation_schema.json`。解释卡保留 sample_id、checkpoint seed/hash、特征版本、预测、两任务 φ、三对 interaction、主模态、窗口全表/曲线或其不可变引用、top 区间、faithfulness、原始证据引用与映射质量。最终 CSV 可扁平化主要字段并以 JSON sidecar 存详细曲线，但必须按官方交付要求完整覆盖样本且可复核。

**关键限制：** 附件2 aligned_50 文档只说明“三模态最多50个相互对应的序列位置”，未提供每个位置的秒级时间戳或 `raw_text` token 位置映射。附件4清单显示有对应视频，但本轮只允许读文件级元数据，不能验证 pkl 内 ID、文本、时间字段或 pkl↔mp4 对应。Q1 的均匀 50 窗是另一路自主特征，不可直接当作附件2/4位置映射。后续必须建立并核验 `sample_id→video`、`[start,end)→raw media time`、文本片段来源/时间戳以及帧 PTS 的映射；若只能粗略对应，应标明 `approximate` 与误差/方法，不能宣称精确局部证据。缺映射时原始证据字段为 null 且 `grounding_status=unverified`，不能伪造文本片段或帧时间。

## 6. 最小实施计划、模块与门槛（E–H）

| 模块（拟新增） | 职责 | 训练？ |
|---|---|---|
| `src/q3/frozen_predictor.py` | 只读 checkpoint/配置、hash 检查、eval/无梯度和统一输出 | 否 |
| `src/q3/data_adapter.py` | valid 与未来无标签专项集共享的 aligned-50 校验和样本适配，不从零值猜 padding | 否 |
| `src/q3/coalitions.py` | 8 联盟批量前向、exact Shapley、pair interaction、效率自检 | 否 |
| `src/q3/temporal_occlusion.py` | 有效前缀连续窗、曲线与 top 区间 | 否 |
| `src/q3/faithfulness.py` | 固定随机种子、同长度删除对照、分组汇总 | 否 |
| `src/q3/evidence_grounding.py` | ID/槽位/原始素材映射及验证、误差标记 | 否 |
| `src/q3/export.py` | JSON schema 校验、CSV/JSON sidecar、完整性与文件哈希 | 否 |

**F. 全框架不需要新增训练。** P2 已训练且冻结；归因、交互、遮挡、验证、证据索引都是确定性或固定随机对照计算。Q3 可如实描述 Q2 的原训练目标和参数，不虚构 HEAF 训练损失。

**G. Attachment4 解封前锁定：** 单个 checkpoint seed/hash；class target/margin、回归 baseline 和后处理；全零 coalition 规则；主模态及并列规则；有效长度获取；ρ 候选与最终值、stride、边界/曲线聚合规则；top interval 与无正证据规则；随机对照 seed/次数、删除比例和排序；分组设计/审计 ID 清单；faithfulness 判据；证据映射标准、无法映射时的降级措辞；schema 版本和最终输出格式。不能根据 Attachment4 的特征分布、预测或媒体观感改变这些选择。

**H. 实现风险：** 整模态全零超出 Q2 局部缺失分布；projection bias 使空联盟非零；softmax attention 在全零有效行仍分配权重；原生零与遮挡零数值相同；只有共享 text_bert mask，附件4内部字段未核实；局部特征可能携带上下文；aligned 槽位到原始秒数没有现成证明；validation 内部选择仍有开发集乐观偏差；同视频多 clip 需分组统计；分类与回归归因单位不同；负 φ/负遮挡变化不可强行解释成支持；seed42 checkpoint 尚需在 Q3 包装器中通过加载与哈希核验。上述风险以明示、测试和门槛管理，不修改 Q2。

### 下一阶段最小验证顺序

1. 将上述 seed42 checkpoint/hash 写入只读 Q3 配置，在 valid 验证 P2 加载、clean 输出与现有 Q2 结果一致、输出有限且重复调用一致；不索引 test。
2. 用 valid 校验 8 联盟输入不改源张量/padding、效率恒等式、全零及原生零案例；检查分类 margin 和回归 baseline。
3. 按 video_id 分组的设计子集选择 ρ；审计子集给出同长度随机对照、deletion curve、负例及分层统计；不把此 Q3 内部分组写回 Q2 protocol。
4. 确定原始证据映射的可执行标准并冻结解释配置、schema、代码版本；仅此后解封 Attachment4，先做输入/ID/媒体对应审计，再做最终推理与解释卡。

**明确结论：`READY_FOR_Q3_EXPERIMENT`。** P2 前向、8 联盟、连续窗和忠实性对照具有可实现接口；还未实测解释忠实性，也未证明 Attachment4 的原始证据时间映射，故当前绝非 `READY_FOR_ATTACHMENT4_INFERENCE`。
