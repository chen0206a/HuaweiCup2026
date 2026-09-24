# Q3-0 — HEAF 代码与数据接口审计

日期：2026-09-24。阶段：**设计记录，无训练、无调参、无预测结果**。本目录按用户指定放置 `notes.md`；尚未执行正式实验，因此不生成虚构的 `config.yaml` 或 `metrics.json`。完整方法和解释卡契约见 `../../../outputs/q3/q3_method_design.md` 与 `../../../outputs/q3/q3_explanation_schema.json`。

## 输入证据

- 官方 E 题 Word 原文第三至五部分：Q3 要求极性、强度、模态作用、主要参考模态、局部片段、原始媒体回看；附件4无标签，aligned/unaligned版本须与训练验证一致。
- `outputs/final/q2/q2_model_lock.md`、`configs/final/q2_b5_p2.yaml`、`outputs/final/q2/q2_checkpoint_manifest.json`：B5-P2 冻结，输入 aligned-50，三维度 768/74/35，三 seed 的 best-robust checkpoint 均在 manifest 中标为本地存在。此轮未加载权重或做推理。
- `src/models/pooling_residual.py`、`src/models/baseline.py`、`src/data/dataset.py`、`src/data/block_mask.py`、`src/data/preprocess.py`：前向只接收三模态张量和共享 padding mask；没有任意模态 mask 参数，但可在调用前把某模态有效特征置零。人工置零后仍保持有效位置身份，与 Q2 缺失模拟一致。
- 仓库根目录 `notes/E_Q1_Q2_REVIEW_20260924.md` 与 `notes/e_q1_q2_audit_evidence_20260924.json`：Q1/Q2 审查与证据。仓库中未找到另一个以 Astra 命名的 Q1/Q2 报告；这里以该审查报告为准。
- `data/manifests/q3/attachment4_inventory.json`：状态 `FILENAME_SIZE_HASH_ONLY`，`content_interpreted=false`；现有清单记录 80 个文件（对齐版 20 pkl + 20 mp4，未对齐版 20 pkl + 20 mp4）。只查看了清单的文件级元数据及目录名，未打开附件4任何 pkl/mp4。

## 审计判断

1. **P2 可作为冻结预测器。** Q3 实验拟固定 Q2 既有 best-robust 分数最高的 seed42 checkpoint（SHA256 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`），不组合三 seed；需新建只读包装器，固定 `mode=mean_attention`、`eval()`、`torch.no_grad()`、float32 和 `normalization=none`。确定性与加载校验留待下一阶段在附件2 valid 上验证。
2. **8 个 coalition 可枚举，exact Shapley 算量可控。** 零向量是训练接口中人工局部缺失的既有替代值；整模态全零比 Q2 已验证的局部缺失更远离原训练分布，因此须在 valid 上做稳定性检查，解释为模型干预归因而非现实因果。
3. **padding/native-zero 不能混淆。** `padding_mask` 必须由原 `text_bert[:,1,:]` 得到并保持不变；`native_zero_mask` 只作审计；coalition/遮挡另记 `intervention_mask`，不可通过全零行重建 padding。
4. **现有 `Aligned50Dataset` 要求标签。** 附件4无标签，后续须单独实现无标签适配器；本轮不解析附件4，不确认其内部键、ID 或时间元数据。
5. **原始证据映射未打通。** 官方说明只保证 aligned-50 位置的对应关系，没有逐位置物理时间戳或 token→位置映射。不能把 Q1 的 50 个均匀时间窗规则直接套到附件2/4，也不能把 `raw_text` 按字符比例切片。解封后需验证真实映射来源及时间误差，才能宣称定位到文本/语音/画面。

## 下次最小验证及停止门槛

仅使用 attachment2 valid；按 `video_id` 分组建立 Q3 内部设计/审计子集，不触碰 Q2 固定验证协议。先验证 8 coalition 的有限输出、Shapley 效率恒等式、全模态输出与冻结 P2 输出一致、全零输入及原生零值语义。再在设计子集按预登记候选窗口长度选择一个参数，在审计子集检查 50 位置连续遮挡、同长度随机对照与 deletion curve。所有协议参数在附件4解封前写入版本化配置并锁定。若全零干预出现非有限输出、mask 语义错误、或解释卡无法完成原始媒体映射，则停止最终解释输出并报告问题；不得借附件4分布修正阈值。

结论：**READY_FOR_Q3_EXPERIMENT**（仅指进入附件2 valid 的 Q3 实现与最小验证；附件4最终推理与证据卡仍受协议锁定和映射验收约束）。
