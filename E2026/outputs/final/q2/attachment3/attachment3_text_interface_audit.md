# 附件3文本接口适配验证

**最终门槛：PASS。** 本报告只验证锁定 Q2 模型的文本输入接口；正式附件3预测须在此门槛通过后另行执行。未重训或修改 Q2 checkpoint，未读取任何标签。

## 30 个附件3文件

对齐版 `附件3_01.pkl`—`附件3_30.pkl` 的清单 SHA256 全部匹配。30/30 文件的顶层 keys 均为 `test`，其中字段 keys 均且仅为 `text_bert`、`audio`、`vision`；没有 `text`、`id` 或任何标签字段。文件中的批量形状分别是 `text_bert[1,3,50]`、`audio[1,50,74]`、`vision[1,50,35]`；取单样本后为 `[3,50]`、`[50,74]`、`[50,35]`。所有数值有限。逐文件证据见 `attachment3_input_audit.json`。

## 复用的 Q3 文本编码路径

- Q3 已验证代码：`scripts/run_q3_text_row_identity.py`；其中相同的有效前缀 BERT 前向现提取为 `src/q3/text_feature_reconstruction.py`，Q3 审计脚本也调用该函数。
- tokenizer/model：`bert-base-uncased`，revision `86b5e0934494bd15c9632b12f734a8a67f723594`。
- `tokenizer.json` SHA256：`ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98`；`model.safetensors` SHA256：`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`。均从本地已有缓存读取并验证，不选择新权重。
- 直接使用给定 `text_bert` 的 token IDs 与 token-type IDs，只送入 `text_bert[1]==1` 的有效前缀；与 Q3 验证流程相同，不传显式 attention mask，不进行层平均或池化，取 `BertModel.last_hidden_state`。补齐到50行时仅在 padding 后缀填精确零；Q2 模型仍使用原始 `text_bert[1]` 有效位掩码，不从零值推断 padding。

## 已知 `text` 样本的数值回归

使用 Q3 已验证的附件4对齐版 20 条无标签样本，因为它们同时包含官方 `text[50,768]` 和 `text_bert[3,50]`；不访问附件2 test。对全部604个有效文本行比较：

| 检查 | 结果 | 预先固定的门槛 |
|---|---:|---:|
| 同索引 cosine argmax | 604/604 | 604/604 |
| 有效行平均同索引 cosine | 0.9999999888 | 最小逐行 cosine ≥0.99999 |
| 特征最大绝对差 | 3.0040741e-5 | ≤5e-5 |
| 特征 RMSE | 8.3976630e-7 | ≤2e-6 |
| 锁定 Q2 模型 logits 最大绝对差 | 2.4437904e-6 | ≤1e-4 |
| 锁定 Q2 模型回归输出最大绝对差 | 1.1175871e-6 | ≤1e-4 |

Q3 已发表在项目内的 20 样本行身份指标可复算；本轮进一步验证了下游锁定模型的预测输出。模型在两次前向中均为 `eval()` 且无梯度，audio、vision、padding_mask 完全相同，仅交换官方 `text` 与重构 `text`。Q2 checkpoint SHA256 前后均为 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`。

本表的平均 cosine 按604个有效行汇总；Q3 原报告中的 `0.999999994` 是先逐样本求均值再对20个样本等权平均，汇总权重不同。最大绝对差、全局 RMSE 和604/604同索引结果与原报告一致。

接受容差在本轮重构前写入 `scripts/validate_attachment3_text_interface.py`，没有依据附件3预测结果调整。历史附件2/4预计算器的原始权重 revision 没有独立留档，但当前固定候选在已知特征与最终模型输出两层均达到数值近似一致，满足本次接口适配门槛。

**TEXT_INTERFACE = PASS。** 允许后续使用已验证 BERT adapter 生成附件3的 `text[50,768]`，再输入未改变的锁定 Q2 模型。机器记录及逐样本误差见 `attachment3_text_interface_audit.json`。
