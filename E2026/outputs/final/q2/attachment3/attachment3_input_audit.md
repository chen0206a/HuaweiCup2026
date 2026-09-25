# 附件3最终推理输入接口审计

**结论：停止正式推理。** 锁定的 Q2 主模型需要预计算的 `text[1,50,768]`，而本地附件3对齐版的 30 个 pickle 均未提供 `text`。按本次任务的“不一致即停止”规则，未执行模型前向，也未生成预测 CSV。状态为 `BLOCKED_INPUT_INTERFACE_MISMATCH`。

## 核验范围

- 数据：官方附件3 `对齐版本/附件3_01.pkl` 至 `附件3_30.pkl`，共30个文件；未使用未对齐版本。
- 清单：`data/manifests/attachment3_sealed_inventory.json`。30/30 文件 SHA256 与既有清单匹配。
- 锁定模型：`outputs/final/q2/q2_model_lock.json` 中的 B5-P2 seed42 最终主 checkpoint，即 `outputs/checkpoints/b5_pooling_p2_best_robust_score.pt`。推理前后 SHA256 均为 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`，与锁定清单一致。
- 本地环境：Python 3.12.3、PyTorch 2.12.0+cpu、NumPy 2.1.3、Pandas 2.2.3。30 条样本规模足以在 CPU 上推理，但输入接口检查先于模型执行。

## 实际数据结构

30/30 文件顶层均只有 `test`，其下字段完全一致：

| 字段 | 实际形状 | dtype | 与锁定模型的关系 |
|---|---|---|---|
| `text_bert` | `[1,3,50]` | float32 | BERT token/attention/type 输入；不能直接替代 768 维预计算 `text` |
| `audio` | `[1,50,74]` | float32 | 形状符合 |
| `vision` | `[1,50,35]` | float32 | 形状符合 |

全部数值字段均为有限值。30/30 文件均缺少 `text[1,50,768]` 和文件内 `id` 字段；样本序号目前仅能由文件名识别。现有 `Aligned50Dataset` 和锁定模型均消费已经预计算的 `text`，不是 `text_bert` token IDs。附件2的 `aligned_50.pkl` 提供该 `text` 字段，但附件3本地对齐文件不提供。

## 停止原因与边界

赛题允许 `text` 与 `text_bert` 作为不同文本接口，但本项目的已锁定 Q2 模型训练于预计算 `text`。临时使用 BERT 将附件3 `text_bert` 转换成 768 维特征会引入一个尚未纳入 Q2 最终推理锁定协议的编码步骤；历史预计算文本编码器的确切权重 revision 也未固定。不能仅因输出维数相同就假定编码结果与附件2训练特征分布完全一致。因此本轮不下载模型、不猜编码器、不调整模型，也不把 token IDs 伪装为 `text` 特征。

要继续正式推理，需要提供与附件2 `text` 同一预计算流程生成的附件3 `text[30,50,768]`，或另行明确批准并核验从 `text_bert` 到该特征空间的固定转换流程。后一方案需先在附件2已知输入上证明数值一致，且不得使用附件3输出分布调参。

本轮没有访问附件2 test、附件4或任何潜在附件3标签，没有计算 Accuracy、F1、MAE 或 Pearson，也没有生成无依据的预测结果。逐文件哈希、字段、形状和 dtype 见 `attachment3_input_audit.json`。

`ATTACHMENT3_FINAL_INFERENCE_COMPLETE = NO`
