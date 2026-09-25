# 附件3最终推理交付检查

**状态：PASS。** 在文本接口回归测试通过后，用未修改的锁定 Q2 模型对附件3对齐版全量30条样本推理。正式预测文件为 `attachment3_predictions.csv`，审计扩展文件为 `attachment3_predictions_audit.csv`，机器汇总为 `attachment3_summary.json`。

| 检查项 | 结果 |
|---|---|
| 附件3对齐文件覆盖 | 30/30，清单 SHA256 全匹配 |
| 实际字段 | 仅 `text_bert`、`audio`、`vision`；无标签字段 |
| 文本接口回归门槛 | PASS；20个已知样本、604个有效行，模型输出差低于预定容差 |
| sample_id | 30/30 唯一；源文件不含 `id`，采用文件名 stem（`附件3_01`—`附件3_30`） |
| 分类值 | 全部在 0/1/2，名称映射与模型锁定文件一致 |
| 回归、logits、概率 | 全部有限值；未截断回归输出 |
| 概率和 | 最大绝对误差 `8.9406967e-8` |
| CSV 重新读取 | 30行、ID顺序与文件清单一致 |
| checkpoint | 前后 SHA256 均为 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff` |

赛题说明要求提交附件3预测 CSV，但没有给出固定列模板。本次正式 CSV 使用 `sample_id,predicted_class_id,predicted_class_name,predicted_intensity`；置信度、三类概率、logits、有效长度与源文件哈希另存在 audit CSV。正式文件采用 UTF-8 with BOM，便于 Excel 读取。若后续组委会另提供模板，只需按其列名重排现有预测，不改变模型输出。

## 无标签预测分布

| 类别 | 数量 | 比例 |
|---|---:|---:|
| Negative | 7 | 23.33% |
| Neutral | 11 | 36.67% |
| Positive | 12 | 40.00% |

连续强度原始输出：均值 `0.123114`，样本标准差 `0.611792`，最小 `-1.642545`，P10 `-0.467665`，P25 `-0.235606`，中位 `0.105127`，P75 `0.554075`，P90 `0.781083`，最大 `1.539758`。平均分类置信度 `0.624599`。这些是无标签预测分布，**不是** Accuracy、F1、MAE 或 Pearson。

未根据附件3输出调阈值、改权重或重训；未使用附件2 test 或附件4标签。附件4仅在预测前用于已有 `text` 的无标签数值接口回归。

ATTACHMENT3_FINAL_INFERENCE_COMPLETE = YES
