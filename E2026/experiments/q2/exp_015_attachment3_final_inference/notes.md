# Attachment3 final inference

原始附件3对齐文件不含预计算 `text`，只含 `text_bert/audio/vision`。先运行 `python scripts/validate_attachment3_text_interface.py`，复用 Q3-2.6 的固定 BERT 路径，在20个同时含官方 `text` 与 `text_bert` 的无标签附件4样本上通过特征及 Q2 模型输出的数值回归。然后运行 `python scripts/run_attachment3_final_inference.py` 对附件3全部30文件推理。

接口门槛：604/604有效行同索引，特征最大绝对差 `3.0041e-5`、RMSE `8.3977e-7`，最终 logits 最大差 `2.4438e-6`、回归最大差 `1.1176e-6`。完整结果见 `outputs/final/q2/attachment3/attachment3_text_interface_audit.json`（路径相对 E2026 根目录）。

交付 QA：30/30覆盖、ID唯一、概率和误差 `<1e-7`、CSV重读通过、checkpoint前后哈希相同。类别预测 Negative/Neutral/Positive=`7/11/12`，连续输出均值 `0.123114`。附件3无标签；这些数值不能写成预测性能。未训练、未调阈值、未用附件2 test。模型论文表述仍沿用已有验证结果，不自动改论文结论。

主要结果：`outputs/final/q2/attachment3/attachment3_predictions.csv`、`attachment3_predictions_audit.csv`、`attachment3_summary.json`、`attachment3_delivery_check.md`。下一步公开 baseline 不属于本轮，尚未开始。
