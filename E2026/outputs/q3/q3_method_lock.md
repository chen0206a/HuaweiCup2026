# E2026 Q3 HEAF method lock

**状态：METHOD_LOCKED_BEFORE_ATTACHMENT4_CONTENT_AUDIT。** 锁定依据为 Q3-1 的 `HEAF_VALIDATION_PASSED` 实测结果，完整机器契约在 `q3_method_lock.json`，可执行参数在 `../../configs/final/q3_heaf.yaml`。锁定时尚未解析 Attachment4 特征或媒体；之后不得依其分布或个例重选 HEAF 参数。

| 部件 | 固定定义 |
|---|---|
| Predictor | 冻结 B5-P2 seed42，`mean_attention`、eval、no_grad、float32、无归一化；checkpoint SHA256 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff` |
| 输入 | Attachment4 `aligned_50`，T/A/V 50×768/74/35；有效前缀按 `text_bert[:,1,:]==1` 校验；排除模态仅清零有效特征行，padding 与原生零记录不变 |
| 模态归因 | 依次枚举 ∅、T、A、V、TA、TV、AV、TAV；三玩家 exact Shapley；分类目标为完整输入预测类别固定后的 log-odds，回归为原始回归输出；空联盟基线是全有效特征置零后的模型输出 |
| Pair interaction | `I_ij=(Δ_ij(∅)+Δ_ij({k}))/2`，`Δ_ij(S)=v(Sij)-v(Si)-v(Sj)+v(S)`；负值只称 negative interaction |
| 主模态 | 分类取最大正 Shapley，否则取绝对值最大并标注非支持；回归取绝对值最大并报告方向；并列 T→A→V；时间主流程跟随分类主模态 |
| 时间定位 | 有效前缀内 `w=max(1,round(0.30 L))`，stride=1；top 为最大正 class-margin drop，若无正值取绝对值最大并标注，平局取最早窗口；50 位曲线取覆盖窗口的最大 margin drop |
| Faithfulness | 同长度随机窗口 32 次；10/20/30/40% 删除曲线；随机种子 20260924；固定预测类别的 margin/confidence drop 为主，raw class logit 辅助，回归保留 signed/absolute shift |
| 解释卡 schema | `q3_explanation_schema.json` v0.2.0，SHA256 `25abe930caad7f0b4a3cb867716bf00a4b58ea554af96cd66e89f5b21809e2d2` |

Q3-1 验证配置 SHA256 `e23e4af9e50521dc95130ec8eac2d919a19067d3f90e1c9ff2ec4e7ae7bf7a64`，实测指标 JSON SHA256 `78aa1c76be1aa2be472dce57eca0c443dd19ae61e2b7bffdd36cd7129e19c2ac`，源提交 `7e79783`。Q3-1 的同窗口 top 与随机对照含选取优势，锁定的是预先验证的扰动协议，不将其写作现实因果解释。

**Grounding 尚未锁定为 VERIFIED/APPROXIMATE。** Q3-2 仅核查 Attachment4 对齐版接口、ID 与实际时序元数据。槽位不能在无证据时转换为词片段、秒数或关键帧；无法证明的解释卡原始证据字段保持 null。Q3-2 不运行全量最终解释。
