# Q2 三篇论文阅读笔记

赛题原文优先：本题是一个或多个模态的**局部连续时间段**缺失，需同时输出 Negative / Neutral / Positive 极性和连续情感强度；附件 2 训练、验证，附件 3 无标签专项推理。以下“最小版本”只是后续实验候选，本阶段没有实现或训练。

## DEAR（Findings of ACL 2026）

原文：[ACL Anthology](https://aclanthology.org/2026.findings-acl.1517/)；本地 `2026.findings-acl.1517.pdf`。

- **解决问题：**缺失输入使重建表示偏离完整模态分布，固定融合权重又会放大不可靠信息；目标是兼顾恢复与可靠决策。
- **缺失形式：**模拟 0–100% 随机缺失率；语音、视觉的部分特征置零，文本 token 换成 `[UNK]`，另测试整个模态丢弃。原文没有将连续缺失区间作为主要生成机制。
- **核心模块：**HDCR（重建后的单模态与联合分布约束）、SURE（样本级/模态级迁移难度与可靠权重）、协同注意力流与鲁棒单模态预测流的动态门控。
- **主要 loss：**单模态和联合表示的多带宽 Gaussian-kernel MMD；SURE 难度对真实重建误差的校准损失；主预测及辅助预测损失。总目标为 `L_main + α L_aux + L_rec + β L_cal`。
- **CMU-MOSEI 用法：**在 MOSEI 上模拟不同缺失率与六种可用模态组合，报告 Acc、F1、MAE、Pearson 等；任务预测以情感强度回归为主，再按常用口径转换分类。其评价口径与本题三分类口径不可直接视为同一基准。
- **本题可借鉴：**重建质量不能只看逐点误差；可用人工遮挡处的恢复误差监督“可靠度估计”，让可靠度参与融合。
- **不匹配点：**SURE 权重按样本/模态给出，不能定位 50 步序列中某一连续区间；随机点缺失与整模态丢失都不能替代本题要求的区间位置、长度分析。
- **最小可实现版本：**在未来的 B3/B4 消融中，先对连续遮挡处的 latent 重建做 SmoothL1，再用遮挡区误差监督 `r[m,t]`；如需检验分布漂移，再单独添加轻量 MMD 消融。无需复现完整 HDCR/SR-DS。

## CMAD（ICCV 2025）

原文：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.html)；本地 `Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf`。

- **解决问题：**完整模态教师和缺失模态学生的特征及样本间关系不一致；不同模态组合难度不均使统一学生训练不稳定。
- **缺失形式：**以三位二值指示量标记整模态可用性，缺失时该模态整体置零；统一学生覆盖七种非空模态组合。
- **核心模块：**完整模态教师、缺失模态学生、CAFD（同样本特征与跨样本相关性蒸馏）、MAR（先等权学习，再依教师/学生损失差估计各组合难度并重加权）。
- **主要 loss：**CAFD 包含师生特征 MSE、正样本相关约束、师生与教师内部相关矩阵的 KL 对齐；MAR 包含任务损失和 logits 蒸馏（分类 DKD，回归 MSE），按模态组合难度加权；总目标 `L_MAR + σ L_CAFD`。
- **CMU-MOSEI 用法：**使用 MOSEI 的 22,856 个片段及预提取 BERT/OpenFace/COVAREP 特征；把情感强度标签/输出转成二分类正负，针对七种完整/缺失模态组合报告 Accuracy/F1。未覆盖本题要求的三分类与强度联合输出。
- **本题可借鉴：**完整输入教师向局部遮挡学生蒸馏表示；不同缺失模态组合的难度可用于检查训练采样和损失是否失衡。
- **不匹配点：**二值模态可用性不能表示 `m,t` 级连续缺失、位置或时长；原文的跨样本相关蒸馏可能给第一阶段引入额外复杂度。
- **最小可实现版本：**待 B2 稳定后，以同一训练集的完整输入模型为冻结教师，学生接受连续 block mask，先单独消融 `||z_student-z_teacher||²`；不预先实现 CAFD/MAR 全套。

## EMOE（CVPR 2025）

原文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.html)；本地 `Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.pdf`。

- **解决问题：**各模态的重要性随样本变化，简单融合易由优势模态主导；融合也容易丢失单模态有预测力的专有信息。
- **缺失形式：**论文主体使用完整多模态输入做情感/意图预测，没有针对缺失模态的训练或连续缺失区间实验。
- **核心模块：**每个模态作为专家，Router Network 输出样本级模态权重；按权重融合；单模态预测头与路由选择的单模态蒸馏；路由熵平衡。
- **主要 loss：**融合预测与各单模态预测的 MAE；路由熵与权重/单模态预测能力的相似约束；融合与加权单模态 logits 的单向蒸馏。总目标 `L_task + λ1 L_balance + λ2 L_ud`。
- **CMU-MOSEI 用法：**使用 MOSEI 完整模态数据，比较 aligned 与 unaligned 特征，报告 Acc-7、Acc-2、F1、MAE；使用 768 维 BERT、74 维 COVAREP、35 维 Facet 等现成特征。其完整输入成绩不能作为局部缺失性能。
- **本题可借鉴：**按输入自适应决定模态贡献；保留轻量单模态辅助预测，避免融合完全依赖文本。
- **不匹配点：**路由权重是每个样本的 `α_m`，不是时间步 `α[m,t]`；没有 availability mask 或缺失重建，熵平衡在真实缺失区域也可能错误抬高不可用模态。
- **最小可实现版本：**未来先在 B4 中用 50 步、三模态的轻量 router 输出 `α[m,t]`，显式输入 availability/reliability，并仅在有效时间步融合；单模态辅助头与熵项各自独立消融。

## 共同的使用边界

三篇论文的 benchmark 数字和分类阈值均不直接移植到赛题；本题应以赛题原文、附件 2 的真实字段和标签编码、以及固定的三分类 Accuracy/F1 与强度 MAE/Pearson 口径为准。上述方法是候选思路，不是已验证的本题结果。
