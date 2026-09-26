# 源稿与现有记录的不一致项

本文件独立于论文正文。依本轮要求，未修改锁定实验、数值表、核心公式或原参考文献条目。以下事项仍需在正式合稿前处理，不能把本次文字改写视为已解决这些冲突。

## 1. LTARP打分器公式与实现不同

- 源稿公式(11)使用含tanh的两层打分器。
- D:/华为杯/E2026/src/models/pooling_residual.py 的attention_scorer为三模态各一个nn.Linear(d,1)；三个打分器共880参数，加三个残差系数为883。现用架构图也绘制线性打分器。
- 本次保留公式(11)原文，没有改模型、改公式或改参数量。公式、图示与代码尚未完全一致。
- 需要后续明确授权的公式修订；不能称原公式已由实现核实。

## 2. 604槽位接口检查的数据来源与输出类型

- 源稿把20条检查样本归于附件2验证集，并把分类差值称为Softmax差值。
- 现有 E2026/outputs/final/q2/attachment3/attachment3_text_interface_audit.md 及JSON明确记载：检查使用附件4的20条无标签已知文本特征样本，比较604个有效行；分类2.4437904e-6是logits最大差值。
- 本次保留原统计数值，正文暂写“20条同时具有词元输入与预计算文本表示的样本”和“分类输出”，没有继续写错误的附件2来源或Softmax名称。
- 这是暂时避开未一致的具体称谓，尚未完成来源表述修订。正式版本应明确实际附件来源与logits含义；不存在本轮新增的附件2一致性实验。

## 3. 原生视觉全零子集的准确率叙述

- 源稿原有MMP约0.467、LTARP约0.400的表述。
- 实际 E2026/outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig6_clean_vision_all_zero.csv 与现用图件记录：

| 初始化 | MMP准确率 | LTARP准确率 |
|---|---:|---:|
| 42 | 0.533333 | 0.466667 |
| 43 | 0.533333 | 0.466667 |
| 44 | 0.466667 | 0.466667 |

- 子集为15条。原源稿数字与现图并不一致；本次只保留“部分初始化下分类表现低于基线”的真实定性边界，没有把上述替代数值写进正文，也没有更改图件。

## 4. 位置退化图的正负方向

- 源稿图注把正值称为退化；现用图 fig12_q2_modality_location_absolute_degradation.pdf 图内已说明“负值表示下降，正值表示改善”，MAE采用完整输入减缺失输入。
- 本次图注改为中性表述“相对完整输入的指标变化”，保留图内真实符号说明；没有翻转色标、改格值或改图。
- 图件版本与源稿部分逐格近似值也存在差异，因此正文不再复述这些旧近似值。

## 5. 公开方法引文条目

源稿引文键及参考文献原文保持不变，但下列条目与原论文不一致：

- q2selfmm 列出的标题是MMIM相关论文。Self-MM原论文为《Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis》，AAAI 2021。[原论文页面](https://ojs.aaai.org/index.php/AAAI/article/view/17289)
- q2mmim 标题“Bi-bimodal information bottleneck…”与MMIM原论文不符。MMIM原论文为《Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis》，EMNLP 2021。[原论文页面](https://aclanthology.org/2021.emnlp-main.723/)
- q2missmodal 所列论文与方法来源不符。MissModal原论文为《MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis》，TACL 2023。[原论文页面](https://aclanthology.org/2023.tacl-1.94/)
- q2m3s 所列CVPR 2023来源与方法来源不符。M3S原论文为《Missing Modality meets Meta Sampling (M3S): An Efficient Universal Approach for Multimodal Sentiment Analysis with Missing Modality》，AACL-IJCNLP 2022。[原论文页面](https://aclanthology.org/2022.aacl-main.10/)

本轮未改这四条参考文献，也未把适配结果写成原论文官方结果。概览表引文编号仍继承源稿，因此正式发表前需要单独修正引用关系。

## 6. 两套选模口径已核实，不是待修正项

- 结构比较及公开方法结果使用完整输入选择，已有checkpoint_selection_fairness和public_baselines_expanded结果支持。
- 缺失规律和初始化图采用历史综合鲁棒分数选择。历史P2综合均值0.744777与完整输入选择口径下的0.744722不是同一套检查点结果。
- 本次2.5已明确两套口径与所对应的表/图，未合并、重新选模或替换任何结果。
