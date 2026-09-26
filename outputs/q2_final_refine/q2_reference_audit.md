# Q2 原始文献与接口适配核对

核对日期：2026-09-27。以ACL Anthology、AAAI出版页面/原始PDF及ACM出版者存入Crossref的DOI元数据为依据。ACM页面访问受限时，明确采用出版者登记元数据，不以二手综述替代。

本稿保留11种公开方法。名称对应原论文；表10描述的是本文统一特征接口下的实际适配机制，不宣称原仓库模型或原论文结果的直接复现。参考文献编号仅为独立Q2稿的局部编号，全文合并后应按首次引用顺序统一生成。

## TFN

**当前引用：** Zadeh A, Chen M, Poria S, et al. Tensor fusion network for multimodal sentiment analysis[C]//Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2017: 1103--1114.

**核对后：** Tensor Fusion Network for Multimodal Sentiment Analysis

- 作者：Amir Zadeh; Minghai Chen; Soujanya Poria; Erik Cambria; Louis-Philippe Morency
- 会议/期刊：Proceedings of EMNLP
- 年份/卷期：2017
- 页码：1103–1114
- DOI：10.18653/v1/D17-1115
- 官方来源：[TFN原论文](https://aclanthology.org/D17-1115/)
- 元数据入口：[出版元数据](https://aclanthology.org/D17-1115/)
- 修订：增加DOI，原文献身份一致

**与代码对应：** 增广三路表示的外积融合；本地采用预计算对齐特征，文本递归编码、语音视觉池化。

## LMF

**当前引用：** Liu Z, Shen Y, Lakshminarasimhan V B, et al. Efficient low-rank multimodal fusion with modality-specific factors[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL). 2018: 2247--2256.

**核对后：** Efficient Low-rank Multimodal Fusion With Modality-Specific Factors

- 作者：Zhun Liu; Ying Shen; Varun Bharadhwaj Lakshminarasimhan; Paul Pu Liang; AmirAli Bagher Zadeh; Louis-Philippe Morency
- 会议/期刊：Proceedings of ACL, Volume 1
- 年份/卷期：2018
- 页码：2247–2256
- DOI：10.18653/v1/P18-1209
- 官方来源：[LMF原论文](https://aclanthology.org/P18-1209/)
- 元数据入口：[出版元数据](https://aclanthology.org/P18-1209/)
- 修订：增加DOI，原文献身份一致

**与代码对应：** 模态专属低秩因子；本地为rank=8低秩融合。

## MFN

**当前引用：** Zadeh A, Liang P P, Mazumder N, et al. Memory fusion network for multi-view sequential learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2018: 5634--5641.

**核对后：** Memory Fusion Network for Multi-view Sequential Learning

- 作者：Amir Zadeh; Paul Pu Liang; Navonil Mazumder; Soujanya Poria; Erik Cambria; Louis-Philippe Morency
- 会议/期刊：Proceedings of the AAAI Conference on Artificial Intelligence
- 年份/卷期：2018, 32(1)
- 页码：5634–5641
- DOI：10.1609/aaai.v32i1.12021
- 官方来源：[MFN原论文](https://ojs.aaai.org/index.php/AAAI/article/view/12021)
- 元数据入口：[出版元数据](https://ojs.aaai.org/index.php/AAAI/article/view/12021)
- 修订：补全卷期和DOI，页码由官方PDF首尾页确认

**与代码对应：** 模态内递归记忆、跨记忆注意力及门控记忆；本地使用三路LSTMCell。

## MulT

**当前引用：** Tsai Y H H, Bai S, Liang P P, et al. Multimodal transformer for unaligned multimodal language sequences[C]//Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL). 2019: 6558--6569.

**核对后：** Multimodal Transformer for Unaligned Multimodal Language Sequences

- 作者：Yao-Hung Hubert Tsai; Shaojie Bai; Paul Pu Liang; J. Zico Kolter; Louis-Philippe Morency; Ruslan Salakhutdinov
- 会议/期刊：Proceedings of ACL
- 年份/卷期：2019
- 页码：6558–6569
- DOI：10.18653/v1/P19-1656
- 官方来源：[MulT原论文](https://aclanthology.org/P19-1656/)
- 元数据入口：[出版元数据](https://aclanthology.org/P19-1656/)
- 修订：增加DOI，原文献身份一致

**与代码对应：** 定向跨模态注意力；本地为六路跨注意力及三路模态内时序记忆。

## MISA

**当前引用：** Hazarika D, Zimmermann R, Poria S. MISA: Modality-invariant and -specific representations for multimodal sentiment analysis[C]//Proceedings of the 28th ACM International Conference on Multimedia (MM). 2020: 1122--1131.

**核对后：** MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis

- 作者：Devamanyu Hazarika; Roger Zimmermann; Soujanya Poria
- 会议/期刊：Proceedings of ACM Multimedia
- 年份/卷期：2020
- 页码：1122–1131
- DOI：10.1145/3394171.3413678
- 官方来源：[MISA原论文](https://dl.acm.org/doi/10.1145/3394171.3413678)
- 元数据入口：[出版元数据](https://api.crossref.org/works/10.1145/3394171.3413678)
- 修订：增加DOI，原文献身份一致

**与代码对应：** 共享与模态特有表示；本地保留重构、差异约束和CMD约束，编码接口适配。

## Self-MM

**当前引用：** Han W, Chen H, Poria S. Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis[C]//Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2021: 9180--9192.

**核对后：** Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis

- 作者：Wenmeng Yu; Hua Xu; Ziqi Yuan; Jiele Wu
- 会议/期刊：Proceedings of the AAAI Conference on Artificial Intelligence
- 年份/卷期：2021, 35(12)
- 页码：10790–10797
- DOI：10.1609/aaai.v35i12.17289
- 官方来源：[Self-MM原论文](https://ojs.aaai.org/index.php/AAAI/article/view/17289)
- 元数据入口：[出版元数据](https://ojs.aaai.org/index.php/AAAI/article/view/17289)
- 修订：原引用误用了MMIM论文；更正为Self-MM原论文

**与代码对应：** 多任务与单模态辅助目标；本地采用标签与融合预测各半的单模态伪目标，并非原动态权重机制的完整复刻。

## MMIM

**当前引用：** Han W, Chen H, Gelbukh A, et al. Bi-bimodal information bottleneck for multimodal sentiment analysis[C]//Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2021: 9180--9192.

**核对后：** Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis

- 作者：Wei Han; Hui Chen; Soujanya Poria
- 会议/期刊：Proceedings of EMNLP
- 年份/卷期：2021
- 页码：9180–9192
- DOI：10.18653/v1/2021.emnlp-main.723
- 官方来源：[MMIM原论文](https://aclanthology.org/2021.emnlp-main.723/)
- 元数据入口：[出版元数据](https://aclanthology.org/2021.emnlp-main.723/)
- 修订：原题名和作者错配；更正为层级互信息最大化原论文

**与代码对应：** 层级互信息约束；本地以模态间和融合—单模态InfoNCE作为约束，不等同于原文全部BA/CPC估计器。

## TFR-Net

**当前引用：** Yuan Z, Li W, Xu H, et al. Transformer-based feature reconstruction network for robust multimodal sentiment analysis[C]//Proceedings of the 29th ACM International Conference on Multimedia (MM). 2021: 4400--4408.

**核对后：** Transformer-based Feature Reconstruction Network for Robust Multimodal Sentiment Analysis

- 作者：Ziqi Yuan; Wei Li; Hua Xu; Wenmeng Yu
- 会议/期刊：Proceedings of ACM Multimedia
- 年份/卷期：2021
- 页码：4400–4407
- DOI：10.1145/3474085.3475585
- 官方来源：[TFR-Net原论文](https://dl.acm.org/doi/10.1145/3474085.3475585)
- 元数据入口：[出版元数据](https://api.crossref.org/works/10.1145/3474085.3475585)
- 修订：末页由4408更正为4407并补DOI

**与代码对应：** 缺失条件下特征重构；本地为时序上下文与潜在表示SmoothL1重构，不宣称原仓库完整运行。

## MissModal

**当前引用：** Yang D, Huang S, Kuang H, et al. Disentangled representation learning for multimodal emotion recognition with missing modalities[C]//Proceedings of the 30th ACM International Conference on Multimedia (MM). 2022: 4434--4443.

**核对后：** MissModal: Increasing Robustness to Missing Modality in Multimodal Sentiment Analysis

- 作者：Ronghao Lin; Haifeng Hu
- 会议/期刊：Transactions of the Association for Computational Linguistics
- 年份/卷期：2023, 11
- 页码：1686–1702
- DOI：10.1162/tacl_a_00628
- 官方来源：[MissModal原论文](https://aclanthology.org/2023.tacl-1.94/)
- 元数据入口：[出版元数据](https://aclanthology.org/2023.tacl-1.94/)
- 修订：原引用为另一篇缺失情感识别论文；更正为MissModal原论文

**与代码对应：** 完整/缺失表征的几何、分布及情感语义对齐；本地用对比项、均值/标准差代理和分类KL实现。

## M3S

**当前引用：** Wang Y, Shen Y, Liu Z, et al. Meta-mining multimodal missing states for incomplete multimodal learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023: 15422--15431.

**核对后：** Missing Modality meets Meta Sampling (M3S): An Efficient Universal Approach for Multimodal Sentiment Analysis with Missing Modality

- 作者：Haozhe Chi; Minghua Yang; Junhao Zhu; Guanhong Wang; Gaoang Wang
- 会议/期刊：Proceedings of AACL-IJCNLP, Volume 1
- 年份/卷期：2022
- 页码：121–130
- DOI：10.18653/v1/2022.aacl-main.10
- 官方来源：[M3S原论文](https://aclanthology.org/2022.aacl-main.10/)
- 元数据入口：[出版元数据](https://aclanthology.org/2022.aacl-main.10/)
- 修订：原题名、作者、会议和年份均错配；以本地适配引用的Missing Modality meets Meta Sampling为准

**与代码对应：** 缺失模态元采样；本地以LMF为骨干，完整输入支持更新及缺失查询的一步一阶元更新，非另一篇multi-head/meta-mining论文。

## MMIN

**当前引用：** Zhao J, Li R, Qin J. Missing modality imagination network for emotion recognition with uncertain missing modalities[C]//Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL). 2021: 2608--2618.

**核对后：** Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities

- 作者：Jinming Zhao; Ruichen Li; Qin Jin
- 会议/期刊：Proceedings of ACL-IJCNLP, Volume 1
- 年份/卷期：2021
- 页码：2608–2618
- DOI：10.18653/v1/2021.acl-long.203
- 官方来源：[MMIN原论文](https://aclanthology.org/2021.acl-long.203/)
- 元数据入口：[出版元数据](https://aclanthology.org/2021.acl-long.203/)
- 修订：作者Qin Jin的姓为Jin，原Qin J更正为Jin Q，并补DOI

**与代码对应：** 缺失模态潜在表示推断；本地含残差细化和循环约束，非原模型全部CRA实现。

## 实现证据

`E2026/src/models/public_baselines.py`：TFN、MulT、MISA。
`E2026/src/models/public_baselines_extended.py`：LMF、MFN、Self-MM、MMIM、TFR-Net、MissModal、MMIN、M3S。
`E2026/scripts/run_public_baseline_extended.py`：M3S支持/查询一步一阶更新与其他辅助损失训练路径。
M3S本地adaptation notes明确指向AACL 2022原论文；模型类继承低秩骨干，训练器执行元更新，文献身份与这条实现路线一致。

MFN页码补充依据：[AAAI官方PDF](https://ojs.aaai.org/index.php/AAAI/article/download/12021/11880)，首尾页为5634和5641。
