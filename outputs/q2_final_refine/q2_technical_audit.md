# Q2 技术口径与现有结果来源

本轮只读取代码、配置、指标及既有接口检查记录，计算文件哈希并编译文稿。没有训练、调参、模型推理或原始数据特征加载。

## 选模规则与图表对应

| 对象 | 实际参数选择 | 回答的问题 | 既有证据 |
|---|---|---|---|
| 表8：MMP与LTARP | 两者各初始化按完整输入得分 S_clean | 同一完整输入标准下的结构比较 | checkpoint_selection_fairness 的 seedwise/summary CSV |
| 表11、12及图14：11公开方法与LTARP | 各方法按 S_clean；同一初始化的完整与54缺失结果使用同一参数 | 完整输入、缺失平均与参数规模对照 | public_baselines_expanded 的 clean/missing summary、checkpoint manifest；汇总脚本导入 CleanSelect B0/P2 |
| 图10–13：比例、位置、收益、稳定性 | LTARP按 R；MMP为对应初始化按 S_clean 选择的基线 | 最终配置在固定场景集合中的响应和初始化波动 | q2_checkpoint_manifest.json；b5_p2_multiseed_summary.json；330场景表及绘图脚本 |
| 表9：候选结构与训练策略 | 保留已归档候选的原有选模及结果 | 结构探索比较 | 原消融表；不改变候选设置，均值—最大值只有一次初始化 |
| 表13：附件3推理 | 锁定的seed42最终LTARP参数 | 无标签预测交付 | attachment3_predictions.csv / predictions_audit.csv及delivery check |

不同规则确实产生不同的seed42 LTARP参数：S_clean选择第20轮，SHA256为 `4a4ab039f3e239961bf68e7afea6a187932f00b14481a8bc757df61ac05644a0`；R选择第15轮，SHA256为 `cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff`。seed43、44在这两类规则下指向同一参数文件，但选择目的仍需分别说明。实际轮次以manifest/seedwise为准，不依据文件名猜测。

文稿明确两类实验各自固定参数，遍历全部对应场景；没有按模态、比例或位置选择有利参数。两类实验的结果不能当作完全相同参数的重复评价。既有六份锁定参数与六条CleanSelect记录均重新核对文件SHA256，检查结果见 `qa/verification.json`。

## 综合分数与缺失语义

- `E2026/src/utils/metrics.py`：S中分类两项合计1/2，回归两项合计1/2；MAE除以标签跨度6并转换方向，Pearson平移缩放。文稿使用“本文定义”，不称官方评价。
- `E2026/src/evaluation/missing_benchmark.py`：R为完整输入得分与54种缺失得分等权均值各占1/2。
- 冻结benchmark SHA256：`3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`，45单模态+9双模态，728条valid。图10的单模态比例变化与图11的固定目标比例0.3不同；图12单模态位置汇总取五比例平均，双模态固定0.3，保留现有图件。
- 有效长度取随附掩码；遮挡有效前缀，padding mask与native-zero保持不变。遮挡分母仍为原有效长度。文稿明确目标比例与实际比例，并说明遮挡区间用零起始索引。
- 三次初始化是均值±样本标准差的统计重复单位，54个场景不是54次独立训练。

## 两处纸面公式修正

1. 式(8)：既有 `baseline.py` 使用带class weight且默认mean reduction的交叉熵。分母是批次标签权重之和，而非批次样本数。仅修订文稿公式，与已有实现保持一致；[PyTorch加权交叉熵定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)给出相同归一化方式。
2. 式(11)：既有 `pooling_residual.py` 对每模态采用 `nn.Linear(dim,1)` 打分，并非tanh两层网络。文稿修正为线性标量打分。三个打分器共880个参数，加三个残差系数为883，与锁定规模一致。

原式(19)重复均值定义，删除后引用式(4)。其他16个编号公式内容逐字保持，评分和预测含义未改。18个公式均有正确编号。

## 公开方法、文本接口及双输出

- 11篇原始文献的作者、题名、venue、年份、页码及DOI分别记录于 `q2_reference_audit.md`。M3S对应AACL 2022缺失元采样原论文，本地为低秩骨干和一步一阶元更新。
- 表10依据真实适配机制重写，参数量不变；原论文机制与本地近似/代理实现的区别写入文献核对文件，不把适配称为原仓库完整复现。
- `text_bert`单样本为3×50词元编号/有效掩码/类型编号。有效词元前缀经BERT最后隐层编码，后缀补零；不把整数词元输入称为768维特征。
- 604行检查实际来自20条无标签附件4样本，不是附件2 valid；分类输出差是logits差，不是Softmax概率差。检查沿用已有 `attachment3_text_interface_audit.json`，本轮未读取或运行附件4。
- 分类头与回归头分别输出，未按回归符号修正类别。30条预测类别、强度、置信度保持，前两项与预测CSV核对，置信度与predictions_audit.csv核对。

## 阅读与版面自检

叙事顺序仍为问题与数据→均值基线→内容残差→缺失评价→结构比较→缺失规律→稳定性与消融→公开方法→无标签交付。新增内容集中在评分依据、实验目的及接口解释，未扩展baseline百科或重复逐项表格数字。

未将宏平均F1改善写成类别不平衡已解决，未将小幅平均收益写成全面稳定提升。原生视觉全零子集n=15保留其局限；均值—最大值单初始化不与三种子细作统计比较。

逐页查看全部15页渲染：正文14页，参考文献单独1页；表10方法列不越界，公式引入/后续解释连续，正常新段落首行2em，原图件未替换。图8–14、表8–13编号不变。最终日志无overfull、undefined reference/citation、缺字或LaTeX错误。
