# 全量结果表整合审计

审计日期：2026-09-25。当前编译产物为 `build/main.pdf`，A4，共26页。本轮只将既有结果排入论文；没有重跑特征提取、模型训练或推理。四组新增表由 `scripts/generate_full_result_tables.py` 从冻结源文件生成，源路径与SHA-256记录在 `generated/full_result_table_sources.json`。

| 新增表 | PDF页码 | 行数 | 主要数据源 | ID核对 |
|---|---:|---:|---|---|
| 表2：附件1全量特征构建 | 7—9 | 100 | `../outputs/q1_final_local/05_appendix/q1_sample_summary_100.csv`；`../outputs/q1_final_local/00_manifest/q1_paper_numbers.json` | 100个唯一ID；与论文目录中的同名CSV逐字节相同；未缺行 |
| 表8：公开模型完整输入对比 | 16 | 4 | `E2026/outputs/final/q2/public_baselines/baseline_clean_mean_sd.csv`、`baseline_clean_per_seed.csv`；`E2026/outputs/metrics/b5_p2_multiseed_summary.json` | TFN、MulT、MISA、本文模型各一行；三seed参数量及均值/样本标准差已核对 |
| 表9：附件3最终预测 | 16—17 | 30 | `E2026/outputs/final/q2/attachment3/attachment3_predictions.csv`、`attachment3_predictions_audit.csv`、`attachment3_summary.json`、`attachment3_input_audit.json` | 30个唯一ID与30个输入文件名一一对应；无缺失ID |
| 表12：附件4最终预测与解释 | 21 | 20 | `E2026/outputs/q3/final/attachment4_predictions_explanations.csv`、`attachment4_summary.json` | 20个唯一ID与最终输入清单完全一致；无缺失ID |

## 数值与口径检查

- Q1 manifest规定100条样本、对齐长度50、三模态维数均768。表2包含全部100行；视觉有效窗数全为50，文本来源为官方75条、媒体ASR 25条。逐样本时间戳回退次数合计21次、分布在7条样本，相关样本均保留。平均有效窗覆盖率从原始CSV复核为文本0.8242、语音0.9910、视觉1.0000，与manifest一致。manifest只提供汇总计数，没有逐样本ID名单；ID级核对以最终100条CSV及其论文副本为准。
- 表8的TFN、MulT、MISA均值与样本标准差已从各自42/43/44三seed逐项复核，参数量在各seed内一致。本文模型的四指标和164,343个参数来自锁定的P2三seed汇总；不与历史B0行混用。该表只报告附件2完整输入验证集，不把公开方法的aligned-50架构适配说成原训练流程复现。
- 表9与正式预测CSV的类别、情感强度逐行一致；置信度来自同批预测审计CSV。分类计数为消极7、中性11、积极12；情感强度均值0.123114、样本标准差0.611792、最小值-1.642545、最大值1.539758；平均置信度0.624599，均与最终summary一致。文本接口报告状态为PASS，20条已知样本的604/604有效行及锁定模型输出差已写入正文。
- 表12的类别计数为消极7、中性5、积极8；分类主导文本19、语音0、视觉1，回归主导文本18、语音0、视觉2，两头主导一致17/20；证据已核验19、未核验1，均与最终summary一致。关键区间统一标作零起点的特征槽位$[a,b)$；视觉主导样本不写秒数或帧号。
- 表内Q1时长显示三位小数，附件3/4连续强度显示六位小数、置信度显示四位小数；公开模型四项指标显示四位小数。上述仅为排版舍入，生成表时直接读取原始数值。
- 附件3和附件4都没有真实情感标签。新增小节只报告预测分布与解释，不报告或暗示Accuracy、F1、MAE、Pearson等无标签性能。

## LaTeX与未完成项

`build_paper.ps1`编译成功；日志无未定义引用/文献、缺字或overfull hbox。表号由LaTeX自动分配，新增表2、8、9、12的交叉引用解析正确。原图1—12仍按序存在，图件没有替换或重绘。

渲染检查覆盖表2的第7—9页、表8及表9的第16—17页、表12的第21页。PDF文本提取后，100个Q1样本ID和30个附件3样本ID均可检索到；表12视觉检查包含01—20全部样本。长表续页表头及表号正常。

结果类占位仍有Q2图9（三seed配对及误差分析）。Q1图2—4仍为方法示意图占位。题目、摘要、关键词及一条Shapley文献核验待办属于后续定稿事项；本轮没有补写。
