# Q1 V1-P 结构分层验证报告

本报告由 `src/q1/validate_v1_partition.py` 从官方 evaluator 结果生成。
V1-P 的拓扑模式、K/2K/4K/8K、切点权重和四候选预算均保持冻结。
选样使用 100 图的结构特征与官方 V0 四核结果；此前用于 V1-lite 开发的
case_014、case_025、case_026、case_093 被排除，以保留独立验证集。

## 样本

12 个 case，按非 COPY Op 数三分层，每层通过七维特征分位距离选择 4 个。
为缩短收口时间，仅从预筛查官方计算耗时不超过 60 秒的样本中选择；
因此结论不覆盖最昂贵的超大图。
七维特征包括 Op 数、连通分量数、最大分量工作量占比、关键路径/总工作量、
Tensor 扇出 P95、V0 四核新增 COPY 和 V0 四核加速比。逐例原因见
`results/q1_v1_validation/selection.csv`。

| 规模 | case |
|---|---|
| small | case_001, case_071, case_078, case_061 |
| medium | case_074, case_007, case_002, case_088 |
| large | case_089, case_024, case_067, case_020 |

## 总体

共 48 个 case×K 组合；mean improvement 20.90%，
median 7.50%，P25/P75 0.00%/49.26%。
改善 62.50%，退化 12.50%，不退化 87.50%，超过 5% 的严重退化 6.25%。
最大改善 79.08%，最大退化 -30.58%。
added COPY 平均变化 -356,983 bytes，中位变化 -23,318 bytes。

| 分组 | n | 平均改善 | 中位改善 | 改善比例 | 退化比例 | 平均新增 COPY 变化 |
|---|---:|---:|---:|---:|---:|---:|
| K=2 | 12 | 14.47% | 4.12% | 58.33% | 16.67% | -305,368 |
| K=3 | 12 | 19.40% | 5.42% | 66.67% | 16.67% | 172,654 |
| K=4 | 12 | 25.39% | 4.38% | 58.33% | 8.33% | -17,198 |
| K=5 | 12 | 24.35% | 8.15% | 66.67% | 8.33% | -1,278,021 |
| size=small | 16 | 18.76% | 5.77% | 56.25% | 18.75% | -186,099 |
| size=medium | 16 | 27.35% | 13.34% | 75.00% | 12.50% | -276,950 |
| size=large | 16 | 16.59% | 0.03% | 56.25% | 6.25% | -607,900 |
| components=few | 16 | 3.01% | 0.05% | 62.50% | 31.25% | -159,568 |
| components=middle | 16 | 34.44% | 30.61% | 75.00% | 6.25% | -667,308 |
| components=many | 16 | 25.25% | 1.66% | 50.00% | 0.00% | -244,073 |

## 最严重的五个退化组合

宽度为块 DAG 同层最大块数，是简单并行度指标；crossing bytes 每个跨块 Tensor 只计一次。

| case/K | 改善率 | V0/V1 模式 | 块数 | 分量数/最大工作量占比 | 关键路径/总工作量 | crossing bytes | added COPY | 宽度 | 官方 Makespan | 诊断 |
|---|---:|---|---|---|---|---|---|---|---|---|
| case_002/K2 | -30.58% | deterministic/deterministic | 16/4 | 1/100.00% | 2976/0.0108 | 313,344/233,472 | 672,768/466,944 | 3/3 | 154017/201114 | A coarse cut structure (fewer blocks; check load and pipe overlap) |
| case_002/K4 | -16.78% | deterministic/deterministic | 32/4 | 1/100.00% | 2976/0.0108 | 347,136/233,472 | 800,256/466,944 | 3/3 | 115766/135196 | A coarse cut structure (fewer blocks; check load and pipe overlap) |
| case_078/K2 | -8.96% | deterministic/deterministic | 8/4 | 8/12.50% | 246816/0.1153 | 0/0 | 15,039,360/13,146,912 | 8/4 | 1080972/1177840 | A coarse cut structure (fewer blocks; check load and pipe overlap) |
| case_078/K5 | -1.80% | deterministic/deterministic | 40/40 | 8/12.50% | 246816/0.1153 | 430,080/369,408 | 15,899,520/15,778,176 | 8/8 | 555249/565216 | E other/needs trace review |
| case_089/K3 | -0.55% | deterministic/deterministic | 12/24 | 40/2.50% | 16272/0.0233 | 65,536/57,344 | 2,756,992/5,605,248 | 4/17 | 236718/238029 | C spill/memory pressure (inference from added COPY versus crossing bytes) |

所有超过 5% 的退化组合及其诊断：

- case_002 K=2: -30.58%；A coarse cut structure (fewer blocks; check load and pipe overlap)
- case_002 K=4: -16.78%；A coarse cut structure (fewer blocks; check load and pipe overlap)
- case_078 K=2: -8.96%；A coarse cut structure (fewer blocks; check load and pipe overlap)

## 运行成本与决策

验证阶段真实 evaluator 调用 260 次，累计 evaluator 耗时 380.9 秒；V0/V1-P 总墙钟分别 186.7/326.3 秒。
筛选阶段 evaluator 与单核官方计算合计 5887.4 秒。
缓存命中不计为新的 evaluator 调用；候选数仍逐行记录。

三项量化门槛全部通过。
严重退化比例为 6.25%，集中在 2/12 个 case。
收口判断把严重退化达到 10% 的组合视为普遍；此口径只用于本轮是否停止优化，
不是预注册的性能统计门槛。
建议：冻结 Q1 的当前 V1-P 实现并停止优化；保留 V0 作为逐例对照。
不运行全 100 例 V1-P；不实施 memory-aware cut。
