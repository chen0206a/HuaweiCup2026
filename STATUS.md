# Project Status

2026-09-24：当前 E 题工作目录为 `E2026/`。

- E Q1/Q2：完成官方要求、方法与结果审查，见`notes/E_Q1_Q2_REVIEW_20260924.md`。Q1方法/辅助实验可收口，当前摘录交付包缺全量特征和timeline；本地测试10通过、1因缺100条timeline失败。Q2相关测试28通过、2因无CUDA跳过；六checkpoint及benchmark哈希、冻结B0张量、已有主要指标复算通过。不建议继续模型探索，优先补全量交付、正文图表和最终附件3推理入口；本轮未训练、未解封附件3。两问使用不同官方附件是正确安排，无需跨问特征对接。

- E Q2：用户正式锁定 B0-WCE 为 baseline，B5-P2 attention residual pooling 为最终主模型，停止后续 Q2 模型探索。六个 seed42/43/44 的 B0/P2 主 checkpoint 已在本地验证；attachment2 aligned-50 本地副本的 split、形状、标签与冻结审计一致。已生成最终 configs、data/benchmark/checkpoint manifests 与实验索引：`E2026/outputs/final/q2/`。冻结 benchmark 哈希匹配；attachment3 保持 SEALED，attachment2 test 未用于本阶段选模。原始本地文件在 `E2026/data/raw/`，与历史服务器 `data/raw/attachment2/` 路径不同，未移动大文件。

- E Q2：B5-N1 seed42 text-only train-stat feature-wise z-score finished. Historical N0 B0-WCE checkpoint reproduced exactly on clean + frozen 54-scenario validation. N1 trained the complete original B0 from matching seed42 initialization; best-clean and best-robust both selected epoch 1. N1 robust 0.734474 vs N0 0.740248 (Δ −0.005774); accuracy and overall score fell despite higher Neutral recall/F1. Stop normalization route; no P2 combination or additional normalization runs.
- Text scaler fit on 83,672 valid train timesteps for 768 text dimensions, epsilon 1e-6; zero/nearly-zero variance dimensions 0/0. Float32 normalized train feature means max abs 6.2e-8, nondegenerate std max deviation from 1 6.8e-8. Synthetic missing is overwritten to exact zero after normalization; unchanged benchmark SHA256.
- CPU/CUDA forward/backward, initial-state identity, N0 reproduction, and both checkpoint round-trips passed. Three scaler tests pass. The aligned pickle is monolithic and deserialized as a container, but test is not indexed into a Dataset, evaluated, or used; attachment3 not accessed.
- Outputs and formal record: `E2026/outputs/metrics/b5_n1_text_zscore_report.md`, metrics/scaler-state JSON, `experiments/exp_013_b5_n1_text_zscore/`.

- E Q2：B5-P2 attention residual pooling seed43/44 paired replication completed, reusing seed42. Seed-specific B0 checkpoints were used; B0 tensors remained frozen and passed exact post-training equality checks. Validation was clean + fixed 54 missing scenarios only; benchmark SHA256 unchanged.
- Paired robust deltas P2−B0: seed42 +0.005600, seed43 +0.003791, seed44 −0.000090; mean +0.003100 ± 0.002907 sample SD. Two of three seeds are positive, with seed44 effectively neutral. Treat as seed-sensitive candidate, not stable universal improvement.
- Results: `E2026/outputs/metrics/b5_p2_multiseed_report.md`, machine-readable `b5_p2_multiseed_summary.json`, formal record `experiments/exp_012_b5_p2_multiseed/`.
- No z-score or module combination started. Test/attachment3 excluded.

- E Q2：完成 B5-H0 只读诊断。历史 B0-WCE seed42 checkpoint 未修改；valid clean + 54场景，benchmark SHA256不变。
- Head一致性：clean/missing agreement 82.69%/81.62%；both-wrong约30.3%；reg-only-correct约5.1%/5.5%。Oracle union上限约69.6%/69.7%，仅诊断。结论：部分互补但共享错误明显，尤其Neutral不能由tau=0回归符号映射恢复。
- 输出：`E2026/outputs/metrics/b5_h0_head_diagnostic.{json,md}`、`b5_h0_predictions.{csv,jsonl}`；正式记录 `experiments/exp_011_b5_h0_head_diagnostic/`。
- 未训练模型、未修改checkpoint、未索引/评估attachment2 test、未访问attachment3；不自动进入H1。
- Q1/Q3完成情况本轮未核实；A题历史资料保留。
