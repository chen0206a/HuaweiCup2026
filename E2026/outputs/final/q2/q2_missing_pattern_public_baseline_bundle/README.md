# Q2 缺失规律与公开 Baseline 结果包

此包根据 E2026 本地锁定 validation 结果生成。没有训练模型、加载 checkpoint 做推理、使用 test/Attachment3/4，或修改论文正文。

## 核心交付

- `fig_q2_modality_missing_degradation.png/.pdf/.svg`：P2 在文本/音频/视觉单模态缺失下的四指标退化曲线。
- `q2_modality_missing_degradation_data.csv`：60 个模态 × rho × 指标统计单元，含三 seed 数值及 mean/sample SD。
- `q2_modality_missing_scenarios_p2.csv`：对应的 135 条位置级记录。
- `q2_public_baseline_missing_comparison.csv/.tex`：TFN、MulT、MISA 与本文 P2 在固定 54 个缺失场景上的描述性比较。
- `q2_public_baseline_missing_scenarios_per_seed.csv`：486 条公开模型 × seed × 场景明细。
- 两份审计、完整预检报告、checkpoint/benchmark 来源清单及逐 seed baseline 结果 JSON。

## 统计规则

对每个 seed × 模态 × rho，先对 early/middle/late 等权平均；再以 seed 为独立重复计算 mean 和 sample SD（ddof=1，n=3）。Baseline 对每 seed 的 54 个场景求均值后，同样按 3 seed 统计。图中阴影表示 seed 间样本 SD。

## 论文表述边界

公开模型按 clean validation 选 checkpoint，P2 按冻结 robust score 选 checkpoint；baseline 比较仅为描述性结果，不支持严格公平的鲁棒性排名或“显著优于”的结论。完整输入 clean参考来自锁定 P2 clean validation。

## 复核

- `DATA_PREFLIGHT_REPORT.md`：两项完整性门槛。
- `q2_modality_missing_degradation_audit.md`：曲线来源、聚合、端点变化与图注。
- `q2_public_baseline_missing_audit.md`：baseline 计数、来源、统计单位与选模限制。
- `source_manifest.json`：源路径、文件大小、SHA256 与数据边界。
