# E2026 本地资产与服务器依赖审计

审计日期：2026-09-24
本地项目：`D:/华为杯/E2026`
历史服务器根目录：`/root/workspace/E2026`

## 审计范围

本轮只核对现存文件名、目录、manifest、CSV/JSON 结果和哈希；没有训练、推理、重新评估、访问 Attachment3/4 特征内容或改动实验结果。Attachment4 内容检查沿用已保存的清单与文件级哈希审计结论。

## 结论一览

- `FIG4_DATA_COMPLETE = NO`：总模态聚合曲线有完整三种子数据，但按模态拆分的 B0 seed43/44 缺失。
- `FIG5_DATA_COMPLETE = NO`：按模态位置的 B0 seed43/44 缺失；双模态位置逐条件结果也缺 B0 seed43/44。
- `Q2_LOCAL_COMPLETE = NO`：关键输入、锁定 checkpoint、代码、配置及汇总结果在本机；逐场景结果尚缺 B0 seed43/44 的 110 个（模型×种子×场景）记录。
- `Q3_LOCAL_COMPLETE = YES`：锁定方法、验证产物、20 个 Attachment4 特征/视频配对、最终输出及 Figure 8/9 数据和图均有本地文件与清单支持。
- `Q2_SERVER_ONLY_ITEMS = []`：未发现必须从服务器取回才能完成锁定 Q2 结果的文件。两个未落地的 P2 seed43/44 best-clean checkpoint 是未锁定的辅助检查点；其远端是否仍存不在本地审计范围内。
- `Q3_SERVER_ONLY_ITEMS = []`。
- `RUNTIME_REMOTE_DEPENDENCIES = [旧版 Q2 脚本存在 /root/workspace/E2026 默认路径；可通过本地参数/路径运行，不是当前 Q3 推理或绘图的强制远端依赖]`。
- `SERVER_CAN_SHUT_DOWN = YES`。

判断依据：本机已有所有锁定的 Q2/Q3 checkpoint、Attachment2 输入、冻结缺失场景定义、评估代码，以及已经保存的 Q2/Q3 最终产物。缺失的 B0 seed43/44 场景评估能够在本机完成。没有发现当前最终流程会访问服务器的代码路径。未尝试连接服务器核验其额外的非必要副本。

## A. Q2 Figure 4 / Figure 5 数据审计

`fig4_all_modality_rho_per_seed.csv` 覆盖 2 个模型×3 个种子×5 个 rho×5 个指标（150/150 行）；相应聚合文件 50/50 行且每格 `n_seeds=3`。这些文件不按模态拆分。真正按模态拆分的数据表只覆盖 B0 seed42 与 P2 seed42/43/44。

`fig5_all_modality_location_per_seed.csv` 覆盖总模态聚合的 2×3×3×5（90/90 行）；真正按模态拆分的位置表只覆盖 B0 seed42 与 P2 seed42/43/44。场景明细只包含 B0 seed42、P2 seed42/43/44 四组，每组有 55 行（clean + 54 个冻结缺失场景）。没有在本地其他 CSV/JSON/JSONL 结果中找到 B0 seed43/44 的 Q2 missing-benchmark per-scenario 输出。

| figure | required_item | local_path | exists | complete | seeds | scenario_count | metrics | usable_for_plot | notes |
|---|---|---|---|---|---|---:|---|---|---|
| Figure 4 | 总模态 rho 曲线，逐种子 | `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig4_all_modality_rho_per_seed.csv` | YES | YES | 42/43/44 | rho=0.1–0.5 | accuracy, macro-F1, MAE, Pearson, selection score | YES（聚合模态） | 150 行；不提供模态拆分 |
| Figure 4 | 模态×rho，逐种子 | `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig4_modality_by_rho_available.csv` | YES | NO | B0:42；P2:42/43/44 | 45/45 单模态条件/模型种子组合 | accuracy, macro-F1, MAE, Pearson, selection score | 部分 | 60/90 组合；缺 B0 seed43/44，各缺 text/audio/vision×5 rho，共 30 组合；无完整模态级三种子 SD |
| Figure 4 | 模态级均值/标准差 | `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig4_all_modality_rho_mean_sd.csv` | YES | NO（按模态要求） | 42/43/44 | rho=0.1–0.5 | 5 项 | 部分 | 50 行均值/SD是总模态聚合，不能替代模态级统计 |
| Figure 5 | 总模态 location 曲线，逐种子 | `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig5_all_modality_location_per_seed.csv` | YES | YES | 42/43/44 | early/middle/late | accuracy, macro-F1, MAE, Pearson, selection score | YES（聚合模态） | 90 行；不提供模态拆分或 TA/TV/AV 拆分 |
| Figure 5 | 单模态×位置，逐种子 | `outputs/final/q2/q2_plotting_handoff_v2/plot_data/fig5_modality_by_location_available.csv` | YES | NO | B0:42；P2:42/43/44 | 27/27 单模态条件/模型种子组合 | accuracy, macro-F1, MAE, Pearson, selection score | 部分 | 36/54 组合；缺 B0 seed43/44，各缺 3 模态×3 位置，共 18 组合 |
| Figure 5 | 双模态 TA/TV/AV×位置 | `outputs/final/q2/q2_plotting_handoff_v2/plot_data/scenario_details_available.csv` | YES | NO | B0:42；P2:42/43/44 | 36/54 双模态组合 | 5 项指标（宽表） | 部分 | 当前明细共 220/330 行；缺 B0 seed43/44 各 55 条，含双模态条件 |

完整缺口：`B0 seed43 per-scenario (55 conditions)`、`B0 seed44 per-scenario (55 conditions)`。已存在的总模态汇总无法填补按模态拆分的空缺。

**`FIG4_DATA_COMPLETE = NO`**
**`FIG5_DATA_COMPLETE = NO`**

## B. Q2 本地化

| 项目 | 本地检查 | 结果 |
|---|---|---|
| Attachment2 aligned-50 与标签 | `data/raw/aligned_50.pkl`（993,842,861 bytes）、`data/raw/label.xlsx`（423,826 bytes）；manifest 记录 train/valid/test、ID/标签交叉核对、split 无重叠 | 已有；训练/验证可读取。未使用 test 做本审计 |
| B0/P2 锁定 checkpoint | `outputs/final/q2/q2_checkpoint_manifest.json` 指定 6 个 checkpoint | 6/6 存在；本轮逐个重算 SHA256，全部匹配 manifest |
| Q2 最终配置与锁 | `configs/final/q2_b0_wce.yaml`、`configs/final/q2_b5_p2.yaml`、`outputs/final/q2/q2_model_lock.{md,json}` | 已有 |
| missing benchmark 定义 | `outputs/metrics/b2_benchmark_definition.json`、`data/manifests/q2_missing_benchmark_manifest.json` | 已有；clean + 45 单模态 + 9 双模态 = 55 条件，seed 20260923 |
| 结果与消融 | `outputs/metrics/`、`outputs/final/q2/q2_plotting_handoff_v2/source_results/` | clean 三种子、missing 汇总、robust score、对比/消融结果文件已有 |
| 逐条件结果 | `outputs/final/q2/q2_plotting_handoff_v2/plot_data/scenario_details_available.csv` | 部分；B0 seed43/44 缺失，220/330 行 |
| Attachment3 prediction | `data/manifests/attachment3_sealed_inventory.json` | Attachment3 仍 SEALED；未发现已完成预测交付件；不属于当前锁定结果的依赖 |
| 模型、评估和绘图代码 | `src/models/baseline.py`、`src/models/pooling_residual.py`、`src/evaluation/missing_benchmark.py`、`src/training/evaluate.py`、Q2 plotting handoff bundle | 已有；当前完整 Figure 4/5 模态逐种子图尚需本地评估补齐 |
| 辅助 P2 best-clean checkpoints | manifest `p2_best_clean_checkpoints_retained_reference` | seed42 在本机；seed43/44 引用路径不存在于本机。不是 Q2 锁定的 best-robust checkpoint，不阻止复现锁定论文结果；远端现存状态未知 |

**`Q2_LOCAL_COMPLETE = NO`**（逐场景结果未完整本地化）。
本地缺少的已知关键结果可用本地六个锁定 checkpoint、Attachment2 数据、54 个缺失场景定义及 `src/evaluation/missing_benchmark.py` 补做，不需服务器。
**`Q2_SERVER_ONLY_ITEMS = []`**（没有证据表明缺失的必要结果只存在服务器）。

## C. Q3 本地化

| 项目 | 本地路径/证据 | 结果 |
|---|---|---|
| Frozen P2 seed42 | `outputs/checkpoints/b5_pooling_p2_best_robust_score.pt` | 存在；SHA256 与 Q2/Q3 锁定值一致：`cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff` |
| Attachment2 validation 与 HEAF | `data/raw/aligned_50.pkl`、`experiments/q3/exp_001_heaf_validation/`、`outputs/q3/heaf_validation_metrics.json` | 已有；设计/审计 ID 与 video_id 不重叠，ID 清单保存在 `validation_ids.json`；rho=.30 及协议锁定 |
| HEAF 诊断结果 | `audit_sample_diagnostics.jsonl`、`design_window_scores.jsonl`、`seed43_comparison.jsonl`、`seed44_comparison.jsonl`、`metrics.json` | 已有 |
| Attachment4 原始配对 | `data/manifests/q3/attachment4_inventory.json` 与 `outputs/q3/q3_grounding_audit.json` | 20 个 aligned pkl + 20 个 paired mp4；ID 20/20 精确匹配、重复/歧义 0；已保存 40 个文件级 SHA256 并匹配。媒体位于项目目录旁的本机 D: 盘路径，不在 E2026 子目录中；manifest 有相对路径 |
| 无标签 Adapter 与 Grounding | `src/q3/data_adapter.py`、`src/q3/evidence_grounding.py`、`src/q3/text_grounding.py` | 已有；现有审计为文件/特征接口校验；本轮未重新读/反序列化 Attachment4 特征 |
| HEAF 代码 | `src/q3/frozen_predictor.py`、`coalitions.py`、`temporal_occlusion.py`、`faithfulness.py` 等 | 已有；最终导出由 `scripts/run_q3_attachment4_final.py` 集成完成，没有单独的 `src/q3/export.py` 文件 |
| 方法配置、锁与 schema | `configs/final/q3_heaf.yaml`、`q3_heaf_validation.yaml`、`outputs/q3/q3_method_lock.json`、`q3_explanation_schema.json` | 已有 |
| 最终输出 | `outputs/q3/final/attachment4_predictions_explanations.csv`、`attachment4_explanations.jsonl`、`attachment4_summary.json`、`attachment4_delivery_check.md` | 已有；交付 QA 记载覆盖 20/20、schema 通过、Shapley efficiency 误差 < 9e-16；A/V 时间字段保持 null/NA |
| Figure 8/9 数据与图 | `outputs/final/q3/figures/data/`、`scripts/` 与 figure 目录 | Figure8 deletion/margin/modality CSV、Figure9 四张案例卡、视频帧 manifest、绘图脚本及最终 PNG/PDF/SVG 均有 |
| tokenizer / BERT 资产 | Hugging Face cache `models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594`；实验记录 `exp_0026_text_row_identity/config.yaml` | tokenizer 与模型资产本地缓存；tokenizer revision 固定。历史 Attachment4 特征提取器的确切 BERT checkpoint revision 未被历史 preprocessing 固定，但不影响已导出的解释文件复核；该限制已在文本行身份报告记录 |

**`Q3_LOCAL_COMPLETE = YES`**
**`Q3_SERVER_ONLY_ITEMS = []`**

## D. 服务器路径搜索

在本地仓库代码、配置、清单、结果和文档中搜索 `/root/workspace/E2026`、`/root/`、`/workspace/`、`scp`、`rsync`、`ssh`、`remote`、`server`。

发现旧版 Q2 脚本在运行默认配置时使用历史路径，例如：

- `src/training/run_b2.py` 的 `--pkl-path` 默认 `/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl`；
- `scripts/smoke_b0.py`、`smoke_b1.py`、`overfit32.py`、`overfit32_b1.py` 的 `--pkl` 默认相同历史目录；
- `src/training/run_b32.py` 含历史 checkpoint 输出根路径。

这些是旧脚本参数/输出默认值，不含 SSH/SCP/rsync 或远程 API 调用。Q3 final runner 使用 `Path(__file__)` 定位项目根和本地 manifests/checkpoint。Q2 本地补评估时可传入 `data/raw/aligned_50.pkl` 等本机路径。因此：

**`RUNTIME_REMOTE_DEPENDENCIES = [旧 Q2 脚本的历史绝对路径默认值；可本地覆盖，不是当前最终产物流程的强制依赖]`**

## E. 本地有结果、服务器独占原始材料检查

- B0 seed43/44 checkpoint：本地存在，hash 与主 checkpoint manifest 一致。
- Attachment2 train/valid：本地同一 `aligned_50.pkl`，manifest 校验 split 与 ID；可供本地评估使用。
- 冻结 missing benchmark：本地定义为 clean + 54 missing conditions，评估实现已落地；不需服务器恢复场景定义。
- Q2 per-scenario：部分缺失（B0 seed43/44），但缺少的是可由本地输入和 checkpoint 重建的评估输出，不是服务器独占的原始数据。
- Attachment4：20 pkl/20 mp4 文件级清单和 hash 已记录，20/20 ID 对应；本机可访问。
- Q3 final checkpoint、解释 CSV/JSONL/summary/delivery：本机存在。
- BERT/tokenizer：当前 pinned tokenizer 与 revision 缓存在本机；历史特征抽取所用 checkpoint revision 的确切 provenance 有限，记录中明确说明。

没有已识别出必须下载才能继续或复核锁定结果的关键文件。

## F. 服务器关闭判断

**`SERVER_CAN_SHUT_DOWN = YES`**

Q2 Figure4/5 模态级三种子结果尚需补齐，但本地具备六个主 checkpoint、Attachment2 validation、冻结的 54 个缺失条件定义和 evaluator；其余锁定 Q2/Q3 产物已在本机。已知历史服务器路径仅是若干旧脚本的可覆盖默认值，没有发现当前 Q3 或绘图流程需要在线访问服务器。服务器上可能另有不必要的副本，但没有证据表明其保存了本机无法恢复的必要结果。无需生成 `SERVER_DOWNLOAD_TODO.md`。

**自然语言结论：服务器现在可以关闭。Figure 4/5 仍缺 B0 seed43/44 的按条件评估结果，但所需数据、checkpoint、场景定义和代码均在本机，可以之后本地补齐；没有需要先从服务器下载的文件。**
