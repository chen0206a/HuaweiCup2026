# Current Context

日期：2026-09-24。当前选题为 E（D005），E 代码与结果位于 `E2026/`。根目录 `q1/`、`q2/`、`q3/` 属于历史 A 题，不可与 E 的问题编号混用。

## E Q2

用户已正式锁定 **B0-WCE 为 baseline，B5-P2 attention residual pooling 为最终主模型**。不再进行 Q2 模型探索、调参或组合。冻结验证协议为 attachment2 validation clean + 54 个连续缺失场景，benchmark seed 20260923，SHA256 `3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff`。

B0 三 seed robust `0.741677 ± 0.001458`；P2 `0.744777 ± 0.001492`；paired delta `+0.003100 ± 0.002907`（sample SD）。seed42/43 为正，seed44 基本持平，结果存在 seed 敏感性。依据：`E2026/outputs/metrics/b5_p2_multiseed_summary.json`。最终配置、数据与 checkpoint manifest、实验索引见 `E2026/configs/final/`、`E2026/data/manifests/`、`E2026/outputs/final/q2/`。

本地 attachment2 原始文件实际位于 `E2026/data/raw/`；历史服务器配置使用 `data/raw/attachment2/`，该差异已记录，原始大文件未移动。六个 B0/P2 主 checkpoint 均在本地并完成哈希索引。Attachment3 仅做文件名、大小与文件级 SHA256 清单，状态 **SEALED**；本阶段未反序列化或推理。Attachment2 test 只用于结构/标签完整性检查，没有参与本阶段模型选择。

B5-P1 只有 seed42 screening；B4′ 仍为单 seed 诊断；fusion、reconstruction、loss 调整和文本 z-score 路线均未进入最终模型。详细原因以既有实验结果和 `E2026/outputs/final/q2/q2_experiment_index.json` 为准。Q1审查见下节；Q3 的 E 题建模进度未核实，本轮没有启动 Q3。

## E Q1/Q2 审查（2026-09-24）

审查报告：`notes/E_Q1_Q2_REVIEW_20260924.md`，复算证据：`notes/e_q1_q2_audit_evidence_20260924.json`。Q1实际交付目录为未跟踪的`Q1_方法流程与实验结果_交付包/`，包含方法与汇总，但不含全量特征、100条timeline和部分复现依赖。Q1方法及辅助实验可收口，当前包不能作为完整提交包；25/25辅助CV折存在同video_id跨训练/测试，结论限于诊断。Q1附件1与Q2附件2/3的数据分工符合官方要求，不需跨问特征衔接。

Q2六checkpoint/benchmark哈希及冻结B0张量核验通过，主要汇总数值复算一致；附件2官方三split的video_id无交集。P2缺失绝对综合分数较高，但相对clean的降幅大于B0，不能声称退化更小或各指标全面提高。已有同clean选模记录支持平均paired robust增量0.003046，无新训练。正式交付仍需附件3预测CSV及可执行推理入口；本轮未解封附件3、未改变D006或验证协议。

## Historical A work

A 题快照保存于 `archive/A_CONTEXT_before_E_20260923.md` 和 `archive/A_STATUS_before_E_20260923.md`。
