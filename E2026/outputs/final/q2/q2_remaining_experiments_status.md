# Q2 剩余实验状态（本轮）

ATTACHMENT3_FINAL_INFERENCE = FAIL（输入接口缺少预计算 `text`；无预测输出）

PUBLIC_BASELINE_TFN = NOT_STARTED

PUBLIC_BASELINE_MULT = NOT_STARTED

PUBLIC_BASELINE_MISA_OR_SELFMM = NOT_STARTED

BASELINE_3SEED_COMPLETE = NO

BASELINE_MISSING_BENCHMARK_COMPLETE = NO

本轮按用户最新指示仅处理本地附件3最终推理；公开 baseline 留待下一轮在新 GPU 服务器执行。新服务器 SSH 端口 49826 可连接，GPU 为 RTX 3090，Python 3.12.3，`/root/workspace` 为空，项目、附件2数据和 checkpoint 尚未部署。无服务器训练发生。

下一步最小工作：获取与附件2预计算文本特征同一流程的附件3 `text[30,50,768]`，或核准并验证固定 `text_bert`→`text` 转换；然后重跑输入审计、执行一次锁定 checkpoint 推理及交付 QA。公开 baseline 的服务器环境与项目部署在下一轮单独处理。

审计开始时本地 Git HEAD：`edab77a852d4b6abdabfbe73749493c656538a56`；最终当前 commit 请以 `git rev-parse HEAD` 为准。仓库另有此前未跟踪的论文/绘图资产，本轮未改动。

本轮新增文件：

- `scripts/audit_attachment3_interface.py`
- `outputs/final/q2/attachment3/attachment3_input_audit.json`
- `outputs/final/q2/attachment3/attachment3_input_audit.md`
- `outputs/final/q2/attachment3/attachment3_delivery_check.md`
- `outputs/final/q2/q2_remaining_experiments_status.md`

是否需要继续占用服务器：否；本轮已停止。
