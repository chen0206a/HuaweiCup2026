# Q2 剩余实验状态（本轮结束）

ATTACHMENT3_FINAL_INFERENCE = PASS

PUBLIC_BASELINE_TFN = NOT_STARTED

PUBLIC_BASELINE_MULT = NOT_STARTED

PUBLIC_BASELINE_MISA_OR_SELFMM = NOT_STARTED

BASELINE_3SEED_COMPLETE = NO

BASELINE_MISSING_BENCHMARK_COMPLETE = NO

本轮按用户最新指示仅完成本地附件3文本接口适配验证与最终推理。公开 baseline 未开始，未上传数据或训练到新服务器。新服务器 SSH 端口 49826 可连接，GPU 为 RTX 3090；预检查时 `/root/workspace` 为空，后续 baseline 需要先部署项目、附件2与运行依赖。

附件3正式结果：`outputs/final/q2/attachment3/attachment3_predictions.csv`（30行）及审计、摘要、接口回归与交付检查文件。没有标签性能指标；Q2 checkpoint 不变。下一步最小工作是下一轮独立启动公开 baseline 的数据与源码适配，不再占用本轮推理结果作任何调参。

本地推理开始时 Git HEAD：`0788d28abf1d584db82554351af1f343bbb30df6`；最终当前 commit 请以 `git rev-parse HEAD` 为准。仓库既有的未跟踪论文/绘图资产未改动。

是否需要继续占用服务器：本轮否；公开 baseline 下一轮需要。
