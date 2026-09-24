# 本地原始数据

当前本地副本为 `aligned_50.pkl` 和 `label.xlsx`，位于本目录并由 Git 忽略。数据哈希、字段、标签、split 和 ID 核验结果见 `../manifests/attachment2_manifest.json`；文件仅供读取，不改写。

历史 GPU 服务器配置中的路径为 `data/raw/attachment2/`。本地复制时放在了 `data/raw/`，本次资产锁定记录实际路径，不移动近 1 GB 的 pickle，也不改变服务器数据路径。最终配置中的相对路径以 `E2026/` 为基准。

Attachment3 保持 **SEALED**：只允许文件名、大小和文件级 SHA256 清单；本阶段没有加载、统计其内容或推理。Attachment4 当前只建立文件元数据清单，没有开始 Q3 建模。
