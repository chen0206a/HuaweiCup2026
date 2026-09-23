# Confirmed Decisions

> 只记录已确认且全员需要遵守的事项。普通讨论放在相应任务记录中。

## D001 - Repository and collaboration structure
决定：按当前仓库目录组织四个小问；使用 `CONTEXT.md` 作为当前快照、`DECISIONS.md` 作为全局决定、各 `qX/HANDOFF.md` 作为小问交接。
原因：三个独立账号需要共享简洁、可追溯的项目记忆。
影响范围：全仓库协作。
日期：2026-09-23

## D002 - Raw data handling
决定：`data/raw/` 默认不纳入 Git；只跟踪 `.gitkeep` 和数据说明。
原因：原始竞赛数据可能很大或受分发限制。
影响范围：数据管理和远程仓库。
日期：2026-09-23

## D003 - Validation protocol pending problem review
决定：目前不预设数据划分、指标或随机种子；赛题和数据结构确认后再填写统一验证方案。
原因：赛题尚未提供，提前选定会制造未经验证的假设。
影响范围：所有预测、分类和机器学习实验。
日期：2026-09-23
