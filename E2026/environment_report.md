# E 题 Q2 第一阶段环境记录

核对时间：2026-09-23（Asia/Shanghai）。服务器工作目录：`/root/workspace/E2026`。

| 项目 | 服务器实际情况 |
| --- | --- |
| 系统 | Ubuntu 24.04.4 LTS，Linux 5.15.0-25-generic |
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB；核对时无运行中的 GPU 进程 |
| 驱动 / CUDA | NVIDIA Driver 580.95.05；`nvidia-smi` 报 CUDA 13.0；CUDA toolkit 13.0.88 位于 `/usr/local/cuda-13.0`（非交互 SSH 的 `PATH` 中未包含 `nvcc`） |
| Python | `/usr/bin/python`，3.12.3；pip 26.2.1 |
| PyTorch | 2.14.0+cu130；`torch.version.cuda=13.0`；`torch.cuda.is_available()=True` |
| CPU / 内存 | Intel Xeon Gold 6148，80 逻辑 CPU；内存 60 GiB，复核时约 59 GiB 可用；无 swap |
| 磁盘 | `/root/workspace`：300 GiB，复核时约 300 GiB 可用 |

## Python 依赖复核

| 包 | 复核结果 |
| --- | --- |
| NumPy | 2.5.2 |
| Pandas | 3.0.6 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1（Python 导入名 `sklearn`） |
| PyYAML | 6.0.3（导入名 `yaml`） |
| tqdm | 4.70.1 |
| matplotlib | 3.11.2 |
| openpyxl | 3.1.5 |
| pytest | 9.1.1 |

以上均在同一服务器的 `/usr/bin/python` 下实际导入成功。用户最初提供的检查显示 Pandas、SciPy、scikit-learn、tqdm、matplotlib、openpyxl、pytest 缺失；用户随后手动安装，本次复核显示全部可用。本阶段我在**服务器端未安装任何包**，也未升级 PyTorch/CUDA。为连接服务器，本地 Windows Python 单独安装了 `paramiko 5.0.0` 及其依赖 `invoke 3.0.3`；这不是 Q2 运行依赖，也未装到服务器。

本阶段未下载或使用任何预训练模型权重。附件 2 的已预计算数值特征足以支持下一阶段的数据审计。
