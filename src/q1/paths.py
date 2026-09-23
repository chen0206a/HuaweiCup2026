"""定位官方附件包（评估器源码、固定 config、case 数据）。

官方附件目录在 .gitignore 中，只存在于本地；路径不写死，按 A 题目录下的
"*附件" 目录查找，可用环境变量 HUAWEI_CUP_ATTACH 覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_EVALUATOR_NAME = "multicore_cut_evaluate_problem_1.py"


class OfficialPathsError(RuntimeError):
    pass


def attachment_dir() -> Path:
    """返回官方附件根目录（含 code/、data/、docs/）。"""
    override = os.environ.get("HUAWEI_CUP_ATTACH")
    if override:
        path = Path(override)
        if not (path / "code" / _EVALUATOR_NAME).is_file():
            raise OfficialPathsError(
                f"HUAWEI_CUP_ATTACH={override} 下找不到 code/{_EVALUATOR_NAME}")
        return path
    for cand in sorted(REPO_ROOT.glob("*/*/A题/*附件")):
        if (cand / "code" / _EVALUATOR_NAME).is_file():
            return cand
    raise OfficialPathsError(
        "找不到官方附件目录；请设置环境变量 HUAWEI_CUP_ATTACH 指向含 code/ 与 data/ 的目录")


def official_code_dir() -> Path:
    return attachment_dir() / "code"


def official_data_dir() -> Path:
    return attachment_dir() / "data"


def config_path() -> Path:
    return official_data_dir() / "config.txt"


def case_path(case: str) -> Path:
    """接受 'case_019'、'case_019.json' 或任意存在的图路径。"""
    candidate = Path(case)
    if candidate.is_file():
        return candidate
    name = candidate.name if candidate.suffix else f"{candidate.name}.json"
    path = official_data_dir() / name
    if not path.is_file():
        raise OfficialPathsError(f"找不到 case 文件: {path}")
    return path


def list_cases() -> list[Path]:
    return sorted(official_data_dir().glob("case_*.json"))
