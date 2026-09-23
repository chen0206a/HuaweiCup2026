"""加载官方评估器源码（只读，不修改）。

官方模块使用顶层平铺 import（`from evaluation_validation import ...`），
必须把 code/ 目录放进 sys.path 才能导入；子进程调用则由 cwd 解决。
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from contextlib import contextmanager

from .paths import attachment_dir, official_code_dir


@contextmanager
def official_path():
    """临时把官方 code/ 加入 sys.path。"""
    entry = str(official_code_dir())
    added = entry not in sys.path
    if added:
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


def official_module(name: str):
    """导入官方模块；导入期间 code/ 在 sys.path 上，之后从 sys.modules 复用。"""
    with official_path():
        return importlib.import_module(name)


def derive_multicore_plan(graph_json, plan):
    """官方方案推导（含 COPY 收缩依赖、子图 DAG、同核顺序校验）。"""
    module = official_module("stub_multicore_cut_and_schedule")
    return module.derive_multicore_plan(graph_json, plan)


def validate_graph(graph_json):
    module = official_module("evaluation_validation")
    return module.validate_graph(graph_json)


def read_official_config():
    module = official_module("evaluation_validation")
    settings = module.read_evaluation_config(str(attachment_dir() / "data" / "config.txt"))
    waits = module.read_required_settings(
        str(attachment_dir() / "data" / "config.txt"),
        "multicore_scene_a",
        ("task_cross_core_wait_cycles", "task_same_core_wait_cycles"))
    return settings["bandwidth"], settings["capacity"], waits


def official_fingerprint() -> dict:
    """官方评估器与固定 config 的哈希，用于评价缓存键与可复现记录。"""
    targets = [
        "code/multicore_cut_evaluate_problem_1.py",
        "code/stub_multicore_cut_and_schedule.py",
        "code/evaluation_validation.py",
        "code/schedule_step1.py",
        "code/schedule_step2.py",
        "code/schedule_step3.py",
        "code/contest_io.py",
        "data/config.txt",
    ]
    root = attachment_dir()
    digest = {}
    for rel in targets:
        path = root / rel
        if path.is_file():
            digest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
