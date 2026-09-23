"""官方 evaluator 封装（只调用，不修改)。

通过子进程调用 `code/multicore_cut_evaluate_problem_1.py`，cwd 设为官方
code/ 目录（官方模块使用顶层平铺 import）。任何异常路径都返回失败结果，
不抛出、不静默、不虚构 Makespan：

- 超时            → failure_reason = "timeout"
- 非零返回码       → failure_reason = "nonzero_returncode"
- evaluator 报错   → failure_reason = "evaluation_error"
- 结果 JSON 不存在 → failure_reason = "result_json_missing"
- JSON 解析失败    → failure_reason = "result_json_invalid"

对相同 (graph, config, problem, K, plan) 的结果做缓存，键为稳定 plan hash。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import REPO_ROOT, attachment_dir, config_path, official_code_dir

DEFAULT_TIMEOUT = 120.0
CACHE_DIR = REPO_ROOT / ".cache" / "q1_eval"

_EVALUATOR = {
    1: "multicore_cut_evaluate_problem_1.py",
}


@dataclass
class EvalResult:
    success: bool
    makespan: int | None
    added_copy_bytes: int | None
    evaluator_runtime: float
    failure_reason: str | None
    cached: bool = False
    returncode: int | None = None
    plan_hash: str = ""
    stderr_tail: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def canonical_plan(plan: dict) -> str:
    return json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def plan_hash(graph_json: dict, plan: dict, problem: int = 1, timeout: float = DEFAULT_TIMEOUT) -> str:
    """稳定 plan hash：图内容 + 固定 config + 问题号 + 精确映射与顺序 + 超时。"""
    digest = hashlib.sha256()
    digest.update(b"q1v0\n")
    digest.update(json.dumps(graph_json, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8"))
    digest.update(b"\n")
    digest.update(config_path().read_bytes())
    digest.update(f"\nproblem={problem}\ntimeout={timeout:.3f}\n".encode())
    digest.update(canonical_plan(plan).encode("utf-8"))
    return digest.hexdigest()


def _scrub_paths(text: str) -> str:
    """把本机绝对路径替换成占位符，避免写进提交产物。"""
    if not text:
        return text
    for raw, token in ((str(CACHE_DIR), "<cache>"),
                       (str(attachment_dir()), "<attach>"),
                       (str(REPO_ROOT), "<repo>")):
        for variant in (raw, raw.replace("\\", "/"), raw.replace("/", "\\")):
            text = text.replace(variant, token)
    return text


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(path: Path) -> EvalResult | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    payload["cached"] = True
    # 让缓存命中与全新评价产出完全一致的记录：路径脱敏，
    # 且成功的结果不保留官方 stdout（它只是 "OK: ..." 一行）。
    # 否则报告文件的内容会随是否命中缓存而变。
    tail = _scrub_paths(payload.get("stderr_tail", ""))
    payload["stderr_tail"] = "" if payload.get("success") else tail
    return EvalResult(**payload)


def _write_cache(path: Path, result: EvalResult):
    payload = asdict(result)
    payload["cached"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_plan(graph_json: dict, plan: dict, graph_name: str = "case",
                  problem: int = 1, timeout: float = DEFAULT_TIMEOUT,
                  use_cache: bool = True, work_dir: Path | None = None,
                  keep_artifacts: bool = False,
                  graph_path: Path | None = None) -> EvalResult:
    """调用官方 evaluator 评价一个方案。

    graph_path 给出时直接用原始 case 文件作为输入，避免重新序列化大图。
    """
    if problem not in _EVALUATOR:
        raise ValueError(f"本轮只封装问题 1；problem={problem} 暂不支持")

    key = plan_hash(graph_json, plan, problem, timeout)
    cache_file = _cache_path(key)
    if use_cache:
        hit = _read_cache(cache_file)
        if hit is not None:
            return hit

    work_dir = Path(work_dir) if work_dir else (CACHE_DIR / "_runs")
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{graph_name}_{key[:12]}"
    plan_file = work_dir / f"{stem}_plan.json"
    out_file = work_dir / f"{stem}_problem_{problem}_res.json"
    trace_file = work_dir / f"{stem}_trace.json"
    log_file = work_dir / f"{stem}_log.txt"

    if graph_path is None:
        graph_path = work_dir / f"{stem}_input.json"
        graph_path.write_text(json.dumps(graph_json, ensure_ascii=False), encoding="utf-8")
    plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    for stale in (out_file, trace_file, log_file):
        if stale.exists():
            stale.unlink()

    command = [
        sys.executable, _EVALUATOR[problem], str(graph_path), str(plan_file),
        "--config", str(config_path()),
        "-o", str(out_file),
        "--trace-output", str(trace_file),
        "--log-output", str(log_file),
    ]

    started = time.perf_counter()
    failure_reason, returncode, stderr_text, stdout_text = None, None, "", ""
    try:
        completed = subprocess.run(
            command, cwd=str(official_code_dir()),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout)
        returncode = completed.returncode
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        if returncode != 0:
            failure_reason = "nonzero_returncode"
    except subprocess.TimeoutExpired as expired:
        stderr_text = (expired.stderr or "") if isinstance(expired.stderr, str) else ""
        failure_reason = "timeout"
    runtime = time.perf_counter() - started

    makespan = added = None
    if failure_reason is None:
        if not out_file.is_file():
            failure_reason = "result_json_missing"
        else:
            try:
                payload = json.loads(out_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                failure_reason = "result_json_invalid"
            else:
                makespan = payload.get("makespan")
                added = payload.get("data_movement_bytes", {}).get("added_copy_bytes")
                if makespan is None or added is None:
                    failure_reason = "evaluator_error"

    # 成功时官方 stdout 只是 "OK: ..." 一行，没有诊断价值，不入库；
    # 失败时才保留 stderr（为空则退回 stdout），并脱敏本机路径。
    tail = stderr_text.strip()
    if not tail and failure_reason is not None:
        tail = stdout_text.strip()
    result = EvalResult(
        success=failure_reason is None,
        makespan=makespan,
        added_copy_bytes=added,
        evaluator_runtime=runtime,
        failure_reason=failure_reason,
        cached=False,
        returncode=returncode,
        plan_hash=key,
        stderr_tail=_scrub_paths(tail[-600:]),
    )

    # 超时受机器负载影响，不写入缓存，避免把偶发超时固化成结论。
    if use_cache and failure_reason != "timeout":
        _write_cache(cache_file, result)
    if not keep_artifacts:
        for artifact in (trace_file, log_file, plan_file, out_file):
            if artifact.exists():
                artifact.unlink()
        if graph_path.name.endswith("_input.json") and graph_path.exists():
            graph_path.unlink()
    return result
