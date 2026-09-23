"""确认官方评估器与固定 config 未被改动。

对比 notes/a_review_checks/evaluator_hashes.json 中记录的哈希；不一致即说明
官方源码被修改，任何基于它的成绩都不可信。

用法：python src/q1/check_official.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.q1"

from .official import official_fingerprint  # noqa: E402
from .paths import REPO_ROOT, attachment_dir  # noqa: E402

RECORDED = REPO_ROOT / "notes" / "a_review_checks" / "evaluator_hashes.json"


def main() -> int:
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))
    root = attachment_dir()
    mismatches = []
    for rel, expected in recorded.items():
        path = root / rel.replace("\\", "/")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        state = "OK  " if actual == expected else "DIFF"
        if actual != expected:
            mismatches.append((rel, expected, actual))
        print(f"{state} {rel}")
    print("\n当前指纹:", json.dumps(official_fingerprint(), indent=2))
    if mismatches:
        print(f"\n{len(mismatches)} 个官方文件与记录不一致 —— 评估结果不可信")
        return 1
    print("\n官方评估器与 config 未被修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
