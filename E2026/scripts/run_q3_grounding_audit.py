"""Repository-root entry point: python E2026/scripts/run_q3_grounding_audit.py."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.q3.attachment4_audit import run  # noqa: E402


if __name__ == "__main__":
    outcome = run()
    print("Q3-2 status", outcome["conclusion"], flush=True)
