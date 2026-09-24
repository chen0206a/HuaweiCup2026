"""Repository-root entry point: python E2026/scripts/run_q3_heaf_validation.py."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.q3.heaf_validation import run  # noqa: E402


if __name__ == "__main__":
    outcome = run()
    print("Q3-1 status", outcome["status"], flush=True)
