#!/usr/bin/env python3
"""Check that the lightweight competition workspace skeleton is present."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
required = ["AGENTS.md", "CONTEXT.md", "STATUS.md", "DECISIONS.md", "TASKS.md", "README.md", ".gitignore", "problem/problem.md", "problem/attachments.md", "shared/validation_protocol.md", "results", "experiments", "paper/main.tex"]
required += [f"q{i}/HANDOFF.md" for i in range(1, 5)]
missing = []
for item in required:
    path = ROOT / item
    ok = path.exists()
    print(f"{'PASS' if ok else 'WARNING'}: {item} {'exists' if ok else 'missing'}")
    if not ok:
        missing.append(item)
print(f"\nProject check: {'PASS' if not missing else 'WARNING'} ({len(required)-len(missing)}/{len(required)} checks passed)")
raise SystemExit(1 if missing else 0)
