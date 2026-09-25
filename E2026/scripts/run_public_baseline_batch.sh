#!/usr/bin/env bash
# Run from E2026/ with Python dependencies available in the environment.
set -euo pipefail
for model in TFN MulT MISA; do
  for seed in 42 43 44; do
    python scripts/run_public_baseline.py "$model" --seed "$seed" --missing
  done
done
python scripts/summarize_public_baselines.py
