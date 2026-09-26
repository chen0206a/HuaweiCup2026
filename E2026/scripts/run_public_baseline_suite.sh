#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
models=(LMF MFN Self-MM MMIM MAG-BERT TFR-Net MissModal M3S MMIN)
for model in "${models[@]}"; do
  for seed in 42 43 44; do
    target="experiments/q2/public_baselines/${model}/metrics_seed${seed}.json"
    if [[ -s "$target" ]]; then
      echo "SKIP completed ${model} seed=${seed}"
      continue
    fi
    echo "START ${model} seed=${seed} $(date -Is)"
    mkdir -p "experiments/q2/public_baselines/${model}"
    python scripts/run_public_baseline_extended.py "$model" --seed "$seed" \
      > "experiments/q2/public_baselines/${model}/train_seed${seed}.log" 2>&1
    echo "DONE ${model} seed=${seed} $(date -Is)"
  done
done
