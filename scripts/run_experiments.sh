#!/usr/bin/env bash
# Stage 1 (baseline) vs Stage 2 (component heads) at three data regimes, run sequentially.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { echo "=== $(date '+%F %T') start $1"; $PY scripts/train.py --name "$@" 2>&1 | grep -v Warning; echo "=== $(date '+%F %T') end $1"; }
run baseline_5    --samples-per-class 5   --epochs 30
run multitask_5   --samples-per-class 5   --epochs 30 --components nested positioned
run baseline_20   --samples-per-class 20  --epochs 20
run multitask_20  --samples-per-class 20  --epochs 20 --components nested positioned
run baseline_all  --epochs 8
run multitask_all --epochs 8 --components nested positioned
echo "QUEUE DONE"
