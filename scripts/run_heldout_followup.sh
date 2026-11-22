#!/bin/bash
# E02: waits for the held-out training run, then dumps predictions and scores zero-shot recognition.
cd "$(dirname "$0")/.."
OUT=results/E02_zero_shot
while [ ! -f $OUT/heldout_multitask/summary.json ]; do sleep 30; done
.venv/bin/python scripts/predict_test.py --results-dir $OUT --name heldout_multitask 2>&1 | grep -v Warning
.venv/bin/python scripts/zero_shot.py --results-dir $OUT --name heldout_multitask
echo FOLLOWUP DONE
