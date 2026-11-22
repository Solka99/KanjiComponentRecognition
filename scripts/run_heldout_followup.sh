#!/bin/bash
# Waits for the held-out training run, then dumps predictions and scores zero-shot recognition.
cd "$(dirname "$0")/.."
while [ ! -f results/heldout_multitask/summary.json ]; do sleep 30; done
.venv/bin/python scripts/predict_test.py --name heldout_multitask 2>&1 | grep -v Warning
.venv/bin/python scripts/zero_shot.py --name heldout_multitask
echo FOLLOWUP DONE
