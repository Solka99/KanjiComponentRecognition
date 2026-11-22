# E01 — Baseline vs component-aware training

**Date:** 2026-11-17 to 2026-11-18 · **Status:** done

## Question
Does adding component outputs during training improve character recognition, and how does the
effect depend on the amount of training data?

## Setup
- Data: ETL9B, writer-disjoint split by sheet (3200 / 400 / 400 sheets). Test set: 60,723 images.
- Labels: `data/component_targets_df5.npz` — nested components and component@position pairs kept
  when they occur in at least 5 characters (382 and 251 labels).
- Model: KanjiNet width 16 (5×5 stride-2 stem, 7 conv layers, global average pooling, dropout 0.2).
  - `baseline_*`: character output only, cross-entropy. 0.54 M parameters.
  - `multitask_*`: plus nested and positioned outputs, binary cross-entropy with positive weight
    capped at 20. 0.62 M parameters. Component outputs unused at test time.
- Training: AdamW, one-cycle, peak LR 2e-3, batch 256, ±4 px random shift, seed 0, CPU
  (i7-10750H). Epochs: 30 at 5 samples/class, 20 at 20, 8 at all (160).
- Best epoch chosen on a 15,000-image validation subset; test scored once.

## How to rerun
```
bash scripts/run_experiments.sh                   # six runs, log in queue.log
for run in baseline_5 multitask_5 baseline_20 multitask_20 baseline_all multitask_all; do
  .venv/bin/python scripts/predict_test.py --results-dir results/E01_baseline_vs_components --name $run
done
for run in multitask_5 multitask_20 multitask_all; do   # validation predictions, used by E03
  .venv/bin/python scripts/predict_test.py --results-dir results/E01_baseline_vs_components --name $run --split validation
done
```

## Results (test set)
| samples/class | baseline top-1 | component-aware top-1 | errors removed | fixed / broken | McNemar p |
|---|---|---|---|---|---|
| 5 | 0.8887 | 0.9114 | 20.4% | 3368 / 1991 | < 1e-9 |
| 20 | 0.9791 | 0.9825 | 16.6% | 708 / 497 | < 1e-9 |
| all (160) | 0.9923 | 0.9929 | 6.7% | 240 / 209 | 0.16 |

Macro-F1, per-character accuracy, confidence metrics and the confusion analysis: `results.ipynb`
sections 1–2. Component detection quality of `multitask_all`: `components.ipynb`.

## Conclusion
Component outputs during training help clearly with scarce data (5 and 20 samples per class) and
make no significant difference with all data. Single seed per cell.

## Files
- `<run>/summary.json`, `history.json`, `model.pt` — per run; `*_predictions.npz` regenerable, not in git.
- `confusion_pairs.csv`, `per_class_accuracy.csv` — from `results.ipynb` section 2.
- `multitask_all/component_detection_per_label.csv` — from `components.ipynb`.
- `queue.log`; `queue_aborted_slow_model.log` is a first attempt with a slower model (two convs at
  64 px, 420 img/s) that was stopped and replaced; none of its runs are used.

## Provenance
These runs predate automatic provenance recording. The code is in the first git commit; options
added later (`--exclude-characters`, `--targets`, `--split`) default to the behaviour used here.
