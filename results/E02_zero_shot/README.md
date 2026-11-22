# E02 — Zero-shot recognition of characters never seen in training

**Date:** 2026-11-18 · **Status:** done · **Builds on:** E01 (same model, labels, training settings)

## Question
Can the component-aware model recognise characters it was never trained on, using only its
predicted components?

## Setup
- Held-out characters: `data/heldout_characters.txt`, 300 kanji chosen by `scripts/pick_heldout.py`
  (seed 0): at least two positioned components, a component vector unique among all 3036 classes,
  and every component label still present in at least 3 other training characters.
- Training: as E01 `multitask_all` (8 epochs, all writers) but with the 300 classes removed from
  training, validation and test (`--exclude-characters`).
- Scoring (`scripts/zero_shot.py`): for each test image of a held-out character, every candidate
  is scored by the Bernoulli log-likelihood of its KanjiVG component vector under the predicted
  component probabilities (clipped to [1e-4, 1 − 1e-4]). Two candidate sets: the 300 held-out
  characters, and all 3036.

## How to rerun
```
.venv/bin/python scripts/pick_heldout.py
.venv/bin/python scripts/train.py --results-dir results/E02_zero_shot --name heldout_multitask \
    --components nested positioned --epochs 8 --exclude-characters data/heldout_characters.txt
bash scripts/run_heldout_followup.sh
```

## Results (5,988 test images of the 300 held-out characters)
| method | candidates | top-1 | top-5 | chance top-1 |
|---|---|---|---|---|
| component matching | 300 held-out | 0.674 | 0.884 | 0.0033 |
| component matching | all 3036 | 0.280 | 0.445 | 0.0003 |
| character output (baseline-style) | all 3036 | 0.000 | 0.000 | 0.0003 |

216 of 300 characters recognised in at least half of their images; 34 never.

## Conclusion
Structure lets the model recognise unseen characters far above chance, which a flat classifier
cannot do at all. Failures are mostly characters with thin KanjiVG descriptions (no positions,
rare parts dropped by the 5-character cutoff). Details: `results.ipynb` section 3.

## Files
`heldout_multitask/` — `summary.json`, `history.json`, `model.pt`, `zero_shot.json`,
`zero_shot_predictions.csv`, `zero_shot_per_character.csv`; `heldout_multitask.log`.

## Provenance
Predates automatic provenance recording; code as in the first git commit.
