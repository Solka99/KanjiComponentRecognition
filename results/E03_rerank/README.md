# E03 — Using components at test time: re-ranking the top-5

**Date:** 2026-11-22 · **Status:** done · **Builds on:** E01 models (no retraining)

## Question
Does using the predicted components at test time, not only during training, improve recognition?

## Setup
- Models: E01 `multitask_5`, `multitask_20`, `multitask_all`.
- For each image, the character output's top-5 candidates are scored by
  `log P(candidate) + weight × component log-likelihood` (same matching as E02) and re-ranked.
- Weight chosen on the full validation split from {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1};
  test scored once with the chosen weight.

## How to rerun
Needs E01 test and validation predictions (see E01 README), then:
```
.venv/bin/python scripts/rerank.py
```

## Results (test set, compared with the same model without re-ranking)
| samples/class | weight | top-1 before | top-1 after | fixed / broken | McNemar p |
|---|---|---|---|---|---|
| 5 | 0.10 | 0.9114 | 0.9166 | 854 / 537 | < 1e-16 |
| 20 | 0.10 | 0.9825 | 0.9829 | 132 / 108 | 0.14 |
| all (160) | 0.02 | 0.9929 | 0.9929 | 11 / 8 | 0.65 |

## Conclusion
Re-ranking removes a further 5.9% of errors at 5 samples per class and nothing significant with
more data. It fixes pairs whose parts differ (椋/掠, 狙/祖) and breaks characters with thin
labels (童 described only as 立@top). Details: `results.ipynb` section 4.

## Files
`rerank.json` (weights, validation curve, test numbers), `rerank_changes.csv` (every changed answer).

## Provenance
Computed before automatic provenance recording; rerunning `scripts/rerank.py` reproduces
`rerank.json` byte for byte (checked 2026-11-22) and writes `provenance.json`.
