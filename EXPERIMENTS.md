# Experiment log

Every experiment variant has its own folder `results/E##_<short_name>/` with a README (question,
what changed, how to rerun, results, conclusion). Earlier folders are never overwritten. New runs
record the git commit, input file checksums and hardware in `summary.json` / `provenance.json`.

| id | date | question | headline result |
|---|---|---|---|
| [E01](results/E01_baseline_vs_components/README.md) | 2026-11-17 | Do component outputs during training improve recognition, depending on data size? | Errors −20% at 5 samples/class, −17% at 20 (significant); tie with all data |
| [E02](results/E02_zero_shot/README.md) | 2026-11-18 | Can the model recognise characters never seen in training from components? | 67% top-1 among 300 unseen kanji (chance 0.3%); baseline 0% |
| [E03](results/E03_rerank/README.md) | 2026-11-22 | Does using components at test time (re-ranking top-5) help further? | A further −5.9% errors at 5 samples/class; no significant effect with more data |

## Planned
- **E04** — label rule B (components in ≥2 characters, every position kept) with region-based
  position outputs that share one component classifier across positions.

## Shared inputs
| file | what | used by |
|---|---|---|
| `data/etl9b/etl9b_cache.npz` | ETL9B images, labels, writer split (`scripts/build_cache.py`; not in git, licence) | all |
| `data/component_targets_df5.npz` | labels, cutoff: component / component@position in ≥5 characters | E01–E03 |
| `data/heldout_characters.txt` | 300 kanji hidden from training | E02 |

## Adding an experiment
1. Pick the next id and create `results/E##_<short_name>/`.
2. New labels go into a new file (`scripts/build_labels.py --output data/component_targets_<rule>.npz`);
   new model or loss variants become options of `scripts/train.py` whose defaults keep earlier runs reproducible.
3. Train with `--results-dir results/E##_<short_name>` (and `--targets` if labels changed).
4. Write the folder README from the template of E01 and add a row above.
