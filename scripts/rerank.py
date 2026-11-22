"""Re-rank the character head's top-5 candidates by agreement with the predicted components.

combined score = log P(candidate | character head) + weight * component log-likelihood, where the
component log-likelihood is the Bernoulli log-likelihood of the candidate's KanjiVG component
vector (nested + positioned) under the predicted component probabilities, as in zero_shot.py.
The weight is chosen on the validation split; the test split is scored once with that weight.

Needs results/<run>/{validation,test}_predictions.npz (scripts/predict_test.py --split ...).
Writes results/rerank.json and results/rerank_changes.csv.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {5: "multitask_5", 20: "multitask_20", 200: "multitask_all"}
WEIGHTS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
PROBABILITY_FLOOR = 1e-4
CHUNK = 10000
RESULTS_DIR = Path("results")

archive = np.load("data/component_targets.npz", allow_pickle=False)
characters = list(archive["characters"])
component_targets = np.concatenate([archive["nested_targets"], archive["positioned_targets"]], axis=1)
component_targets = component_targets.astype(np.float32)


def candidate_component_scores(predictions):
    """Component log-likelihood of each of the 5 candidates, shape (images, 5)."""
    probabilities = np.concatenate([predictions["nested_probabilities"], predictions["positioned_probabilities"]],
                                   axis=1).astype(np.float32).clip(PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    candidates = predictions["top5"].astype(np.int64)
    scores = np.empty(candidates.shape, dtype=np.float32)
    for start in range(0, len(candidates), CHUNK):
        log_present = np.log(probabilities[start:start + CHUNK])
        log_absent = np.log1p(-probabilities[start:start + CHUNK])
        candidate_targets = component_targets[candidates[start:start + CHUNK]]
        scores[start:start + CHUNK] = (np.einsum("nk,nck->nc", log_present, candidate_targets)
                                       + np.einsum("nk,nck->nc", log_absent, 1 - candidate_targets))
    return scores


def reranked_top1(predictions, component_scores, weight):
    character_scores = np.log(np.clip(predictions["top5_probabilities"], 1e-12, 1))
    best = np.argmax(character_scores + weight * component_scores, axis=1)
    return predictions["top5"][np.arange(len(best)), best].astype(np.int64)


def exact_mcnemar_p_value(fixed, broken):
    trials = fixed + broken
    tail = sum(comb(trials, k) for k in range(min(fixed, broken) + 1)) / 2 ** trials
    return min(1.0, 2 * tail)


results, change_rows = [], []
for samples_per_class, run in RUNS.items():
    validation = np.load(RESULTS_DIR / run / "validation_predictions.npz")
    test = np.load(RESULTS_DIR / run / "test_predictions.npz")
    validation_scores = candidate_component_scores(validation)
    validation_labels = validation["labels"].astype(np.int64)
    validation_accuracy = {weight: float((reranked_top1(validation, validation_scores, weight) == validation_labels).mean())
                           for weight in WEIGHTS}
    best_weight = max(validation_accuracy, key=validation_accuracy.get)

    test_scores = candidate_component_scores(test)
    test_labels = test["labels"].astype(np.int64)
    before = test["top5"][:, 0].astype(np.int64)
    after = reranked_top1(test, test_scores, best_weight)
    fixed = int(((after == test_labels) & (before != test_labels)).sum())
    broken = int(((after != test_labels) & (before == test_labels)).sum())
    results.append({
        "samples_per_class": samples_per_class, "run": run, "chosen_weight": best_weight,
        "validation_top1_by_weight": validation_accuracy,
        "test_top1_without_rerank": float((before == test_labels).mean()),
        "test_top1_with_rerank": float((after == test_labels).mean()),
        "test_top5": float((test["top5"] == test_labels[:, None]).any(1).mean()),
        "answers_changed": int((after != before).sum()), "fixed": fixed, "broken": broken,
        "mcnemar_p_value": exact_mcnemar_p_value(fixed, broken),
    })
    for row in np.flatnonzero(after != before):
        change_rows.append({"samples_per_class": samples_per_class, "test_index": int(test["indices"][row]),
                            "true": characters[test_labels[row]], "before": characters[before[row]],
                            "after": characters[after[row]],
                            "outcome": "fixed" if after[row] == test_labels[row]
                            else "broken" if before[row] == test_labels[row] else "still wrong"})
    print(json.dumps({key: value for key, value in results[-1].items() if key != "validation_top1_by_weight"}))

(RESULTS_DIR / "rerank.json").write_text(json.dumps(results, indent=1))
pd.DataFrame(change_rows).to_csv(RESULTS_DIR / "rerank_changes.csv", index=False)
print(f"saved results/rerank.json and results/rerank_changes.csv ({len(change_rows)} changed answers)")
