"""Recognise held-out characters from predicted components alone.

Each candidate character is scored by the Bernoulli log-likelihood of its KanjiVG component
target vector under the component probabilities the model predicted for the image; the best
scoring candidate is the prediction. Needs <results-dir>/<name>/test_predictions.npz.

Writes zero_shot.json, zero_shot_predictions.csv and provenance.json into <results-dir>/<name>.
"""
import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from kanji_components.provenance import write_provenance

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="heldout_multitask")
parser.add_argument("--heldout", default="data/heldout_characters.txt")
parser.add_argument("--results-dir", default="results/E02_zero_shot")
parser.add_argument("--targets", default="data/component_targets_df5.npz")
arguments = parser.parse_args()
run_dir = Path(arguments.results_dir) / arguments.name

predictions = np.load(run_dir / "test_predictions.npz")
archive = np.load(arguments.targets, allow_pickle=False)
characters = list(archive["characters"])
heldout_characters = Path(arguments.heldout).read_text().strip()
heldout_labels = np.array(sorted(characters.index(character) for character in heldout_characters))
is_heldout_class = np.zeros(len(characters), dtype=bool)
is_heldout_class[heldout_labels] = True

PROBABILITY_FLOOR = 1e-4


def log_likelihood(name):
    probabilities = predictions[f"{name}_probabilities"].astype(np.float32).clip(PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    targets = archive[f"{name}_targets"].astype(np.float32)
    return np.log(probabilities) @ targets.T + np.log1p(-probabilities) @ (1 - targets).T


scores = sum(log_likelihood(name) for name in ("nested", "positioned"))
labels = predictions["labels"].astype(np.int64)
heldout_rows = is_heldout_class[labels]


def ranks(score_matrix, candidate_mask):
    masked = np.where(candidate_mask[None, :], score_matrix, -np.inf)
    true_scores = masked[np.arange(len(labels)), labels]
    return (masked > true_scores[:, None]).sum(1) + 1


def accuracy(rank_array, rows):
    return {"top1": float((rank_array[rows] == 1).mean()), "top5": float((rank_array[rows] <= 5).mean()),
            "images": int(rows.sum())}


rank_within_heldout = ranks(scores, is_heldout_class)
rank_within_all = ranks(scores, np.ones(len(characters), dtype=bool))
character_head_rank_all = (predictions["top5"] == labels[:, None]).argmax(1) + 1
character_head_rank_all[~(predictions["top5"] == labels[:, None]).any(1)] = 99

results = {
    "heldout_characters": len(heldout_labels),
    "components_within_heldout": accuracy(rank_within_heldout, heldout_rows),
    "components_within_all": accuracy(rank_within_all, heldout_rows),
    "character_head_on_heldout": accuracy(character_head_rank_all, heldout_rows),
    "components_within_all_on_seen": accuracy(rank_within_all, ~heldout_rows),
    "character_head_on_seen": accuracy(character_head_rank_all, ~heldout_rows),
    "chance_within_heldout_top1": 1 / len(heldout_labels),
}
(run_dir / "zero_shot.json").write_text(json.dumps(results, indent=1))
write_provenance(run_dir, [arguments.targets, arguments.heldout])

heldout_candidates = np.where(is_heldout_class[None, :], scores, -np.inf)
table = pd.DataFrame({
    "test_index": predictions["indices"][heldout_rows],
    "true": [characters[label] for label in labels[heldout_rows]],
    "predicted_within_heldout": [characters[label] for label in heldout_candidates[heldout_rows].argmax(1)],
    "rank_within_heldout": rank_within_heldout[heldout_rows],
    "predicted_within_all": [characters[label] for label in scores[heldout_rows].argmax(1)],
    "rank_within_all": rank_within_all[heldout_rows],
})
table.to_csv(run_dir / "zero_shot_predictions.csv", index=False)
print(json.dumps(results, indent=1))
