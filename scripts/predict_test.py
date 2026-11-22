"""Run a trained model over a whole split (test by default) and save predictions for analysis.

Writes results/<name>/<split>_predictions.npz with the true labels, the top-5 predicted labels, their
softmax probabilities, the probability of the true class and, for every component head, the
sigmoid probabilities (float16).

Example: python scripts/predict_test.py --results-dir results/E01_baseline_vs_components --name multitask_all
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from kanji_components.etl9b import Etl9bCache
from kanji_components.model import KanjiNet
from kanji_components.training import GlyphDataset

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--split", default="test", choices=["validation", "test"])
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--results-dir", required=True, help="experiment folder holding the run")
arguments = parser.parse_args()
torch.set_num_threads(arguments.threads)

run_dir = Path(arguments.results_dir) / arguments.name
checkpoint = torch.load(run_dir / "model.pt", weights_only=False)
model = KanjiNet(checkpoint["head_sizes"], width=checkpoint["width"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()

cache = Etl9bCache()
assert cache.characters == checkpoint["characters"]
test_indices = cache.indices(arguments.split)
loader = DataLoader(GlyphDataset(cache.images, cache.labels, test_indices, augment=False), batch_size=512)

component_names = [name for name in checkpoint["head_sizes"] if name != "character"]
top5_batches, top5_probability_batches, true_probability_batches = [], [], []
probability_batches = {name: [] for name in component_names}
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        character_probabilities = torch.softmax(outputs["character"], dim=1)
        top5 = character_probabilities.topk(5, dim=1)
        top5_batches.append(top5.indices.numpy().astype(np.int16))
        top5_probability_batches.append(top5.values.numpy().astype(np.float32))
        true_probability_batches.append(character_probabilities[torch.arange(len(labels)), labels].numpy())
        for name in component_names:
            probability_batches[name].append(torch.sigmoid(outputs[name]).numpy().astype(np.float16))

predictions = {"indices": test_indices, "labels": cache.labels[test_indices], "top5": np.concatenate(top5_batches),
               "top5_probabilities": np.concatenate(top5_probability_batches),
               "true_probability": np.concatenate(true_probability_batches)}
predictions.update({f"{name}_probabilities": np.concatenate(batches) for name, batches in probability_batches.items()})
np.savez_compressed(run_dir / f"{arguments.split}_predictions.npz", **predictions)
top1 = (predictions["top5"][:, 0] == predictions["labels"]).mean()
print(f"{arguments.name}: {len(test_indices)} {arguments.split} images, top1 {top1:.4f}")
