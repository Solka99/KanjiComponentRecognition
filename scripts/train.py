"""Train the character classifier with or without component heads.

Every run belongs to an experiment folder under results/ (see EXPERIMENTS.md). Examples:
  python scripts/train.py --results-dir results/E01_baseline_vs_components --name baseline_5 --samples-per-class 5
  python scripts/train.py --results-dir results/E01_baseline_vs_components --name multitask_20 \
      --samples-per-class 20 --components nested positioned
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from kanji_components.etl9b import Etl9bCache
from kanji_components.model import KanjiNet, count_parameters
from kanji_components.provenance import provenance
from kanji_components.training import GlyphDataset, MultiTaskLoss, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--samples-per-class", type=int, default=None, help="None keeps all training samples")
parser.add_argument("--components", nargs="*", default=[], choices=["nested", "positioned"])
parser.add_argument("--component-weight", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--learning-rate", type=float, default=2e-3)
parser.add_argument("--width", type=int, default=16)
parser.add_argument("--threads", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--validation-subset", type=int, default=15000,
                    help="validation samples scored after each epoch; full validation and test run at the end")
parser.add_argument("--results-dir", required=True, help="experiment folder, e.g. results/E04_...")
parser.add_argument("--targets", default="data/component_targets_df5.npz", help="component label file")
parser.add_argument("--exclude-characters", default=None,
                    help="text file of characters removed from train, validation and test (zero-shot experiment)")
arguments = parser.parse_args()

torch.manual_seed(arguments.seed)
torch.set_num_threads(arguments.threads)
run_dir = Path(arguments.results_dir) / arguments.name
run_dir.mkdir(parents=True, exist_ok=True)

cache = Etl9bCache()
targets_archive = np.load(arguments.targets, allow_pickle=False)
assert list(targets_archive["characters"]) == cache.characters
component_targets = {name: targets_archive[f"{name}_targets"] for name in arguments.components}

excluded_labels = np.zeros(len(cache.characters), dtype=bool)
if arguments.exclude_characters:
    excluded_characters = set(Path(arguments.exclude_characters).read_text().strip())
    excluded_labels[[cache.characters.index(character) for character in excluded_characters]] = True
    print(f"excluding {excluded_labels.sum()} characters")


def keep_seen(indices):
    return indices[~excluded_labels[cache.labels[indices]]]


train_indices = keep_seen(cache.subsample_per_class(cache.indices("train"), arguments.samples_per_class, arguments.seed))
validation_indices = keep_seen(cache.indices("validation"))
test_indices = keep_seen(cache.indices("test"))
print(f"train {len(train_indices)}  validation {len(validation_indices)}  test {len(test_indices)}")

loader_options = dict(batch_size=arguments.batch_size, num_workers=2, persistent_workers=True)
train_loader = DataLoader(GlyphDataset(cache.images, cache.labels, train_indices, augment=True),
                          shuffle=True, drop_last=True, **loader_options)
validation_subset = np.sort(np.random.default_rng(0).choice(validation_indices, arguments.validation_subset, replace=False))
validation_loader = DataLoader(GlyphDataset(cache.images, cache.labels, validation_subset, augment=False),
                               **loader_options)
full_validation_loader = DataLoader(GlyphDataset(cache.images, cache.labels, validation_indices, augment=False),
                                    **loader_options)
test_loader = DataLoader(GlyphDataset(cache.images, cache.labels, test_indices, augment=False), **loader_options)

head_sizes = {"character": len(cache.characters)}
head_sizes.update({name: targets.shape[1] for name, targets in component_targets.items()})
model = KanjiNet(head_sizes, width=arguments.width).to(memory_format=torch.channels_last)
criterion = MultiTaskLoss(component_targets, arguments.component_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-4)
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=arguments.learning_rate,
                                                total_steps=arguments.epochs * steps_per_epoch, pct_start=0.15)
print(f"model {count_parameters(model) / 1e6:.2f}M parameters, heads {head_sizes}, {steps_per_epoch} steps/epoch")

history = []
best_validation_top1, best_state = -1.0, None
for epoch in range(1, arguments.epochs + 1):
    model.train()
    epoch_started = time.time()
    running = {}
    for step, (images, labels) in enumerate(train_loader, start=1):
        images = images.contiguous(memory_format=torch.channels_last)
        total_loss, losses = criterion(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        for name, value in losses.items():
            running[name] = running.get(name, 0.0) + value.item()
        if step % 200 == 0:
            rate = step * arguments.batch_size / (time.time() - epoch_started)
            print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {total_loss.item():.3f} {rate:.0f} img/s", flush=True)
    validation_metrics = evaluate(model, validation_loader, component_targets)
    record = {"epoch": epoch, "seconds": round(time.time() - epoch_started),
              **{f"train_loss_{name}": round(value / steps_per_epoch, 4) for name, value in running.items()},
              **{f"validation_{key}": round(value, 4) for key, value in validation_metrics.items()}}
    history.append(record)
    print(json.dumps(record), flush=True)
    if validation_metrics["top1"] > best_validation_top1:
        best_validation_top1 = validation_metrics["top1"]
        best_state = {key: value.clone() for key, value in model.state_dict().items()}
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))

model.load_state_dict(best_state)
full_validation_metrics = evaluate(model, full_validation_loader, component_targets)
test_metrics = evaluate(model, test_loader, component_targets)
summary = {"name": arguments.name, "arguments": vars(arguments), "train_samples": len(train_indices),
           "best_validation_subset_top1": best_validation_top1, "validation": full_validation_metrics,
           "test": test_metrics, "history": history,
           "provenance": provenance([arguments.targets] + ([arguments.exclude_characters] if arguments.exclude_characters else []))}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
torch.save({"state_dict": best_state, "head_sizes": head_sizes, "width": arguments.width,
            "characters": cache.characters}, run_dir / "model.pt")
print("TEST", json.dumps(test_metrics))
