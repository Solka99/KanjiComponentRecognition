"""Datasets, loss, and evaluation shared by the training script."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

PAD = 4


class GlyphDataset(Dataset):
    """Yields (image float tensor 1x64x64, class index). Training mode applies a random shift of up to PAD px."""

    def __init__(self, images, labels, indices, augment):
        self.images, self.labels, self.indices, self.augment = images, labels, indices, augment
        self.generator = np.random.default_rng(0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        image = self.images[index]
        if self.augment:
            shift_y, shift_x = self.generator.integers(-PAD, PAD + 1, size=2)
            image = np.roll(image, (shift_y, shift_x), axis=(0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
        return tensor, int(self.labels[index])


class MultiTaskLoss(nn.Module):
    """Cross-entropy on characters plus BCE on each component head, targets looked up from the class index.

    Positives are ~1% of component labels, so each label gets pos_weight = negatives/positives capped at 20."""

    MAX_POSITIVE_WEIGHT = 20.0

    def __init__(self, component_targets, component_weight):
        super().__init__()
        self.component_weight = component_weight
        self.component_names = list(component_targets)
        self.cross_entropy = nn.CrossEntropyLoss()
        for name, targets in component_targets.items():
            targets = torch.from_numpy(targets)
            positives = targets.sum(0)
            positive_weight = ((len(targets) - positives) / positives.clamp(min=1)).clamp(max=self.MAX_POSITIVE_WEIGHT)
            self.register_buffer(f"targets_{name}", targets)
            self.register_buffer(f"positive_weight_{name}", positive_weight)

    def forward(self, outputs, labels):
        losses = {"character": self.cross_entropy(outputs["character"], labels)}
        for name in self.component_names:
            targets = getattr(self, f"targets_{name}")[labels]
            losses[name] = nn.functional.binary_cross_entropy_with_logits(
                outputs[name], targets, pos_weight=getattr(self, f"positive_weight_{name}"))
        total = losses["character"] + self.component_weight * sum(losses[name] for name in self.component_names)
        return total, losses


@torch.no_grad()
def evaluate(model, loader, component_targets):
    """Top-1/top-5 character accuracy plus micro and macro F1 at 0.5 for every component head."""
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    true_positive, predicted_positive, actual_positive = {}, {}, {}
    for images, labels in loader:
        outputs = model(images)
        top5 = outputs["character"].topk(5, dim=1).indices
        correct_top1 += (top5[:, 0] == labels).sum().item()
        correct_top5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        total += len(labels)
        for name, targets in component_targets.items():
            predicted = outputs[name] > 0
            actual = torch.from_numpy(targets)[labels] > 0.5
            true_positive[name] = true_positive.get(name, 0) + (predicted & actual).sum(0)
            predicted_positive[name] = predicted_positive.get(name, 0) + predicted.sum(0)
            actual_positive[name] = actual_positive.get(name, 0) + actual.sum(0)
    metrics = {"top1": correct_top1 / total, "top5": correct_top5 / total}
    for name in component_targets:
        tp, pp, ap = true_positive[name].float(), predicted_positive[name].float(), actual_positive[name].float()
        per_label_f1 = 2 * tp / (pp + ap).clamp(min=1)
        metrics[f"{name}_f1_micro"] = (2 * tp.sum() / (pp.sum() + ap.sum()).clamp(min=1)).item()
        metrics[f"{name}_f1_macro"] = per_label_f1[ap > 0].mean().item()
    return metrics
