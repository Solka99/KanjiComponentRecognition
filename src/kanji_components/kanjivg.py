"""Component targets derived from KanjiVG decomposition trees."""
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

KANJIVG_DIR = Path("data/kanji")
KVG = "{http://kanjivg.tagaini.net}"
SVG_GROUP_TAG = "{http://www.w3.org/2000/svg}g"
SVG_PATH_TAG = "{http://www.w3.org/2000/svg}path"


def root_group(character, kanjivg_dir=KANJIVG_DIR):
    root = ET.parse(kanjivg_dir / f"{ord(character):05x}.svg").getroot()
    stroke_paths = next(g for g in root.iter(SVG_GROUP_TAG) if g.get("id", "").startswith("kvg:StrokePaths"))
    return next(g for g in stroke_paths if g.tag == SVG_GROUP_TAG)


def main_components(group):
    """Direct labelled children; unlabelled groups are transparent. Returns (element, position) pairs."""
    pairs = []
    pending = [child for child in group if child.tag == SVG_GROUP_TAG]
    while pending:
        candidate = pending.pop(0)
        element = candidate.get(KVG + "element")
        if element is not None:
            pairs.append((element, candidate.get(KVG + "position") or "none"))
        else:
            pending = [child for child in candidate if child.tag == SVG_GROUP_TAG] + pending
    return pairs


def nested_components(group):
    """Every labelled group below the root at any depth, split parts counted once, variants also
    mapped to their kvg:original form."""
    elements = set()
    for descendant in group.iter(SVG_GROUP_TAG):
        if descendant is group or descendant.get(KVG + "element") is None:
            continue
        if descendant.get(KVG + "part") not in (None, "1"):
            continue
        elements.add(descendant.get(KVG + "element"))
        original = descendant.get(KVG + "original")
        if original:
            elements.add(original)
    return elements


def stroke_count(group):
    return sum(1 for _ in group.iter(SVG_PATH_TAG))


class ComponentTargets:
    """Multi-hot component targets for a list of characters.

    nested_vocabulary: components (any depth) with document frequency >= min_document_frequency.
    positioned_vocabulary: (main component, position) pairs with the same frequency floor.
    """

    def __init__(self, characters, min_document_frequency=5, kanjivg_dir=KANJIVG_DIR):
        self.characters = list(characters)
        self.nested_sets, self.positioned_sets, self.stroke_counts = [], [], []
        for character in self.characters:
            group = root_group(character, kanjivg_dir)
            self.nested_sets.append(nested_components(group))
            self.positioned_sets.append({f"{element}@{position}" for element, position in main_components(group)})
            self.stroke_counts.append(stroke_count(group))
        self.nested_vocabulary = self._vocabulary(self.nested_sets, min_document_frequency)
        self.positioned_vocabulary = self._vocabulary(self.positioned_sets, min_document_frequency)
        self.nested_targets = self._multi_hot(self.nested_sets, self.nested_vocabulary)
        self.positioned_targets = self._multi_hot(self.positioned_sets, self.positioned_vocabulary)

    @staticmethod
    def _vocabulary(label_sets, min_document_frequency):
        document_frequency = Counter(label for labels in label_sets for label in labels)
        kept = [label for label, count in document_frequency.items() if count >= min_document_frequency]
        return sorted(kept, key=lambda label: (-document_frequency[label], label))

    @staticmethod
    def _multi_hot(label_sets, vocabulary):
        index = {label: position for position, label in enumerate(vocabulary)}
        targets = np.zeros((len(label_sets), len(vocabulary)), dtype=np.float32)
        for row, labels in enumerate(label_sets):
            for label in labels:
                if label in index:
                    targets[row, index[label]] = 1.0
        return targets

    def save(self, path):
        np.savez_compressed(
            path,
            characters=np.array(self.characters),
            nested_vocabulary=np.array(self.nested_vocabulary),
            positioned_vocabulary=np.array(self.positioned_vocabulary),
            nested_targets=self.nested_targets,
            positioned_targets=self.positioned_targets,
            stroke_counts=np.array(self.stroke_counts),
        )
