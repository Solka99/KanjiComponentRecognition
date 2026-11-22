"""Choose kanji to hide from training for the zero-shot experiment; writes data/heldout_characters.txt.

A character qualifies when it has at least two positioned components, its combined component
vector is unique among all classes, and every one of its component labels stays present in at
least MIN_REMAINING other training characters after removal.
"""
import numpy as np

HELDOUT_COUNT = 300
MIN_REMAINING = 3
SEED = 0

archive = np.load("data/component_targets.npz", allow_pickle=False)
characters = list(archive["characters"])
combined = np.concatenate([archive["nested_targets"], archive["positioned_targets"]], axis=1) > 0.5
positioned_count = archive["positioned_targets"].sum(1)

_, first_index, vector_counts = np.unique(combined, axis=0, return_index=True, return_counts=True)
unique_rows = set(first_index[vector_counts == 1])
candidates = [i for i, character in enumerate(characters)
              if ord(character) >= 0x4E00 and positioned_count[i] >= 2 and i in unique_rows]
print(f"candidates {len(candidates)} of {len(characters)}")

remaining = combined.sum(0).astype(int)
heldout = []
for index in np.random.default_rng(SEED).permutation(candidates):
    labels = np.flatnonzero(combined[index])
    if (remaining[labels] > MIN_REMAINING).all():
        heldout.append(index)
        remaining[labels] -= 1
    if len(heldout) == HELDOUT_COUNT:
        break
heldout_characters = [characters[i] for i in sorted(heldout)]
with open("data/heldout_characters.txt", "w") as handle:
    handle.write("".join(heldout_characters) + "\n")
print(f"held out {len(heldout_characters)}: {''.join(heldout_characters[:40])}...")
