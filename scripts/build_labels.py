"""Build component targets for the ETL9B classes from KanjiVG.

Each labelling rule gets its own file so earlier experiments stay reproducible.
Example: python scripts/build_labels.py --min-document-frequency 5 --output data/component_targets_df5.npz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from kanji_components.kanjivg import ComponentTargets

parser = argparse.ArgumentParser()
parser.add_argument("--min-document-frequency", type=int, default=5)
parser.add_argument("--output", required=True)
arguments = parser.parse_args()

characters = list(np.load("data/etl9b/etl9b_cache.npz", allow_pickle=False)["characters"])
targets = ComponentTargets(characters, min_document_frequency=arguments.min_document_frequency)
targets.save(arguments.output)
print("saved", arguments.output)
print("classes:", len(characters))
print("nested vocabulary:", len(targets.nested_vocabulary), "first 20:", "".join(targets.nested_vocabulary[:20]))
print("positioned vocabulary:", len(targets.positioned_vocabulary), "first 10:", targets.positioned_vocabulary[:10])
print("classes with no nested target:", int((targets.nested_targets.sum(1) == 0).sum()),
      " no positioned target:", int((targets.positioned_targets.sum(1) == 0).sum()))
print("mean positives per class: nested", targets.nested_targets.sum(1).mean().round(2),
      " positioned", targets.positioned_targets.sum(1).mean().round(2))
