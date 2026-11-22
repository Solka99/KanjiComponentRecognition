"""Decode the ETL9B mirror into data/etl9b/etl9b_cache.npz (images, labels, writer-disjoint split)."""
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from kanji_components.etl9b import build_cache, Etl9bCache

started = time.time()
path = build_cache()
cache = Etl9bCache(path)
print(f"wrote {path} in {time.time() - started:.0f}s")
print("images:", cache.images.shape, "classes:", len(cache.characters))
for split_name in ("train", "validation", "test"):
    indices = cache.indices(split_name)
    sheet_count = len(np.unique(cache.sheet_numbers[indices]))
    print(f"{split_name}: {len(indices)} samples, {sheet_count} sheets")
