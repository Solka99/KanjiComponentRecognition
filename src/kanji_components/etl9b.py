"""Loading the repacked ETL9B mirror into dense numpy arrays."""
import io
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ETL9B_DIR = Path("data/etl9b/repacked_etl9b")
CACHE_PATH = Path("data/etl9b/etl9b_cache.npz")
IMAGE_SIZE = 64
ROWS_PER_SOURCE_FILE = 121_440
SHEET_COUNT_TRAIN, SHEET_COUNT_VALIDATION = 3200, 400


def load_metadata(etl9b_dir=ETL9B_DIR):
    """Metadata rows for the regular 607,200 samples; the extra block at the end of file 5 is dropped."""
    metadata = pd.read_csv(etl9b_dir / "metadata.tsv", sep="\t")
    metadata["source_file"] = metadata["dataset_source"].str.split("/").str[-1]
    row_within_file = metadata.groupby("source_file").cumcount()
    regular = metadata[row_within_file < ROWS_PER_SOURCE_FILE].reset_index(drop=True)
    return regular[["char", "sheet_number", "seek_start", "seek_end", "source_file"]]


def _decode_chunk(arguments):
    images_bin_path, seek_starts, seek_ends = arguments
    decoded = np.zeros((len(seek_starts), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    with open(images_bin_path, "rb") as images_file:
        for index, (start, end) in enumerate(zip(seek_starts, seek_ends)):
            images_file.seek(int(start))
            png_bytes = images_file.read(int(end - start))
            pixels = np.array(Image.open(io.BytesIO(png_bytes)), dtype=np.uint8)
            decoded[index, : pixels.shape[0], : pixels.shape[1]] = pixels
    return decoded


def decode_all_images(metadata, etl9b_dir=ETL9B_DIR, workers=8, chunk_size=20_000):
    """Decode every PNG into a (N, 64, 64) uint8 array of 0/1 values, bottom row zero-padded."""
    images_bin_path = etl9b_dir / "images.bin"
    chunks = [
        (images_bin_path, metadata["seek_start"].values[i : i + chunk_size],
         metadata["seek_end"].values[i : i + chunk_size])
        for i in range(0, len(metadata), chunk_size)
    ]
    with Pool(workers) as pool:
        decoded_chunks = pool.map(_decode_chunk, chunks)
    return np.concatenate(decoded_chunks)


def assign_splits(sheet_numbers, seed=0):
    """Writer-disjoint split: every sheet goes wholly to train, validation or test."""
    unique_sheets = np.sort(np.unique(sheet_numbers))
    shuffled = np.random.default_rng(seed).permutation(unique_sheets)
    train_sheets = set(shuffled[:SHEET_COUNT_TRAIN])
    validation_sheets = set(shuffled[SHEET_COUNT_TRAIN : SHEET_COUNT_TRAIN + SHEET_COUNT_VALIDATION])
    split = np.where(
        np.isin(sheet_numbers, list(train_sheets)), "train",
        np.where(np.isin(sheet_numbers, list(validation_sheets)), "validation", "test"),
    )
    return split


def build_cache(cache_path=CACHE_PATH, etl9b_dir=ETL9B_DIR, seed=0):
    metadata = load_metadata(etl9b_dir)
    images = decode_all_images(metadata, etl9b_dir)
    characters = np.array(sorted(metadata["char"].unique()))
    character_to_index = {character: index for index, character in enumerate(characters)}
    labels = metadata["char"].map(character_to_index).values.astype(np.int16)
    split = assign_splits(metadata["sheet_number"].values, seed)
    np.savez_compressed(
        cache_path,
        packed_images=np.packbits(images.reshape(len(images), -1), axis=1),
        labels=labels,
        sheet_numbers=metadata["sheet_number"].values.astype(np.int16),
        split=split,
        characters=characters,
    )
    return cache_path


class Etl9bCache:
    """In-memory copy of the cache: images (N, 64, 64) uint8 0/1, labels, splits, characters."""

    def __init__(self, cache_path=CACHE_PATH):
        archive = np.load(cache_path, allow_pickle=False)
        packed = archive["packed_images"]
        self.images = np.unpackbits(packed, axis=1)[:, : IMAGE_SIZE * IMAGE_SIZE]
        self.images = self.images.reshape(len(packed), IMAGE_SIZE, IMAGE_SIZE)
        self.labels = archive["labels"].astype(np.int64)
        self.sheet_numbers = archive["sheet_numbers"]
        self.split = archive["split"]
        self.characters = list(archive["characters"])

    def indices(self, split_name):
        return np.flatnonzero(self.split == split_name)

    def subsample_per_class(self, indices, samples_per_class, seed=0):
        """Keep at most `samples_per_class` indices per class, chosen with a fixed seed."""
        if samples_per_class is None:
            return indices
        generator = np.random.default_rng(seed)
        kept = []
        for label in np.unique(self.labels[indices]):
            candidates = indices[self.labels[indices] == label]
            kept.append(generator.choice(candidates, min(samples_per_class, len(candidates)), replace=False))
        return np.sort(np.concatenate(kept))
