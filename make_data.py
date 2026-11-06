"""Download and preprocess MNIST reproducibly. Usage: python make_data.py"""
import json

from sudoku.config import RESULTS, SPLIT_SEED
from sudoku.data import download_mnist, preprocess

if __name__ == "__main__":
    print("Downloading MNIST (four gzip files, about 11 MiB)...")
    sources = download_mnist()
    counts = preprocess()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data_summary.json").write_text(json.dumps(
        {"split_seed": SPLIT_SEED, "counts_after_dropping_0": counts, "sources": sources}, indent=2))
    print("Images after dropping digit 0:", counts)
