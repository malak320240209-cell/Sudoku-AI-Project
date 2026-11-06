"""MNIST download, verification, parsing and preprocessing."""
import gzip
import hashlib
import struct
import urllib.request

import numpy as np

from .config import DATA_PROCESSED, DATA_RAW, N_TRAIN, N_VAL, SPLIT_SEED

# Original source first. It is often unreachable now, so mirrors of the same files follow.
MIRRORS = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
PROCESSED_FILE = DATA_PROCESSED / "mnist_1to9.npz"


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def download_mnist():
    """Fetch the four gzip files into data/raw, keeping a verified local copy."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, md5 in FILES.items():
        path = DATA_RAW / name
        if path.exists() and _md5(path) == md5:
            sources[name] = "local copy"
            continue
        for base in MIRRORS:
            try:
                request = urllib.request.Request(base + name, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    path.write_bytes(response.read())
            except OSError as error:
                print(f"  {base}: {error}")
                continue
            if _md5(path) == md5:
                sources[name] = base
                print(f"  {name}: {base}")
                break
            path.unlink()
            print(f"  {base}: checksum mismatch, discarded")
        else:
            raise RuntimeError(f"Could not download {name} from any mirror")
    return sources


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: bad magic {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: bad magic {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def preprocess():
    """Split 50,000/10,000, then drop digit 0 from every split. Test writers stay untouched."""
    x = _read_images(DATA_RAW / "train-images-idx3-ubyte.gz")
    y = _read_labels(DATA_RAW / "train-labels-idx1-ubyte.gz")
    x_test = _read_images(DATA_RAW / "t10k-images-idx3-ubyte.gz")
    y_test = _read_labels(DATA_RAW / "t10k-labels-idx1-ubyte.gz")

    order = np.random.default_rng(SPLIT_SEED).permutation(len(x))
    train, val = order[:N_TRAIN], order[N_TRAIN:N_TRAIN + N_VAL]

    def keep(images, labels):
        mask = labels != 0
        return images[mask], labels[mask]

    x_train, y_train = keep(x[train], y[train])
    x_val, y_val = keep(x[val], y[val])
    x_test, y_test = keep(x_test, y_test)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_FILE, x_train=x_train, y_train=y_train, x_val=x_val,
                        y_val=y_val, x_test=x_test, y_test=y_test)
    return {"train": len(y_train), "validation": len(y_val), "test": len(y_test)}


def load_processed():
    """Return images scaled to 0-1 as float32, and labels 1-9."""
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError("Processed data missing. Run: python make_data.py")
    with np.load(PROCESSED_FILE) as d:
        return {k: (d[k].astype(np.float32) / 255.0 if k.startswith("x") else d[k].astype(np.int64))
                for k in d.files}
