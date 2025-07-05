"""MNIST IDX loader, normalisation, and mini-batch iterator.

The dataset is fetched in its original IDX format (4 gzipped files: train/test
images and labels). Pixel values are scaled into [0, 1] and flattened from
28x28 to a 784-vector per sample, ready to feed into the input layer.
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# IDX format magic numbers — see http://yann.lecun.com/exdb/mnist/
_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049


def download_if_missing(target_dir: Path = DATA_DIR) -> None:
    """Download any of the four MNIST gzip files that aren't already on disk."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        path = target_dir / fname
        if not path.exists():
            urllib.request.urlretrieve(MIRROR + fname, path)


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != _IMAGE_MAGIC:
            raise ValueError(f"Bad image magic in {path}: {magic}")
        buffer = f.read()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(n, rows * cols)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != _LABEL_MAGIC:
            raise ValueError(f"Bad label magic in {path}: {magic}")
        buffer = f.read()
    return np.frombuffer(buffer, dtype=np.uint8).copy()


def load_mnist(
    target_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test). Pixels normalised to [0, 1]."""
    download_if_missing(target_dir)
    X_train = _read_images(target_dir / FILES["train_images"]).astype(np.float32) / 255.0
    y_train = _read_labels(target_dir / FILES["train_labels"]).astype(np.int64)
    X_test = _read_images(target_dir / FILES["test_images"]).astype(np.float32) / 255.0
    y_test = _read_labels(target_dir / FILES["test_labels"]).astype(np.int64)
    return X_train, y_train, X_test, y_test


def iterate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (X_batch, y_batch) pairs. Shuffles via `rng` if provided."""
    n = len(X)
    if rng is not None:
        order = rng.permutation(n)
        X, y = X[order], y[order]
    for start in range(0, n, batch_size):
        yield X[start : start + batch_size], y[start : start + batch_size]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"X_train: shape={X_train.shape}, dtype={X_train.dtype}")
    print(f"y_train: shape={y_train.shape}, classes={np.unique(y_train)}")
    print(f"X_test:  shape={X_test.shape}")
    print(f"pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"class balance (train): {np.bincount(y_train)}")

    rng = np.random.default_rng(0)
    n_batches = sum(1 for _ in iterate_minibatches(X_train, y_train, 128, rng=rng))
    print(f"batches per epoch (batch_size=128): {n_batches}")
