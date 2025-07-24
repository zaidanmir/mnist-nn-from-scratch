"""Sweep hidden layer width and report final test accuracy.

Trains the same MLP architecture with hidden_dim in {32, 64, 128, 256} for
10 epochs each at lr=0.1, batch=128. Writes results to experiments/results.csv
and prints a table.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data import load_mnist
from src.model import MLP
from src.train import train

RESULTS = Path(__file__).resolve().parent / "hidden_size_results.csv"
HIDDEN_DIMS = [32, 64, 128, 256]


def main() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    rows = [("hidden_dim", "final_train_acc", "final_test_acc", "final_train_loss")]

    for hidden_dim in HIDDEN_DIMS:
        rng = np.random.default_rng(42)  # same seed -> same data shuffling across runs
        model = MLP(input_dim=784, hidden_dim=hidden_dim, output_dim=10, rng=rng)
        print(f"\n=== hidden_dim = {hidden_dim} ===")
        history = train(
            model, X_train, y_train, X_test, y_test,
            epochs=10, batch_size=128, lr=0.1, rng=rng,
        )
        final = history[-1]
        rows.append(
            (hidden_dim, f"{final.train_acc:.4f}", f"{final.test_acc:.4f}", f"{final.train_loss:.4f}")
        )

    print(f"\n{'hidden_dim':>12}{'train_acc':>14}{'test_acc':>14}{'train_loss':>14}")
    for r in rows[1:]:
        print(f"{r[0]:>12}{r[1]:>14}{r[2]:>14}{r[3]:>14}")

    with RESULTS.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nResults: {RESULTS}")


if __name__ == "__main__":
    main()
