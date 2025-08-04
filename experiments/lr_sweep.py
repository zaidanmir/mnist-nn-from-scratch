"""Sweep learning rate at fixed architecture and report final test accuracy.

Trains the same MLP (hidden_dim=128) for 10 epochs at batch=128 with
lr in {0.01, 0.03, 0.1, 0.3, 1.0}. Writes results to lr_results.csv.

The expected pattern: too-small lr underfits in 10 epochs, mid-range lr
converges cleanly, too-large lr destabilises (loss explodes or oscillates).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.data import load_mnist
from src.model import MLP
from src.train import train

RESULTS = Path(__file__).resolve().parent / "lr_results.csv"
LRS = [0.01, 0.03, 0.1, 0.3, 1.0]


def main() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    rows = [("lr", "final_train_acc", "final_test_acc", "final_train_loss")]

    for lr in LRS:
        rng = np.random.default_rng(42)
        model = MLP(input_dim=784, hidden_dim=128, output_dim=10, rng=rng)
        print(f"\n=== lr = {lr} ===")
        history = train(
            model, X_train, y_train, X_test, y_test,
            epochs=10, batch_size=128, lr=lr, rng=rng,
        )
        final = history[-1]
        rows.append(
            (lr, f"{final.train_acc:.4f}", f"{final.test_acc:.4f}", f"{final.train_loss:.4f}")
        )

    print(f"\n{'lr':>10}{'train_acc':>14}{'test_acc':>14}{'train_loss':>14}")
    for r in rows[1:]:
        print(f"{r[0]:>10}{r[1]:>14}{r[2]:>14}{r[3]:>14}")

    with RESULTS.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nResults: {RESULTS}")


if __name__ == "__main__":
    main()
