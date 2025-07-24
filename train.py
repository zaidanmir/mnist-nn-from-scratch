"""Top-level training entrypoint: load -> fit -> evaluate -> save metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data import load_mnist
from src.model import MLP
from src.train import train

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.csv",
        help="Filename under runs/ to write per-epoch metrics CSV.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Loaded MNIST: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Config: hidden_dim={args.hidden_dim}  epochs={args.epochs}  "
          f"batch_size={args.batch_size}  lr={args.lr}  seed={args.seed}\n")

    model = MLP(input_dim=784, hidden_dim=args.hidden_dim, output_dim=10, rng=rng)
    history = train(
        model, X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rng=rng,
    )

    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / args.output
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc"])
        for m in history:
            writer.writerow([m.epoch, f"{m.train_loss:.6f}", f"{m.train_acc:.6f}", f"{m.test_acc:.6f}"])

    final = history[-1]
    print(f"\nFinal: train_acc={final.train_acc:.4f}  test_acc={final.test_acc:.4f}")
    print(f"Per-epoch metrics: {out_path}")


if __name__ == "__main__":
    main()
