"""Generate the headline training-curves figure for the README.

Runs the default config (hidden=128, lr=0.1, 10 epochs) and saves
figures/training_curves.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_mnist
from src.model import MLP
from src.train import train

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def main() -> None:
    rng = np.random.default_rng(42)
    X_train, y_train, X_test, y_test = load_mnist()
    model = MLP(input_dim=784, hidden_dim=128, output_dim=10, rng=rng)
    history = train(
        model, X_train, y_train, X_test, y_test,
        epochs=10, batch_size=128, lr=0.1, rng=rng,
    )

    epochs = [m.epoch for m in history]
    loss = [m.train_loss for m in history]
    train_acc = [m.train_acc for m in history]
    test_acc = [m.test_acc for m in history]

    FIG_DIR.mkdir(exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, loss, marker="o", color="tab:red")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Mean batch cross-entropy loss")
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    ax_acc.plot(epochs, train_acc, marker="o", label="train", color="tab:blue")
    ax_acc.plot(epochs, test_acc, marker="s", label="test", color="tab:orange")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Train vs test accuracy")
    ax_acc.set_ylim(0.9, 1.0)
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        f"MLP from scratch on MNIST -- final test accuracy {test_acc[-1]:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "training_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
