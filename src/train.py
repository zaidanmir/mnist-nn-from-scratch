"""Mini-batch SGD training loop for the MLP.

The update rule comes straight from `notes/backprop.md` Section 7:

    theta <- theta - lr * dL/dtheta

with `dL/dtheta` computed by `model.backward()` (already mean-batch-scaled
by the loss layer) and applied to every parameter the model exposes via
`params_and_grads()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import iterate_minibatches
from src.losses import cross_entropy_loss
from src.model import MLP


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    test_acc: float


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray, batch_size: int = 1024) -> float:
    """Compute classification accuracy in batches to avoid materialising large activations."""
    correct = 0
    for start in range(0, len(X), batch_size):
        preds = model.predict(X[start : start + batch_size])
        correct += int((preds == y[start : start + batch_size]).sum())
    return correct / len(X)


def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.1,
    rng: np.random.Generator,
    verbose: bool = True,
) -> list[EpochMetrics]:
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, rng=rng):
            logits = model.forward(X_batch)
            loss, grad_logits = cross_entropy_loss(logits, y_batch)
            model.backward(grad_logits)
            for param, grad in model.params_and_grads():
                param -= lr * grad
            epoch_loss += loss
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
        metrics = EpochMetrics(epoch, train_loss, train_acc, test_acc)
        history.append(metrics)

        if verbose:
            print(
                f"epoch {epoch:>2}/{epochs}  "
                f"loss={train_loss:.4f}  "
                f"train_acc={train_acc:.4f}  "
                f"test_acc={test_acc:.4f}"
            )

    return history


if __name__ == "__main__":
    from src.data import load_mnist

    rng = np.random.default_rng(42)
    X_train, y_train, X_test, y_test = load_mnist()

    model = MLP(input_dim=784, hidden_dim=128, output_dim=10, rng=rng)
    train(model, X_train, y_train, X_test, y_test, epochs=3, batch_size=128, lr=0.1, rng=rng)
