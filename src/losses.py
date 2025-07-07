"""Cross-entropy loss with a numerically stable log-softmax.

See `notes/backprop.md` Section 4 for the gradient derivation. The famous
result is that softmax + cross-entropy collapse to a clean

    dL/dz = (softmax(z) - one_hot(y)) / B

with no Jacobian materialised. The 1/B scaling is applied here so every
downstream layer's backward already operates on the mean-batch gradient.
"""
from __future__ import annotations

import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis.

    Subtracting `max(z)` before exponentiating is mathematically equivalent
    (numerator and denominator both scale by `exp(-max)`) but keeps every
    `exp` call inside float32 range. Without this, any logit above ~89
    overflows.
    """
    z_max = z.max(axis=-1, keepdims=True)
    z_shifted = z - z_max
    log_sum_exp = np.log(np.exp(z_shifted).sum(axis=-1, keepdims=True))
    return z_shifted - log_sum_exp


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    return np.exp(log_softmax(z))


def cross_entropy_loss(
    logits: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean-batch cross-entropy loss and gradient w.r.t. logits.

    Args:
        logits: shape (B, n_classes).
        y: shape (B,) integer class labels.

    Returns:
        loss: scalar mean-batch loss.
        grad_logits: shape (B, n_classes), equal to (softmax(logits) - one_hot(y)) / B.
    """
    B = len(y)
    log_p = log_softmax(logits)

    # Loss = mean over batch of -log P(y_i)
    loss = float(-log_p[np.arange(B), y].mean())

    # Gradient: softmax - one_hot, divided by B for mean-batch.
    grad = np.exp(log_p)        # softmax(logits), avoids recomputing
    grad[np.arange(B), y] -= 1
    grad /= B

    return loss, grad


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Stability test: huge logits should not overflow.
    big = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    print(f"log_softmax of {big[0]} = {log_softmax(big)[0]} (no NaN)")

    # Forward + backward on a small batch.
    B, n_classes = 8, 10
    logits = rng.standard_normal((B, n_classes)).astype(np.float32) * 2
    y = rng.integers(0, n_classes, size=B)

    loss, grad = cross_entropy_loss(logits, y)
    print(f"\nlogits shape={logits.shape}, y shape={y.shape}")
    print(f"loss = {loss:.4f}")
    print(f"grad shape={grad.shape}, sum_per_row=\n{grad.sum(axis=-1)}")

    # Sanity: gradient should sum to zero per row (softmax sums to 1, one_hot sums to 1).
    assert np.allclose(grad.sum(axis=-1), 0.0, atol=1e-6), "rows of grad must sum to 0"
    print("\nrow-sums of grad ≈ 0 ✓  (softmax minus one_hot conserves total probability)")
