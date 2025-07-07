"""Two-layer Multi-Layer Perceptron — Linear -> ReLU -> Linear.

Composes the primitives from `src/layers.py` and works with the loss in
`src/losses.py`. The forward pass returns logits; cross-entropy is computed
externally so the loss + softmax can collapse cleanly into the gradient
expression of `notes/backprop.md` Section 4.

Architecture:

    x (784) -> fc1 (Linear, 784->h) -> ReLU -> fc2 (Linear, h->10) -> logits
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from src.layers import Linear, ReLU


class MLP:
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        *,
        rng: np.random.Generator,
    ) -> None:
        self.fc1 = Linear(input_dim, hidden_dim, rng=rng)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, output_dim, rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = self.fc1.forward(x)
        a1 = self.relu.forward(z1)
        return self.fc2.forward(a1)  # logits, not softmax — see module docstring

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        grad_a1 = self.fc2.backward(grad_logits)
        grad_z1 = self.relu.backward(grad_a1)
        return self.fc1.backward(grad_z1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each input row."""
        return self.forward(x).argmax(axis=-1)

    def params_and_grads(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (param, grad) pairs for the optimiser. Grads are populated by backward()."""
        yield self.fc1.W, self.fc1.dW
        yield self.fc1.b, self.fc1.db
        yield self.fc2.W, self.fc2.dW
        yield self.fc2.b, self.fc2.db


if __name__ == "__main__":
    from src.losses import cross_entropy_loss

    rng = np.random.default_rng(0)
    model = MLP(input_dim=784, hidden_dim=128, output_dim=10, rng=rng)

    x = rng.standard_normal((4, 784)).astype(np.float32)
    y = rng.integers(0, 10, size=4)

    logits = model.forward(x)
    print(f"logits shape: {logits.shape}")

    loss, grad = cross_entropy_loss(logits, y)
    print(f"loss: {loss:.4f}")

    grad_x = model.backward(grad)
    print(f"grad_x shape: {grad_x.shape}")

    for name, (param, g) in zip(["fc1.W", "fc1.b", "fc2.W", "fc2.b"], model.params_and_grads()):
        print(f"  {name:<6} param {param.shape}  grad {g.shape}  ||grad||={np.linalg.norm(g):.4f}")
