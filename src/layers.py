"""Layer primitives: Linear (affine) and ReLU activation.

Each layer is a small class with a `forward()` / `backward()` pair, mirroring
the chain-rule structure laid out in `notes/backprop.md`. The forward pass
caches whatever the backward pass needs — there is no autograd.

Convention. Each `backward()` receives `grad_y = dL/d(output)` and returns
`dL/d(input)`. Parameter gradients (`dW`, `db`) are stored on the layer.
The loss layer is responsible for scaling its initial `grad_y` by `1/B` so
that everything downstream is the gradient of the **mean** batch loss.
"""
from __future__ import annotations

import numpy as np


class Linear:
    """Affine layer: ``y = x W + b`` with x shaped (B, in_features).

    Weights are initialised with He scaling (``sqrt(2/in_features)``), which
    is standard for ReLU networks; biases start at zero. See
    notes/backprop.md sections 4–5 for the gradient formulas.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rng: np.random.Generator,
    ) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.W = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self._x_cache: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_cache = x
        return x @ self.W + self.b

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        x = self._x_cache
        assert x is not None, "Linear.backward called before forward"
        self.dW = x.T @ grad_y          # (in_features, out_features)
        self.db = grad_y.sum(axis=0)    # (out_features,)
        return grad_y @ self.W.T        # (B, in_features)

    @property
    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.W, self.b

    @property
    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dW, self.db


class ReLU:
    """Elementwise ``y = max(0, x)``.

    Backward: ``dL/dx = dL/dy * 1[x > 0]``. See notes/backprop.md section 5.
    """

    def __init__(self) -> None:
        self._mask_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask_cache = x > 0
        return x * self._mask_cache

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        mask = self._mask_cache
        assert mask is not None, "ReLU.backward called before forward"
        return grad_y * mask


if __name__ == "__main__":
    # Smoke test: shapes line up forward and back through Linear -> ReLU.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5)).astype(np.float32)

    linear = Linear(5, 3, rng=rng)
    relu = ReLU()

    z = linear.forward(x)
    a = relu.forward(z)
    print(f"forward:  x{x.shape} -> z{z.shape} -> a{a.shape}")

    grad_a = np.ones_like(a)
    grad_z = relu.backward(grad_a)
    grad_x = linear.backward(grad_z)
    print(f"backward: dL/da{grad_a.shape} -> dL/dz{grad_z.shape} -> dL/dx{grad_x.shape}")
    print(f"params:   W{linear.W.shape}, b{linear.b.shape}")
    print(f"grads:    dW{linear.dW.shape}, db{linear.db.shape}")
