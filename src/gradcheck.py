"""Numerical gradient checking for the MLP.

This is the test that catches backprop bugs before training does. We compare
the analytic gradient computed by `model.backward()` against a centered
finite-difference gradient

    g_num[i] = (L(θ + ε e_i) − L(θ − ε e_i)) / (2 ε)

on a small toy model where the full O(|θ|) finite-difference loop is fast.
The analytic and numerical gradients should agree up to float32 precision.
"""
from __future__ import annotations

import numpy as np

from src.losses import cross_entropy_loss
from src.model import MLP


def numerical_gradient(
    loss_fn,
    param: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Compute the numerical gradient of `loss_fn()` w.r.t. every entry of `param`.

    Uses centered finite differences. The function `loss_fn` should mutate
    nothing other than its own internal state — gradients are computed by
    perturbing `param` in place and calling `loss_fn` repeatedly.

    Cost: 2 × `param.size` calls to `loss_fn`. Use only on small models.
    """
    grad = np.zeros_like(param, dtype=np.float64)
    for idx in np.ndindex(*param.shape):
        original = float(param[idx])
        param[idx] = original + eps
        f_plus = float(loss_fn())
        param[idx] = original - eps
        f_minus = float(loss_fn())
        param[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad.astype(param.dtype)


def check_gradients(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-6,
) -> dict[str, dict]:
    """Run forward + backward, then compare analytic to numerical grads.

    Casts the model parameters and inputs to float64 in-place so the centered
    finite-difference loop avoids float32 cancellation noise. Caller should
    discard the model after this; it's mutated.
    """
    # Upcast for tight precision.
    model.fc1.W = model.fc1.W.astype(np.float64)
    model.fc1.b = model.fc1.b.astype(np.float64)
    model.fc2.W = model.fc2.W.astype(np.float64)
    model.fc2.b = model.fc2.b.astype(np.float64)
    x = x.astype(np.float64)

    logits = model.forward(x)
    _, grad_logits = cross_entropy_loss(logits, y)
    model.backward(grad_logits)

    analytic = {
        "fc1.W": model.fc1.dW.copy(),
        "fc1.b": model.fc1.db.copy(),
        "fc2.W": model.fc2.dW.copy(),
        "fc2.b": model.fc2.db.copy(),
    }
    params = {
        "fc1.W": model.fc1.W,
        "fc1.b": model.fc1.b,
        "fc2.W": model.fc2.W,
        "fc2.b": model.fc2.b,
    }

    def loss_fn():
        return cross_entropy_loss(model.forward(x), y)[0]

    results = {}
    for name, param in params.items():
        numerical = numerical_gradient(loss_fn, param, eps=eps)
        a, n = analytic[name].astype(np.float64), numerical.astype(np.float64)
        # Relative error per element with a small floor to avoid 0/0.
        rel_err = np.abs(a - n) / (np.abs(a) + np.abs(n) + 1e-8)
        results[name] = {
            "max_rel_err": float(rel_err.max()),
            "mean_rel_err": float(rel_err.mean()),
            "shape": tuple(param.shape),
        }
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Tiny model so the full O(|theta|) finite-difference loop is fast.
    model = MLP(input_dim=20, hidden_dim=8, output_dim=4, rng=rng)
    x = rng.standard_normal((6, 20)).astype(np.float32)
    y = rng.integers(0, 4, size=6)

    print("Running gradient check on a (20 -> 8 -> 4) model with batch=6...")
    print("(Parameters upcast to float64 for tight finite-difference precision.)\n")
    results = check_gradients(model, x, y, eps=1e-6)

    print(f"{'param':<8}{'shape':<14}{'max rel err':>14}{'mean rel err':>16}")
    all_pass = True
    for name, info in results.items():
        passed = info["max_rel_err"] < 1e-5
        flag = "PASS" if passed else "FAIL"
        all_pass &= passed
        print(f"{name:<8}{str(info['shape']):<14}{info['max_rel_err']:>14.2e}{info['mean_rel_err']:>16.2e}  {flag}")

    print(f"\n{'ALL PASS' if all_pass else 'FAIL'}: max relative error < 1e-5 on all parameters.")
