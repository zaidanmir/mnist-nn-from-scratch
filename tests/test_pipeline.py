"""Unit tests for layers, losses, and gradient checks.

Run with: python -m unittest discover tests
"""
from __future__ import annotations

import unittest

import numpy as np

from src.gradcheck import check_gradients
from src.layers import Linear, ReLU
from src.losses import cross_entropy_loss, log_softmax, softmax
from src.model import MLP


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.layer = Linear(4, 3, rng=self.rng)

    def test_forward_shape(self):
        x = self.rng.standard_normal((5, 4)).astype(np.float32)
        y = self.layer.forward(x)
        self.assertEqual(y.shape, (5, 3))

    def test_he_initialisation_scale(self):
        # He init: std should be ~sqrt(2/in_features) = sqrt(2/4) = 0.707
        big = Linear(1000, 100, rng=np.random.default_rng(0))
        std = float(big.W.std())
        expected = np.sqrt(2 / 1000)
        self.assertAlmostEqual(std, expected, places=2)

    def test_bias_initialised_to_zero(self):
        np.testing.assert_array_equal(self.layer.b, np.zeros(3))

    def test_backward_shapes(self):
        x = self.rng.standard_normal((5, 4)).astype(np.float32)
        self.layer.forward(x)
        grad_y = self.rng.standard_normal((5, 3)).astype(np.float32)
        grad_x = self.layer.backward(grad_y)
        self.assertEqual(grad_x.shape, x.shape)
        self.assertEqual(self.layer.dW.shape, self.layer.W.shape)
        self.assertEqual(self.layer.db.shape, self.layer.b.shape)

    def test_backward_before_forward_errors(self):
        with self.assertRaises(AssertionError):
            self.layer.backward(np.zeros((5, 3)))


class TestReLU(unittest.TestCase):
    def test_forward_clamps_negative_to_zero(self):
        relu = ReLU()
        x = np.array([[-1.0, 0.0, 1.0, -2.5, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(relu.forward(x), [[0.0, 0.0, 1.0, 0.0, 3.0]])

    def test_backward_masks_by_input_sign(self):
        relu = ReLU()
        x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        relu.forward(x)
        grad_y = np.array([[5.0, 5.0, 5.0, 5.0]], dtype=np.float32)
        np.testing.assert_array_equal(relu.backward(grad_y), [[0.0, 0.0, 5.0, 5.0]])


class TestLossFunctions(unittest.TestCase):
    def test_log_softmax_handles_huge_logits_without_overflow(self):
        z = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        out = log_softmax(z)
        self.assertTrue(np.all(np.isfinite(out)))
        # Same softmax as if logits were [0, 1, 2] (shift-invariant)
        z_shifted = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(out, log_softmax(z_shifted), atol=1e-5)

    def test_softmax_sums_to_one(self):
        z = np.random.default_rng(0).standard_normal((4, 10)).astype(np.float32)
        s = softmax(z)
        np.testing.assert_allclose(s.sum(axis=-1), np.ones(4), atol=1e-5)

    def test_cross_entropy_grad_rows_sum_to_zero(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((8, 10)).astype(np.float32)
        y = rng.integers(0, 10, size=8)
        _, grad = cross_entropy_loss(logits, y)
        np.testing.assert_allclose(grad.sum(axis=-1), 0.0, atol=1e-5)

    def test_cross_entropy_loss_perfect_predictions(self):
        # If model assigns ~100% probability to the correct class, loss is ~0.
        logits = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
        loss, _ = cross_entropy_loss(logits, np.array([0]))
        self.assertAlmostEqual(loss, 0.0, places=4)


class TestGradientCheck(unittest.TestCase):
    def test_full_gradient_check_passes_on_tiny_model(self):
        rng = np.random.default_rng(0)
        model = MLP(input_dim=20, hidden_dim=8, output_dim=4, rng=rng)
        x = rng.standard_normal((6, 20)).astype(np.float32)
        y = rng.integers(0, 4, size=6)
        results = check_gradients(model, x, y, eps=1e-6)
        for name, info in results.items():
            self.assertLess(
                info["max_rel_err"], 1e-5,
                f"{name} gradient check failed: max rel err = {info['max_rel_err']:.2e}",
            )


if __name__ == "__main__":
    unittest.main()
