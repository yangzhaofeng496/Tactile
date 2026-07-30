import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import TactileMagnitudeWeightedMSE


class TactileWeightedLossTests(unittest.TestCase):

    def make_criterion(
        self,
        **overrides,
    ):
        params = {
            "channel_mean": [0.0] * 12,
            "channel_std": [1.0] * 12,
            "tau": 1.3243,
            "alpha": 2.0,
            "slope": 5.0,
            "eps": 1e-6,
            "tactile_input_already_normalized": False,
            "use_weighted_loss": True,
        }
        params.update(overrides)
        return TactileMagnitudeWeightedMSE(**params)

    def test_shape_output_scalar(self):
        criterion = self.make_criterion()
        pred = torch.randn(4, 5, 6, requires_grad=True)
        target = torch.randn(4, 5, 6)
        tactile = torch.randn(4, 16, 12)
        loss, metrics = criterion(pred, target, tactile)
        self.assertEqual(loss.ndim, 0)
        self.assertIn("weighted_loss", metrics)

    def test_alpha_zero_matches_plain_mse(self):
        criterion = self.make_criterion(alpha=0.0)
        pred = torch.randn(3, 5, 6, requires_grad=True)
        target = torch.randn(3, 5, 6)
        tactile = torch.randn(3, 16, 12)
        weighted_loss, metrics = criterion(pred, target, tactile)
        plain = torch.mean((pred - target) ** 2)
        self.assertTrue(
            torch.allclose(weighted_loss, plain.to(weighted_loss.dtype), atol=1e-6, rtol=1e-5)
        )
        self.assertTrue(
            torch.allclose(metrics["unweighted_loss"], plain.to(metrics["unweighted_loss"].dtype), atol=1e-6, rtol=1e-5)
        )

    def test_high_magnitude_gets_higher_weight(self):
        criterion = self.make_criterion()
        tactile = torch.zeros(2, 16, 12)
        tactile[1] = 4.0
        magnitudes, weights = criterion.compute_window_weights(tactile)
        self.assertGreater(float(magnitudes[1]), float(magnitudes[0]))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_weight_at_tau_is_one_plus_half_alpha(self):
        criterion = self.make_criterion(
            channel_mean=[0.0] * 12,
            channel_std=[1.0] * 12,
            tactile_input_already_normalized=True,
        )
        tactile = torch.full((1, 16, 12), float(criterion.tau.item()))
        _, weights = criterion.compute_window_weights(tactile)
        expected = 1.0 + float(criterion.alpha.item()) / 2.0
        self.assertAlmostEqual(float(weights.item()), expected, places=4)

    def test_weight_normalization_invariance(self):
        criterion = self.make_criterion()
        losses = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
        weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        base = criterion.reduce_weighted_losses(
            losses,
            weights,
            criterion.eps,
        )
        scaled = criterion.reduce_weighted_losses(
            losses,
            weights * 17.0,
            criterion.eps,
        )
        self.assertTrue(torch.allclose(base, scaled, atol=1e-6, rtol=1e-6))

    def test_gradients_only_flow_to_prediction(self):
        criterion = self.make_criterion()
        pred = torch.randn(2, 5, 6, requires_grad=True)
        target = torch.randn(2, 5, 6)
        tactile = torch.randn(2, 16, 12, requires_grad=True)
        loss, _ = criterion(pred, target, tactile)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertIsNone(tactile.grad)

    def test_numerical_stability(self):
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        criterion = self.make_criterion(
            channel_std=[1e-8] * 12,
            tau=1e-6,
            eps=1e-6,
        )
        pred = torch.randn(2, 5, 6, dtype=dtype, requires_grad=True)
        target = torch.randn(2, 5, 6, dtype=dtype)
        tactile = torch.randn(2, 16, 12, dtype=dtype)
        loss, metrics = criterion(pred, target, tactile)
        self.assertTrue(torch.isfinite(loss))
        for value in metrics.values():
            self.assertTrue(torch.isfinite(value))


if __name__ == "__main__":
    unittest.main()
