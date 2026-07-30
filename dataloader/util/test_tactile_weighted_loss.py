import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import (
    TactileMagnitudeWeightedMSE,
    TactileResidualACT,
    compute_target_delta,
    residual_loss,
)


def load_train_functions(*function_names):
    source_path = PROJECT_ROOT / "train.py"
    source_tree = ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=str(source_path),
    )
    selected_nodes = []
    for node in source_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected_nodes.append(node)

    module = ast.Module(
        body=selected_nodes,
        type_ignores=[],
    )
    namespace = {
        "json": json,
        "Path": Path,
        "warnings": __import__("warnings"),
        "TactileMagnitudeWeightedMSE": TactileMagnitudeWeightedMSE,
        "compute_target_delta": compute_target_delta,
        "residual_loss": residual_loss,
    }
    exec(
        compile(module, filename=str(source_path), mode="exec"),
        namespace,
    )
    return [namespace[name] for name in function_names]


class TactileWeightedLossTests(unittest.TestCase):

    def make_criterion(
        self,
        **overrides,
    ):
        params = {
            "tactile_type": "force",
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

    def test_force_shape_30x12_weighted_loss_backward(self):
        criterion = self.make_criterion()
        pred = torch.randn(2, 30, 6, requires_grad=True)
        target = torch.randn(2, 30, 6)
        tactile = torch.randn(2, 30, 12)
        loss, metrics = criterion(pred, target, tactile)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(torch.isfinite(metrics["weighted_loss"]))
        self.assertTrue(torch.isfinite(metrics["unweighted_loss"]))

    def test_pred_target_shape_mismatch_raises(self):
        criterion = self.make_criterion()
        pred = torch.randn(2, 30, 6)
        target = torch.randn(2, 29, 6)
        tactile = torch.randn(2, 30, 12)
        with self.assertRaisesRegex(ValueError, "pred_delta and target_delta must match"):
            criterion(pred, target, tactile)


class TactileWeightedLossImageTests(unittest.TestCase):

    def make_criterion(
        self,
        **overrides,
    ):
        params = {
            "tactile_type": "image",
            "channel_mean": [0.0] * 6,
            "channel_std": [1.0] * 6,
            "tau": 1.3243,
            "alpha": 2.0,
            "slope": 5.0,
            "eps": 1e-6,
            "tactile_input_already_normalized": False,
            "use_weighted_loss": True,
        }
        params.update(overrides)
        return TactileMagnitudeWeightedMSE(**params)

    def test_image_weighted_loss_with_small_frames(self):
        criterion = self.make_criterion()
        pred = torch.randn(2, 30, 6, requires_grad=True)
        target = torch.randn(2, 30, 6)
        tactile = torch.randn(2, 10, 6, 16, 16)
        loss, metrics = criterion(pred, target, tactile)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(metrics["weighted_loss"].ndim, 0)

    def test_image_weighted_loss_with_already_normalized_input(self):
        criterion = self.make_criterion(
            tactile_input_already_normalized=True,
        )
        tactile = torch.full((1, 10, 6, 16, 16), 2.0)
        magnitude = criterion.compute_window_magnitude(tactile)
        expected = torch.sqrt(torch.tensor(4.0 + 1e-6))
        self.assertTrue(torch.allclose(magnitude, expected.view(1), atol=1e-6, rtol=1e-5))

    def test_image_channel_stats_mismatch_raises(self):
        criterion = self.make_criterion(
            channel_mean=[0.0] * 5,
            channel_std=[1.0] * 5,
        )
        pred = torch.randn(2, 30, 6)
        target = torch.randn(2, 30, 6)
        tactile = torch.randn(2, 10, 6, 16, 16)
        with self.assertRaisesRegex(ValueError, "channel dim must match tactile stats"):
            criterion(pred, target, tactile)

    def test_image_unweighted_mode_backward_and_metrics(self):
        criterion = self.make_criterion(use_weighted_loss=False)
        pred = torch.randn(2, 30, 6, requires_grad=True)
        target = torch.randn(2, 30, 6)
        tactile = torch.randn(2, 10, 6, 16, 16)
        loss, metrics = criterion(pred, target, tactile)
        plain = torch.mean((pred - target) ** 2)
        loss.backward()
        self.assertTrue(torch.allclose(loss, plain.to(loss.dtype), atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(torch.isfinite(metrics["weighted_loss"]))
        self.assertTrue(torch.isfinite(metrics["unweighted_loss"]))


class TrainLossLogicTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (
            cls.load_tactile_stats,
            cls.resolve_tactile_stats_path,
            cls.build_tactile_criterion,
            cls.compute_losses,
        ) = load_train_functions(
            "load_tactile_stats",
            "resolve_tactile_stats_path",
            "build_tactile_criterion",
            "compute_losses",
        )

    def test_compute_losses_image_unweighted_uses_plain_residual_mse(self):
        class DummyModel(nn.Module):
            def __init__(self, pred):
                super().__init__()
                self.pred = nn.Parameter(pred.clone())

            def forward(self, tactile_history, state, act_chunk):
                return self.pred

        pred = torch.randn(2, 30, 6)
        act_chunk = torch.randn(2, 30, 6)
        expert_action = torch.randn(2, 30, 6)
        batch = {
            "tactile_history": torch.randn(2, 10, 6, 16, 16),
            "observation.state": torch.randn(2, 6),
            "act_chunk": act_chunk,
            "expert_action": expert_action,
        }
        criterion = TactileMagnitudeWeightedMSE(
            tactile_type="image",
            channel_mean=[0.0] * 6,
            channel_std=[1.0] * 6,
            tau=1.0,
            use_weighted_loss=False,
        )
        model = DummyModel(pred)
        objective_loss, metrics, pred_delta, target_delta = self.__class__.compute_losses(
            model=model,
            criterion=criterion,
            batch=batch,
        )
        expected = residual_loss(pred_delta, expert_action, act_chunk)
        self.assertTrue(torch.allclose(objective_loss, expected, atol=1e-6, rtol=1e-5))
        self.assertIn("weighted_loss", metrics)
        self.assertIn("unweighted_loss", metrics)
        self.assertTrue(torch.allclose(metrics["unweighted_loss"], expected.detach(), atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(target_delta, expert_action - act_chunk, atol=1e-6, rtol=1e-5))

    def test_build_tactile_criterion_legacy_stats_path_force(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "force_stats.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "channel_mean": [0.0] * 12,
                        "channel_std": [1.0] * 12,
                        "tau_value": 1.1,
                    }
                ),
                encoding="utf-8",
            )
            criterion, metadata = self.__class__.build_tactile_criterion(
                training_cfg={
                    "tactile_stats_path": str(stats_path),
                    "use_tactile_weighted_loss": True,
                },
                tactile_type="force",
                tactile_channels=12,
                action_horizon=30,
                action_dim=6,
            )
        self.assertEqual(criterion.tactile_type, "force")
        self.assertEqual(metadata["tactile_stats_path_source"], "legacy")

    def test_build_tactile_criterion_channel_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "image_stats.json"
            stats_path.write_text(
                json.dumps(
                    {
                        "channel_mean": [0.0] * 12,
                        "channel_std": [1.0] * 12,
                        "tau_value": 1.1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "has 12 channels"):
                self.__class__.build_tactile_criterion(
                    training_cfg={
                        "tactile_stats_paths": {
                            "image": str(stats_path),
                        },
                    },
                    tactile_type="image",
                    tactile_channels=6,
                    action_horizon=30,
                    action_dim=6,
                )


class TactileTrainingSmokeTests(unittest.TestCase):

    def run_smoke_step(
        self,
        tactile_type,
        tactile_history,
        criterion,
    ):
        model = TactileResidualACT(
            tactile_encoder_type=tactile_type,
            action_horizon=30,
            action_dim=6,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        state = torch.randn(tactile_history.shape[0], 6)
        act_chunk = torch.randn(tactile_history.shape[0], 30, 6)
        expert_action = torch.randn(tactile_history.shape[0], 30, 6)

        optimizer.zero_grad(set_to_none=True)
        pred_delta = model(tactile_history, state, act_chunk)
        target_delta = compute_target_delta(expert_action, act_chunk)
        loss, metrics = criterion(
            pred_delta=pred_delta,
            target_delta=target_delta,
            tactile_history=tactile_history,
            act_chunk=act_chunk,
            expert_action=expert_action,
        )
        loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        for value in metrics.values():
            self.assertTrue(torch.isfinite(value))

    def test_force_smoke_step(self):
        criterion = TactileMagnitudeWeightedMSE(
            tactile_type="force",
            channel_mean=[0.0] * 12,
            channel_std=[1.0] * 12,
            tau=1.0,
            use_weighted_loss=True,
        )
        tactile_history = torch.randn(2, 30, 12)
        self.run_smoke_step("force", tactile_history, criterion)

    def test_image_smoke_step(self):
        criterion = TactileMagnitudeWeightedMSE(
            tactile_type="image",
            channel_mean=[0.0] * 6,
            channel_std=[1.0] * 6,
            tau=1.0,
            use_weighted_loss=True,
        )
        tactile_history = torch.randn(2, 10, 6, 16, 16)
        self.run_smoke_step("image", tactile_history, criterion)


if __name__ == "__main__":
    unittest.main()
