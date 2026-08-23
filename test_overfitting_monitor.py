from train import (
    OverfitMonitor,
    add_random_gain,
    add_relative_gaussian_noise,
    resolve_current_force_normalization,
)
from dataloader.cache_loader import (
    CachedResidualDataset,
    EpisodeRandomWindowSampler,
)
from dataloader.dataloader import DatasetKeys
from model import ForceFiLM


def test_detector_waits_for_three_degraded_test_epochs():
    monitor = OverfitMonitor(patience=3, min_relative_increase=0.01)
    assert not monitor.update(10.0, 9.0, 8.0)
    assert not monitor.update(9.0, 8.5, 8.1)
    assert not monitor.update(8.0, 8.0, 8.2)
    assert monitor.update(7.0, 7.5, 8.15)


def test_detector_counts_a_small_recovery_while_still_above_best():
    monitor = OverfitMonitor(patience=3, min_relative_increase=0.01)
    assert not monitor.update(10.0, 9.0, 8.0)
    assert not monitor.update(9.0, 8.5, 8.2)
    assert not monitor.update(8.0, 8.0, 8.3)
    assert monitor.update(7.0, 7.5, 8.25)


def test_detector_does_not_stop_when_test_rise_is_below_threshold():
    monitor = OverfitMonitor(patience=3, min_relative_increase=0.01)
    for value in (8.0, 8.01, 8.02, 8.03, 8.04):
        assert not monitor.update(7.0, 7.0, value)


def test_detector_serializes_epoch_metrics():
    monitor = OverfitMonitor(patience=3, min_relative_increase=0.01)
    monitor.update(10.0, 9.0, 8.0)
    assert monitor.history[0]["epoch"] == 1
    assert monitor.history[0]["test_loss"] == 8.0


def test_cached_dataset_stride_reduces_redundant_windows_per_episode():
    keys = DatasetKeys(
        tactile_type="vqvae",
        tactile_force=["force"],
        current_force=["force"],
        state="state",
        expert_action="action",
    )
    cache = {
        index: {
            "tactile_history": None,
            "current_force": None,
            "state": None,
            "act_chunk": None,
            "expert_action": None,
        }
        for index in range(10)
    }
    dataset = CachedResidualDataset(
        cache=cache,
        episode_bounds={0: (0, 10)},
        episode_ids=[0],
        keys=keys,
        tactile_history=1,
        action_horizon=0,
        window_stride=3,
    )
    assert dataset.valid_indices == [0, 3, 6, 9]


def test_episode_sampler_represents_each_episode():
    keys = DatasetKeys(
        tactile_type="vqvae",
        tactile_force=["force"],
        current_force=["force"],
        state="state",
        expert_action="action",
    )
    cache = {
        index: {"tactile_history": None, "current_force": None,
                "state": None, "act_chunk": None, "expert_action": None}
        for index in range(20)
    }
    dataset = CachedResidualDataset(
        cache=cache,
        episode_bounds={0: (0, 10), 1: (10, 20)},
        episode_ids=[0, 1],
        keys=keys,
        tactile_history=1,
        action_horizon=0,
    )
    sampler = EpisodeRandomWindowSampler(dataset, fraction=0.25, seed=42)
    sampled = list(sampler)
    assert len(sampled) == 4
    assert any(index < 10 for index in sampled)
    assert any(index >= 10 for index in sampled)


def test_episode_sampler_can_balance_force_extremes():
    keys = DatasetKeys(
        tactile_type="vqvae", tactile_force=["force"],
        current_force=["force"], state="state", expert_action="action",
    )
    cache = {
        index: {
            "tactile_history": None,
            "current_force": __import__("torch").tensor([[float(index)]]),
            "state": None, "act_chunk": None, "expert_action": None,
        }
        for index in range(10)
    }
    dataset = CachedResidualDataset(
        cache=cache, episode_bounds={0: (0, 10)}, episode_ids=[0],
        keys=keys, tactile_history=1, action_horizon=0,
    )
    sampler = EpisodeRandomWindowSampler(
        dataset, fraction=0.4, seed=42, balance_current_force=True
    )
    sampled = list(sampler)
    assert len(sampled) == 4
    assert min(sampled) < 5
    assert max(sampled) >= 5


def test_sampler_hard_replay_adds_high_force_low_action_examples():
    import torch

    keys = DatasetKeys(
        tactile_type="vqvae", tactile_force=["force"],
        current_force=["force"], state="state", expert_action="action",
    )
    cache = {}
    for index in range(20):
        cache[index] = {
            "tactile_history": None,
            "current_force": torch.tensor([[10.0 if index >= 18 else 1.0]]),
            "state": None, "act_chunk": None,
            "expert_action": torch.tensor([[1.0 if index >= 18 else 10.0]]),
        }
    dataset = CachedResidualDataset(
        cache=cache, episode_bounds={0: (0, 20)}, episode_ids=[0],
        keys=keys, tactile_history=1, action_horizon=0,
    )
    sampler = EpisodeRandomWindowSampler(
        dataset, fraction=0.25, seed=42, hard_replay_fraction=0.5
    )
    sampled = list(sampler)
    assert len(sampled) > 5
    assert any(index >= 18 for index in sampled)


def test_relative_force_noise_preserves_shape_and_zero_setting():
    import torch

    values = torch.ones(4, 3)
    assert torch.equal(add_relative_gaussian_noise(values, 0.0), values)
    noisy = add_relative_gaussian_noise(values, 0.05)
    assert noisy.shape == values.shape


def test_random_gain_preserves_shape_and_zero_setting():
    import torch

    values = torch.ones(8, 12)
    assert torch.equal(add_random_gain(values, 0.0), values)
    scaled = add_random_gain(values, 0.2)
    assert scaled.shape == values.shape
    assert float(scaled.min()) >= 0.8
    assert float(scaled.max()) <= 1.2


def test_current_force_normalization_uses_tactile_stats_when_enabled():
    import torch

    model_config = {"current_force_encoder": {"normalize_input": True}}
    metadata = {"stats_payload": {
        "channel_mean": [0.0] * 12,
        "channel_std": [1.0] * 12,
    }}
    mean, std, enabled = resolve_current_force_normalization(
        model_config, metadata
    )
    assert enabled
    assert torch.equal(mean, torch.zeros(12))
    assert torch.equal(std, torch.ones(12))


def test_force_film_is_identity_at_initialization():
    import torch

    film = ForceFiLM(condition_dim=4, feature_dim=6, hidden_dim=8)
    features = torch.randn(3, 6)
    condition = torch.randn(3, 4)
    assert torch.allclose(film(features, condition), features)
