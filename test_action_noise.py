import torch

from train import add_action_noise


def test_action_noise_is_disabled_for_zero_std():
    action = torch.ones(2, 30, 6)
    noisy = add_action_noise(action, std=0.0)
    assert torch.equal(noisy, action)


def test_action_noise_preserves_shape_and_changes_values():
    torch.manual_seed(7)
    action = torch.zeros(2, 30, 6)
    noisy = add_action_noise(action, std=0.01)
    assert noisy.shape == action.shape
    assert not torch.equal(noisy, action)
    assert noisy.std().item() > 0
