import torch

from train import apply_modality_dropout


def test_modality_dropout_masks_selected_samples_only():
    action = torch.ones(2, 3, 1)
    visual = torch.ones(2, 4, 1)
    mask = torch.tensor([True, False])

    dropped_action, dropped_visual = apply_modality_dropout(
        action,
        visual,
        mask,
    )

    assert torch.equal(dropped_action[0], torch.zeros_like(action[0]))
    assert torch.equal(dropped_visual[0], torch.zeros_like(visual[0]))
    assert torch.equal(dropped_action[1], action[1])
    assert torch.equal(dropped_visual[1], visual[1])
