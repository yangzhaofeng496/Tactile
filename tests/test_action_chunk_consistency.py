import torch

from train import action_chunk_overlap_consistency_loss


def test_adjacent_windows_compare_overlapping_action_steps():
    predictions = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0], [5.0]],
        ]
    )
    absolute_indices = torch.tensor([10, 11])
    episode_ids = torch.tensor([4, 4])

    loss = action_chunk_overlap_consistency_loss(
        predictions,
        absolute_indices,
        episode_ids,
    )

    # Window 10's steps 1..3 should match window 11's steps 0..2.
    expected = torch.tensor((1.0**2 + 1.0**2 + 1.0**2) / 3.0)
    assert torch.isclose(loss, expected)


def test_non_adjacent_or_cross_episode_windows_are_ignored():
    predictions = torch.zeros(3, 4, 1)
    absolute_indices = torch.tensor([10, 12, 13])
    episode_ids = torch.tensor([4, 4, 5])

    loss = action_chunk_overlap_consistency_loss(
        predictions,
        absolute_indices,
        episode_ids,
    )

    assert loss.item() == 0.0
