import torch

from train import apply_fixed_modality_ablation


def test_fixed_modality_ablation_zeros_selected_inputs_only():
    batch = {
        "current_force": torch.ones(2, 12),
        "observation.state": torch.ones(2, 6),
        "act_visual_tokens": torch.ones(2, 512),
        "act_chunk": torch.ones(2, 30, 6),
    }
    result = apply_fixed_modality_ablation(batch, {"current_force", "state"})
    assert torch.count_nonzero(result["current_force"]) == 0
    assert torch.count_nonzero(result["observation.state"]) == 0
    assert torch.count_nonzero(result["act_visual_tokens"]) > 0
    assert torch.count_nonzero(result["act_chunk"]) > 0
