import torch

from model import ResidualDiffusionDecoder, compute_target_delta
from model import TactileResidualACT


def test_diffusion_residual_is_conditioned_on_act_chunk_and_stays_under_budget():
    decoder = ResidualDiffusionDecoder(
        context_dim=32,
        action_dim=6,
        horizon=8,
        model_dim=32,
        num_heads=4,
        num_layers=1,
        ffn_dim=64,
        diffusion_steps=4,
        residual_scale=0.1,
        residual_clip=0.2,
    )

    assert sum(p.numel() for p in decoder.parameters()) < 1_000_000
    context = torch.zeros(2, 32)
    act_chunk = torch.randn(2, 8, 6)
    noisy_residual = torch.randn(2, 8, 6)
    timestep = torch.zeros(2, dtype=torch.long)

    output = decoder.predict_noise(noisy_residual, timestep, context, act_chunk)
    assert output.shape == noisy_residual.shape


def test_residual_target_is_expert_minus_act_chunk():
    act_chunk = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    expert_action = torch.tensor([[[1.5, 1.0], [2.0, 5.0]]])

    target = compute_target_delta(expert_action, act_chunk)

    assert torch.equal(target, torch.tensor([[[0.5, -1.0], [-1.0, 1.0]]]))


def test_residual_scale_and_clip_bound_sampled_action():
    decoder = ResidualDiffusionDecoder(
        context_dim=8,
        action_dim=2,
        horizon=3,
        model_dim=16,
        num_heads=4,
        num_layers=1,
        ffn_dim=32,
        diffusion_steps=2,
        residual_scale=0.1,
        residual_clip=0.2,
    )
    residual = torch.full((1, 3, 2), 100.0)
    act_chunk = torch.zeros_like(residual)

    output = decoder.apply_residual(act_chunk, residual)

    assert output.abs().max().item() <= 0.02 + 1e-6


def test_action_quantization_uses_configured_unit_and_preserves_zero():
    model = object.__new__(TactileResidualACT)
    model.action_quantization_unit = 1.0
    model.training = False
    action = torch.tensor([[[-0.4, 0.0, 0.6, 1.4, 1.6, -1.6]]])

    quantized = model.quantize_action(action)

    assert torch.equal(
        quantized,
        torch.tensor([[[-0.0, 0.0, 1.0, 1.0, 2.0, -2.0]]]),
    )
