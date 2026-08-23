import torch

from model import MultiHeadFusionEncoder
from train import compute_head_router_losses


def test_head_gate_is_per_sample_and_normalized():
    torch.manual_seed(0)
    fusion = MultiHeadFusionEncoder(
        input_dim=216,
        hidden_dim=96,
        output_dim=80,
        num_heads=4,
        dropout=0.0,
    )
    x = torch.randn(2, 216)

    output, metrics = fusion(x, return_attention=True)
    gate = metrics["gate_weights"]

    assert output.shape == (2, 80)
    assert gate.shape == (2, 4)
    assert torch.allclose(gate.sum(dim=-1), torch.ones(2))
    assert not torch.allclose(gate[0], gate[1])


def test_head_gate_is_deterministic_for_identical_inputs():
    torch.manual_seed(0)
    fusion = MultiHeadFusionEncoder(
        input_dim=216,
        hidden_dim=96,
        output_dim=80,
        num_heads=4,
        dropout=0.0,
    )
    x = torch.randn(1, 216)

    _, first = fusion(x, return_attention=True)
    _, second = fusion(x, return_attention=True)

    assert torch.allclose(first["gate_weights"], second["gate_weights"])


def test_head_gate_can_be_disabled():
    fusion = MultiHeadFusionEncoder(
        input_dim=8,
        hidden_dim=16,
        output_dim=8,
        num_heads=4,
        dropout=0.0,
        use_gate=False,
    ).eval()
    x = torch.randn(3, 8)
    _, metrics = fusion(x, return_attention=True)

    expected = metrics["head_weights"].expand(x.shape[0], -1)
    assert torch.allclose(metrics["gate_weights"], expected)


def test_modality_gate_is_per_sample_and_normalized():
    torch.manual_seed(0)
    fusion = MultiHeadFusionEncoder(
        input_dim=10,
        hidden_dim=16,
        output_dim=8,
        num_heads=4,
        dropout=0.0,
        modality_dims=[2, 3, 5],
        use_modality_gate=True,
    )
    x = torch.randn(2, 10)

    _, metrics = fusion(x, return_attention=True)
    modality_gate = metrics["modality_weights"]

    assert modality_gate.shape == (2, 3)
    assert torch.allclose(
        modality_gate.sum(dim=-1), torch.ones(2), atol=1e-6
    )
    assert not torch.allclose(modality_gate[0], modality_gate[1])


def test_modality_gate_residual_scale_preserves_uniform_gate():
    fusion = MultiHeadFusionEncoder(
        input_dim=10,
        hidden_dim=16,
        output_dim=8,
        num_heads=4,
        dropout=0.0,
        modality_dims=[2, 3, 5],
        use_modality_gate=True,
        modality_gate_residual_scale=0.5,
    )
    with torch.no_grad():
        fusion.modality_gate[-1].weight.zero_()
        fusion.modality_gate[-1].bias.zero_()

    x = torch.randn(2, 10)
    _, metrics = fusion(x, return_attention=True)
    assert torch.allclose(
        metrics["modality_weights"],
        torch.full((2, 3), 1.0 / 3.0),
        atol=1e-6,
    )


def test_head_router_losses_are_finite_and_differentiable():
    torch.manual_seed(0)
    fusion = MultiHeadFusionEncoder(
        input_dim=16,
        hidden_dim=16,
        output_dim=8,
        num_heads=8,
        dropout=0.0,
        modality_dims=[4, 4, 4, 4],
        use_gate=True,
        use_modality_gate=True,
    )
    _, attention = fusion(torch.randn(4, 16), return_attention=True)
    balance_loss, diversity_loss = compute_head_router_losses(
        {
            "fusion_gate_weights": attention["gate_weights"],
            "fusion_head_outputs": attention["head_outputs"],
        }
    )
    total = balance_loss + diversity_loss
    total.backward()

    assert torch.isfinite(balance_loss)
    assert torch.isfinite(diversity_loss)
    assert fusion.head_gate[-1].weight.grad is not None
    assert fusion.heads[0][0].weight.grad is not None
