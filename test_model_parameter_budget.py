from pathlib import Path

import torch
import yaml

from model import TactileResidualACT


ROOT = Path(__file__).resolve().parent


def build_model_from_configs():
    with (ROOT / "config/model_config.yaml").open() as f:
        model_config = yaml.safe_load(f)
    with (ROOT / "dataloader/util/tactile_analysis/state_stats.json").open() as f:
        state_stats = yaml.safe_load(f)

    tactile_type = model_config["tactile_encoder"]["type"]
    decoder_cfg = model_config["decoder"]
    return TactileResidualACT(
        tactile_encoder_type=tactile_type,
        action_horizon=decoder_cfg["action_horizon"],
        action_dim=decoder_cfg["action_dim"],
        use_tactile_history=model_config["tactile_encoder"].get("enabled", True),
        tactile_encoder_cfg=model_config["tactile_encoder"][tactile_type],
        state_encoder_cfg=model_config["state_encoder"],
        action_encoder_cfg=model_config["action_encoder"],
        current_force_encoder_cfg=model_config["current_force_encoder"],
        fusion_cfg=model_config["fusion"],
        decoder_cfg=decoder_cfg,
        state_mean=state_stats["mean"],
        state_std=state_stats["std"],
        state_channel_names=state_stats["channel_names"],
        normalize_state_input=model_config["state_encoder"].get("normalize_input", True),
        use_act_visual=model_config["act_visual"].get("enabled", False),
        visual_encoder_cfg=model_config.get("act_visual"),
    )


def test_residual_model_stays_within_reduced_parameter_budget():
    model = build_model_from_configs()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 90_000 <= trainable_params <= 115_000


def test_reduced_residual_model_preserves_prediction_shape():
    model = build_model_from_configs().eval()
    with torch.no_grad():
        prediction = model(
            tactile_history=None,
            current_force=torch.zeros(2, 12),
            state=torch.zeros(2, 6),
            act_chunk=torch.zeros(2, 30, 6),
            act_visual_tokens=torch.zeros(2, 512),
        )
    assert prediction.shape == (2, 30, 6)
