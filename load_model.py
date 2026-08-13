import json
from pathlib import Path

import torch
import yaml
from model import TactileResidualACT


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model_checkpoint(checkpoint_path, device="cuda"):
    """
    从checkpoint加载模型

    Args:
        checkpoint_path: checkpoint文件路径
        device: 模型加载设备

    Returns:
        dict: {"model": 模型实例}
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config_snapshot = checkpoint.get("config")
    if isinstance(config_snapshot, dict) and "model_config" in config_snapshot:
        config_snapshot = config_snapshot["model_config"]
    if config_snapshot is None:
        config_snapshot = load_config(
            str(Path(__file__).parent / "config" / "model_config.yaml")
        )

    model, _ = create_model_from_config_dict(config_snapshot)

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    return {"model": model}


def create_model_from_config(config_path):
    """
    从配置文件创建模型

    Args:
        config_path: 配置文件路径

    Returns:
        model: TactileResidualACT 模型实例
        config: 配置字典
    """
    config = load_config(config_path)
    return create_model_from_config_dict(config)


def create_model_from_config_dict(config):
    """
    从配置字典创建模型

    Args:
        config: 配置字典

    Returns:
        model: TactileResidualACT 模型实例
        config: 配置字典
    """
    # 获取触觉编码器类型
    encoder_type = config['tactile_encoder']['type']
    action_horizon = int(config['decoder']['action_horizon'])
    action_dim = int(config['decoder']['action_dim'])
    encoder_config = config['tactile_encoder'][encoder_type]
    state_config = config['state_encoder']
    training_config = config['training']
    stats_path = training_config['tactile_stats_paths'][encoder_type]
    if not stats_path:
        raise ValueError(
            f"No tactile statistics configured for encoder type {encoder_type!r}."
        )
    with Path(stats_path).open('r', encoding='utf-8') as file:
        tactile_stats = json.load(file)
    channel_names = tactile_stats.get('channel_names')
    if encoder_type == 'force':
        expected_order = encoder_config.get('channel_order')
        if channel_names != expected_order:
            raise ValueError(
                "Force channel order mismatch between model config and stats: "
                f"model={expected_order}, stats={channel_names}."
            )
    state_stats_path = state_config.get('stats_path')
    if not state_stats_path:
        raise ValueError('state_encoder.stats_path is required.')
    with Path(state_stats_path).open('r', encoding='utf-8') as file:
        state_stats = json.load(file)
    if state_stats.get('channel_names') != state_config.get('channel_order'):
        raise ValueError(
            'State channel order mismatch between model config and statistics.'
        )
    if int(state_config['input_dim']) != action_dim:
        raise ValueError(
            'state_encoder.input_dim must match decoder.action_dim.'
        )

    print(f"创建模型，使用触觉编码器类型: {encoder_type}")

    # 创建模型
    model = TactileResidualACT(
        tactile_encoder_type=encoder_type,
        action_horizon=action_horizon,
        action_dim=action_dim,
        tactile_encoder_cfg=encoder_config,
        state_encoder_cfg=config.get('state_encoder'),
        current_force_encoder_cfg=config.get('current_force_encoder'),
        action_encoder_cfg=config.get('action_encoder'),
        fusion_cfg=config.get('fusion'),
        decoder_cfg=config.get('decoder'),
        tactile_channel_mean=tactile_stats['channel_mean'],
        tactile_channel_std=tactile_stats['channel_std'],
        tactile_channel_names=channel_names,
        normalize_tactile_input=not bool(
            training_config.get('tactile_input_already_normalized', False)
        ),
        state_mean=state_stats['mean'],
        state_std=state_stats['std'],
        state_channel_names=state_stats['channel_names'],
        normalize_state_input=bool(
            state_config.get('normalize_input', True)
        ),
        use_act_visual=bool(
            config.get('act_visual', {}).get('enabled', False)
        ),
        visual_encoder_cfg=config.get('act_visual'),
    )

    return model, config


if __name__ == "__main__":

    config_path = "config/model_config.yaml"

    # 从配置文件创建模型
    model, config = create_model_from_config(config_path)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"设备: {device}")
    print(f"触觉编码器类型: {config['tactile_encoder']['type']}")

    # 根据编码器类型准备测试数据
    B = config['training']['batch_size']

    if config['tactile_encoder']['type'] == "force":
        history_length = config['tactile_encoder']['force']['history_length']
        input_dim = config['tactile_encoder']['force']['input_dim']
        tactile_history = torch.randn(B, history_length, input_dim).to(device)
        print(f"触觉输入形状: {tactile_history.shape}")

    elif config['tactile_encoder']['type'] == "image":
        history_length = config['tactile_encoder']['image']['history_length']
        in_channels = config['tactile_encoder']['image']['in_channels']
        H, W = config['tactile_encoder']['image']['image_size']
        tactile_history = torch.randn(B, history_length, in_channels, H, W).to(device)
        print(f"触觉输入形状: {tactile_history.shape}")

    state = torch.randn(B, 6).to(device)
    act_chunk = torch.randn(
        B,
        config['decoder']['action_horizon'],
        config['decoder']['action_dim'],
    ).to(device)

    # 前向传播测试
    with torch.no_grad():
        delta_pred = model(tactile_history, state, act_chunk)

    print(f"残差预测形状: {delta_pred.shape}")

    # 统计参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {params / 1e6:.3f} M")

    print("\n✅ 模型创建和测试成功！")
