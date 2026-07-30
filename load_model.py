import torch
import yaml
from pathlib import Path
from model import TactileResidualACT


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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

    # 获取触觉编码器类型
    encoder_type = config['tactile_encoder']['type']
    action_horizon = int(config['decoder']['action_horizon'])
    action_dim = int(config['decoder']['action_dim'])

    print(f"创建模型，使用触觉编码器类型: {encoder_type}")

    # 创建模型
    model = TactileResidualACT(
        tactile_encoder_type=encoder_type,
        action_horizon=action_horizon,
        action_dim=action_dim,
        action_encoder_cfg=config.get('action_encoder'),
        decoder_cfg=config.get('decoder'),
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
