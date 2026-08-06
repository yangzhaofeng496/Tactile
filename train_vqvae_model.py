"""
使用VQ-VAE集成架构和修改后dataloader的完整训练脚本
"""

import sys
sys.path.insert(0, '/home/yang/TactileEncoder/dataloader')

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
import wandb

from model import TactileResidualACT, residual_loss
from dataloader import (
    load_yaml,
    set_seed,
    build_base_dataset,
    build_normal_dataloaders,
    load_lerobot_policy,
    build_augmented_loaders,
)


def load_model_from_config(model_config_path):
    """从配置文件加载VQ-VAE集成模型"""

    with open(model_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 加载状态归一化统计信息
    state_mean = None
    state_std = None
    state_channel_names = None
    if config['state_encoder'].get('normalize_input', False):
        stats_path = config['state_encoder'].get('stats_path')
        if stats_path:
            import json
            with open(stats_path, 'r') as f:
                state_stats = json.load(f)
            state_mean = state_stats['mean']
            state_std = state_stats['std']
            state_channel_names = state_stats.get('channel_names')

    # 使用vqvae配置
    model = TactileResidualACT(
        tactile_encoder_type="vqvae",
        action_horizon=config['decoder']['action_horizon'],
        action_dim=config['decoder']['action_dim'],
        tactile_encoder_cfg=config['tactile_encoder']['vqvae'],
        current_force_encoder_cfg=config['current_force_encoder'],
        state_encoder_cfg=config['state_encoder'],
        action_encoder_cfg=config['action_encoder'],
        fusion_cfg=config['fusion'],
        decoder_cfg=config['decoder'],
        normalize_tactile_input=False,
        normalize_current_force_input=config['current_force_encoder'].get('normalize_input', False),
        normalize_state_input=config['state_encoder'].get('normalize_input', False),
        state_mean=state_mean,
        state_std=state_std,
        state_channel_names=state_channel_names,
    )

    return model, config


def compute_per_axis_mse(final_action, expert_action):
    """计算每个轴的MSE

    Args:
        final_action: 最终预测动作 (ACT预测 + 残差) [B, T, 6]
        expert_action: 专家动作 [B, T, 6]

    Returns:
        per_axis_mse: 6个轴的MSE [6]
    """
    # 计算每个轴的MSE: 对batch和time维度求平均
    per_axis_mse = ((final_action - expert_action) ** 2).mean(dim=(0, 1))  # [6]
    return per_axis_mse


def train_epoch(model, dataloader, optimizer, device, epoch, global_step):
    """训练一个epoch"""

    model.train()
    total_loss = 0.0
    num_batches = 0

    # 累积指标
    epoch_per_axis_mse_final = []
    epoch_per_axis_mse_act = []
    epoch_modality_contributions = {
        'tactile': [],
        'current_force': [],
        'state': [],
        'action': []
    }

    for batch_idx, batch in enumerate(dataloader):
        # 提取数据
        tactile_history = batch['tactile_history']
        current_force = batch['current_force']
        state = batch['observation.state']  # ACT observation key
        act_chunk = batch['act_chunk']
        expert_action = batch['expert_action']

        # 前向传播
        optimizer.zero_grad()

        delta_pred, feature_metrics = model(
            tactile_history=tactile_history,
            current_force=current_force,
            state=state,
            act_chunk=act_chunk,
            return_feature_metrics=True,
        )

        # 计算损失
        loss = residual_loss(delta_pred, expert_action, act_chunk)

        # 计算最终预测动作的每个轴MSE
        final_action = act_chunk + delta_pred  # ACT预测 + 残差 = 最终动作
        per_axis_mse_final = compute_per_axis_mse(final_action, expert_action)

        # 计算ACT原始输出的每个轴MSE（未修正）
        per_axis_mse_act = compute_per_axis_mse(act_chunk, expert_action)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # 累积指标
        epoch_per_axis_mse_final.append(per_axis_mse_final.cpu())
        epoch_per_axis_mse_act.append(per_axis_mse_act.cpu())
        epoch_modality_contributions['tactile'].append(feature_metrics['tactile_contribution_ratio'])
        epoch_modality_contributions['current_force'].append(feature_metrics['current_force_contribution_ratio'])
        epoch_modality_contributions['state'].append(feature_metrics['state_contribution_ratio'])
        epoch_modality_contributions['action'].append(feature_metrics['action_contribution_ratio'])

        # 每个batch记录到wandb
        if wandb.run is not None:
            wandb.log({
                'train/loss': loss.item(),
                'train/mse_final_x': per_axis_mse_final[0].item(),
                'train/mse_final_y': per_axis_mse_final[1].item(),
                'train/mse_final_z': per_axis_mse_final[2].item(),
                'train/mse_final_rx': per_axis_mse_final[3].item(),
                'train/mse_final_ry': per_axis_mse_final[4].item(),
                'train/mse_final_rz': per_axis_mse_final[5].item(),
                'train/mse_act_x': per_axis_mse_act[0].item(),
                'train/mse_act_y': per_axis_mse_act[1].item(),
                'train/mse_act_z': per_axis_mse_act[2].item(),
                'train/mse_act_rx': per_axis_mse_act[3].item(),
                'train/mse_act_ry': per_axis_mse_act[4].item(),
                'train/mse_act_rz': per_axis_mse_act[5].item(),
                'train/modality_tactile': feature_metrics['tactile_contribution_ratio'],
                'train/modality_current_force': feature_metrics['current_force_contribution_ratio'],
                'train/modality_state': feature_metrics['state_contribution_ratio'],
                'train/modality_action': feature_metrics['action_contribution_ratio'],
                'global_step': global_step,
            }, step=global_step)

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}]")
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Tactile: {feature_metrics['tactile_contribution_ratio']:.2%}, "
                  f"Current: {feature_metrics['current_force_contribution_ratio']:.2%}, "
                  f"State: {feature_metrics['state_contribution_ratio']:.2%}, "
                  f"Action: {feature_metrics['action_contribution_ratio']:.2%}")

    # Epoch平均指标
    avg_per_axis_mse_final = torch.stack(epoch_per_axis_mse_final).mean(dim=0)
    avg_per_axis_mse_act = torch.stack(epoch_per_axis_mse_act).mean(dim=0)
    avg_modality_contributions = {
        k: sum(v) / len(v) for k, v in epoch_modality_contributions.items()
    }

    return total_loss / num_batches, avg_per_axis_mse_final, avg_per_axis_mse_act, avg_modality_contributions, global_step


def validate(model, dataloader, device):
    """验证"""

    model.eval()
    total_loss = 0.0
    num_batches = 0

    # 累积指标
    epoch_per_axis_mse_final = []
    epoch_per_axis_mse_act = []
    epoch_modality_contributions = {
        'tactile': [],
        'current_force': [],
        'state': [],
        'action': []
    }

    with torch.no_grad():
        for batch in dataloader:
            tactile_history = batch['tactile_history']
            current_force = batch['current_force']
            state = batch['observation.state']
            act_chunk = batch['act_chunk']
            expert_action = batch['expert_action']

            delta_pred, feature_metrics = model(
                tactile_history=tactile_history,
                current_force=current_force,
                state=state,
                act_chunk=act_chunk,
                return_feature_metrics=True,
            )

            loss = residual_loss(delta_pred, expert_action, act_chunk)

            # 计算最终预测动作的每个轴MSE
            final_action = act_chunk + delta_pred  # ACT预测 + 残差 = 最终动作
            per_axis_mse_final = compute_per_axis_mse(final_action, expert_action)

            # 计算ACT原始输出的每个轴MSE（未修正）
            per_axis_mse_act = compute_per_axis_mse(act_chunk, expert_action)

            total_loss += loss.item()
            num_batches += 1

            # 累积指标
            epoch_per_axis_mse_final.append(per_axis_mse_final.cpu())
            epoch_per_axis_mse_act.append(per_axis_mse_act.cpu())
            epoch_modality_contributions['tactile'].append(feature_metrics['tactile_contribution_ratio'])
            epoch_modality_contributions['current_force'].append(feature_metrics['current_force_contribution_ratio'])
            epoch_modality_contributions['state'].append(feature_metrics['state_contribution_ratio'])
            epoch_modality_contributions['action'].append(feature_metrics['action_contribution_ratio'])

    # Epoch平均指标
    avg_per_axis_mse_final = torch.stack(epoch_per_axis_mse_final).mean(dim=0)
    avg_per_axis_mse_act = torch.stack(epoch_per_axis_mse_act).mean(dim=0)
    avg_modality_contributions = {
        k: sum(v) / len(v) for k, v in epoch_modality_contributions.items()
    }

    return total_loss / num_batches, avg_per_axis_mse_final, avg_per_axis_mse_act, avg_modality_contributions


def main():
    print("=" * 70)
    print("VQ-VAE集成架构完整训练脚本")
    print("=" * 70)

    # 配置路径
    dataloader_config_path = "dataloader/tactile_dataloader.yaml"
    model_config_path = "config/model_config.yaml"

    # 训练参数
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化wandb
    wandb.init(
        project="tactile-vqvae-residual",
        config={
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "architecture": "VQ-VAE + 4-modality fusion",
            "modalities": ["tactile_history", "current_force", "state", "action"],
            "dataloader_config": dataloader_config_path,
            "model_config": model_config_path,
        },
        name=f"vqvae_4modality_lr{learning_rate}",
    )

    print(f"\n配置:")
    print(f"  Dataloader config: {dataloader_config_path}")
    print(f"  Model config: {model_config_path}")
    print(f"  设备: {device}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")

    # 加载dataloader配置
    print(f"\n加载dataloader...")
    dataloader_config = load_yaml(dataloader_config_path)
    set_seed(42)

    # 构建数据集
    print(f"  构建数据集...")
    dataset = build_base_dataset(dataloader_config)
    print(f"  数据集: {dataset.repo_id}")
    print(f"  Episodes: {dataset.num_episodes}")

    # 构建普通dataloader
    print(f"  构建base dataloader...")
    normal_loaders, datasets = build_normal_dataloaders(dataloader_config, dataset)

    # 加载冻结的ACT策略
    print(f"  加载ACT策略...")
    policy, preprocessor, postprocessor, policy_device = load_lerobot_policy(
        dataloader_config, dataset
    )

    # 构建带ACT预测的dataloader
    print(f"  构建augmented dataloader...")
    augmented_loaders = build_augmented_loaders(
        dataloader_config,
        normal_loaders,
        policy,
        preprocessor,
        postprocessor,
        policy_device,
    )

    # 加载VQ-VAE集成模型
    print(f"\n加载VQ-VAE集成模型...")
    model, model_config = load_model_from_config(model_config_path)
    model = model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n模型统计:")
    print(f"  总参数: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  冻结参数 (VQ-VAE): {frozen_params / 1e6:.2f}M")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    # 记录模型统计到wandb
    wandb.config.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    })

    # 训练循环
    print(f"\n开始训练...")
    print("=" * 70)

    best_val_loss = float('inf')
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        # 训练
        train_loss, train_per_axis_mse_final, train_per_axis_mse_act, train_modality_contrib, global_step = train_epoch(
            model, augmented_loaders['train'], optimizer, device, epoch + 1, global_step
        )
        print(f"  训练损失: {train_loss:.4f}")

        # 验证
        val_loss, val_per_axis_mse_final, val_per_axis_mse_act, val_modality_contrib = validate(
            model, augmented_loaders['val'], device
        )
        print(f"  验证损失: {val_loss:.4f}")

        # 记录epoch级别的指标到wandb
        wandb.log({
            'epoch': epoch + 1,
            'train/epoch_loss': train_loss,
            'train/epoch_mse_final_x': train_per_axis_mse_final[0].item(),
            'train/epoch_mse_final_y': train_per_axis_mse_final[1].item(),
            'train/epoch_mse_final_z': train_per_axis_mse_final[2].item(),
            'train/epoch_mse_final_rx': train_per_axis_mse_final[3].item(),
            'train/epoch_mse_final_ry': train_per_axis_mse_final[4].item(),
            'train/epoch_mse_final_rz': train_per_axis_mse_final[5].item(),
            'train/epoch_mse_act_x': train_per_axis_mse_act[0].item(),
            'train/epoch_mse_act_y': train_per_axis_mse_act[1].item(),
            'train/epoch_mse_act_z': train_per_axis_mse_act[2].item(),
            'train/epoch_mse_act_rx': train_per_axis_mse_act[3].item(),
            'train/epoch_mse_act_ry': train_per_axis_mse_act[4].item(),
            'train/epoch_mse_act_rz': train_per_axis_mse_act[5].item(),
            'train/epoch_modality_tactile': train_modality_contrib['tactile'],
            'train/epoch_modality_current_force': train_modality_contrib['current_force'],
            'train/epoch_modality_state': train_modality_contrib['state'],
            'train/epoch_modality_action': train_modality_contrib['action'],
            'val/epoch_loss': val_loss,
            'val/epoch_mse_final_x': val_per_axis_mse_final[0].item(),
            'val/epoch_mse_final_y': val_per_axis_mse_final[1].item(),
            'val/epoch_mse_final_z': val_per_axis_mse_final[2].item(),
            'val/epoch_mse_final_rx': val_per_axis_mse_final[3].item(),
            'val/epoch_mse_final_ry': val_per_axis_mse_final[4].item(),
            'val/epoch_mse_final_rz': val_per_axis_mse_final[5].item(),
            'val/epoch_mse_act_x': val_per_axis_mse_act[0].item(),
            'val/epoch_mse_act_y': val_per_axis_mse_act[1].item(),
            'val/epoch_mse_act_z': val_per_axis_mse_act[2].item(),
            'val/epoch_mse_act_rx': val_per_axis_mse_act[3].item(),
            'val/epoch_mse_act_ry': val_per_axis_mse_act[4].item(),
            'val/epoch_mse_act_rz': val_per_axis_mse_act[5].item(),
            'val/epoch_modality_tactile': val_modality_contrib['tactile'],
            'val/epoch_modality_current_force': val_modality_contrib['current_force'],
            'val/epoch_modality_state': val_modality_contrib['state'],
            'val/epoch_modality_action': val_modality_contrib['action'],
        }, step=global_step)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_vqvae_model.pth"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config,
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型: {checkpoint_path}")

            # 记录最佳checkpoint到wandb
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch + 1

    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("=" * 70)

    # 结束wandb
    wandb.finish()


if __name__ == "__main__":
    main()
