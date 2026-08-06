"""
VQ-VAE推理脚本：加载checkpoint并对触觉力历史数据进行编码/解码
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import argparse
from vqvae_model import TemporalVQVAE


class VQVAEInference:
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        """
        初始化推理器

        Args:
            checkpoint_path: checkpoint文件路径
            config_path: 配置文件路径
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 构建模型
        self.model = TemporalVQVAE(self.config['model'])

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"加载checkpoint (无epoch信息)")

        self.model.to(self.device)
        self.model.eval()

        print(f"模型加载成功，使用设备: {self.device}")
        print(f"Codebook大小: {self.config['model']['quantizer']['num_embeddings']}")
        print(f"Embedding维度: {self.config['model']['quantizer']['embedding_dim']}")

    def encode(self, force_history):
        """
        编码触觉力历史为离散token

        Args:
            force_history: [B, T, D] 或 [T, D] 的numpy数组或tensor

        Returns:
            indices: [B] 离散token索引
            z_q: [B, embedding_dim] 量化后的embedding
        """
        # 转换为tensor
        if isinstance(force_history, np.ndarray):
            force_history = torch.from_numpy(force_history).float()

        # 添加batch维度
        if force_history.ndim == 2:
            force_history = force_history.unsqueeze(0)

        force_history = force_history.to(self.device)

        with torch.no_grad():
            indices, z_q = self.model.encode(force_history)

        return indices.cpu().numpy(), z_q.cpu().numpy()

    def decode(self, z_q):
        """
        从量化embedding解码回力历史

        Args:
            z_q: [B, embedding_dim] 量化embedding

        Returns:
            x_recon: [B, T, D] 重建的力历史
        """
        if isinstance(z_q, np.ndarray):
            z_q = torch.from_numpy(z_q).float()

        z_q = z_q.to(self.device)

        with torch.no_grad():
            x_recon = self.model.decode(z_q)

        return x_recon.cpu().numpy()

    def decode_from_indices(self, indices):
        """
        从token索引解码回力历史

        Args:
            indices: [B] token索引

        Returns:
            x_recon: [B, T, D] 重建的力历史
        """
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long()
        elif isinstance(indices, (int, list)):
            indices = torch.tensor(indices).long()

        if indices.ndim == 0:
            indices = indices.unsqueeze(0)

        indices = indices.to(self.device)

        with torch.no_grad():
            x_recon = self.model.decode_from_indices(indices)

        return x_recon.cpu().numpy()

    def reconstruct(self, force_history):
        """
        完整的重建过程：编码 -> 解码

        Args:
            force_history: [B, T, D] 或 [T, D]

        Returns:
            x_recon: 重建结果
            indices: token索引
            recon_error: 重建误差
        """
        if isinstance(force_history, np.ndarray):
            force_history_tensor = torch.from_numpy(force_history).float()
        else:
            force_history_tensor = force_history

        if force_history_tensor.ndim == 2:
            force_history_tensor = force_history_tensor.unsqueeze(0)

        force_history_tensor = force_history_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(force_history_tensor)

        x_recon = output['x_recon'].cpu().numpy()
        indices = output['indices'].cpu().numpy()
        recon_error = output['recon_loss'].item()

        return x_recon, indices, recon_error

    def get_codebook(self):
        """
        获取完整的codebook

        Returns:
            codebook: [num_embeddings, embedding_dim]
        """
        return self.model.quantizer.embedding.cpu().numpy()

    def batch_encode(self, force_histories, batch_size=32):
        """
        批量编码

        Args:
            force_histories: [N, T, D] 多个力历史
            batch_size: 批大小

        Returns:
            all_indices: [N] 所有token索引
            all_embeddings: [N, embedding_dim] 所有embeddings
        """
        if isinstance(force_histories, np.ndarray):
            force_histories = torch.from_numpy(force_histories).float()

        n_samples = force_histories.shape[0]
        all_indices = []
        all_embeddings = []

        for i in range(0, n_samples, batch_size):
            batch = force_histories[i:i+batch_size].to(self.device)

            with torch.no_grad():
                indices, z_q = self.model.encode(batch)

            all_indices.append(indices.cpu())
            all_embeddings.append(z_q.cpu())

        all_indices = torch.cat(all_indices, dim=0).numpy()
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

        return all_indices, all_embeddings


def demo():
    """演示推理流程"""
    parser = argparse.ArgumentParser(description='VQ-VAE推理演示')
    parser.add_argument('--checkpoint', type=str,
                       default='vqvae_checkpoints/checkpoint_best.pth',
                       help='Checkpoint路径')
    parser.add_argument('--config', type=str,
                       default='vqvae_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备')

    args = parser.parse_args()

    # 构建完整路径
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / args.checkpoint
    config_path = script_dir / args.config

    print("="*60)
    print("VQ-VAE触觉编码器推理演示")
    print("="*60)

    # 初始化推理器
    inferencer = VQVAEInference(checkpoint_path, config_path, args.device)

    # 生成随机测试数据
    B, T, D = 4, 15, 12
    force_history = np.random.randn(B, T, D).astype(np.float32)

    print(f"\n输入数据形状: {force_history.shape}")

    # 测试编码
    print("\n1. 测试编码...")
    indices, z_q = inferencer.encode(force_history)
    print(f"   Token索引: {indices}")
    print(f"   Embedding形状: {z_q.shape}")

    # 测试从索引解码
    print("\n2. 测试从索引解码...")
    x_recon_from_indices = inferencer.decode_from_indices(indices)
    print(f"   重建数据形状: {x_recon_from_indices.shape}")

    # 测试从embedding解码
    print("\n3. 测试从embedding解码...")
    x_recon_from_embedding = inferencer.decode(z_q)
    print(f"   重建数据形状: {x_recon_from_embedding.shape}")

    # 测试完整重建
    print("\n4. 测试完整重建...")
    x_recon, indices_recon, recon_error = inferencer.reconstruct(force_history)
    print(f"   重建误差 (MSE): {recon_error:.6f}")
    print(f"   Token索引: {indices_recon}")

    # 获取codebook
    print("\n5. 获取codebook...")
    codebook = inferencer.get_codebook()
    print(f"   Codebook形状: {codebook.shape}")

    # 测试批量编码
    print("\n6. 测试批量编码...")
    large_batch = np.random.randn(100, T, D).astype(np.float32)
    all_indices, all_embeddings = inferencer.batch_encode(large_batch, batch_size=32)
    print(f"   处理 {len(all_indices)} 个样本")
    print(f"   索引分布: {np.bincount(all_indices, minlength=8)}")

    print("\n" + "="*60)
    print("推理演示完成！")
    print("="*60)


if __name__ == "__main__":
    demo()
