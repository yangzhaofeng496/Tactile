"""
Inference and visualization for T-Rex VQ-VAE.

Features:
- Encode tactile history to discrete tokens
- Decode tokens back to force history
- Visualize reconstruction quality
- Export encoder for downstream use
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
import argparse
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from TactileSelfencoder.trex_vqvae_model import TRexTactileVQVAE
from TactileSelfencoder.compute_tactile_stats import (
    load_tactile_stats,
    normalize_tactile_data,
    denormalize_tactile_data
)
from dataloader.dataloader import build_dataloader_from_config


class TRexVQVAEInference:
    """Inference interface for T-Rex VQ-VAE."""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run on
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.config = checkpoint['config']
        self.model_config = self.config['model']

        # Build model
        self.model = TRexTactileVQVAE(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully")

        # Load normalization stats
        if 'norm_mean' in checkpoint and 'norm_std' in checkpoint:
            self.norm_mean = checkpoint['norm_mean']
            self.norm_std = checkpoint['norm_std']
            self.norm_mean_t = torch.tensor(
                self.norm_mean, dtype=torch.float32, device=self.device
            )
            self.norm_std_t = torch.tensor(
                self.norm_std, dtype=torch.float32, device=self.device
            )
            print("Normalization stats loaded from checkpoint")
        else:
            self.norm_mean = None
            self.norm_std = None
            self.norm_mean_t = None
            self.norm_std_t = None
            print("Warning: No normalization stats in checkpoint")

    @torch.no_grad()
    def encode(self, x):
        """
        Encode tactile history to discrete tokens.

        Args:
            x: [B, T, num_fingers, force_dim] raw force history

        Returns:
            indices: [B, num_fingers] discrete tokens
            z_q: [B, num_fingers, embedding_dim] quantized embeddings
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        # Normalize
        if self.norm_mean_t is not None:
            x_norm = normalize_tactile_data(x, self.norm_mean_t, self.norm_std_t)
        else:
            x_norm = x

        # Encode
        indices, z_q = self.model.encode(x_norm)

        return indices.cpu().numpy(), z_q.cpu().numpy()

    @torch.no_grad()
    def decode(self, indices):
        """
        Decode from token indices.

        Args:
            indices: [B, num_fingers] discrete tokens

        Returns:
            x_recon: [B, T, num_fingers, force_dim] reconstructed force history (denormalized)
        """
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)

        # Decode
        x_recon_norm = self.model.decode_from_indices(indices)

        # Denormalize
        if self.norm_mean_t is not None:
            x_recon = denormalize_tactile_data(
                x_recon_norm, self.norm_mean_t, self.norm_std_t
            )
        else:
            x_recon = x_recon_norm

        return x_recon.cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x):
        """
        Full reconstruction: encode then decode.

        Args:
            x: [B, T, num_fingers, force_dim] raw force history

        Returns:
            x_recon: [B, T, num_fingers, force_dim] reconstructed
            indices: [B, num_fingers] tokens used
        """
        indices, z_q = self.encode(x)
        x_recon = self.decode(indices)

        return x_recon, indices

    def export_encoder(self, output_path):
        """
        Export encoder + quantizer for downstream use.

        Saves:
        - encoder weights
        - quantizer codebook
        - normalization stats
        - config
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_dict = {
            'encoder_state_dict': self.model.encoder.state_dict(),
            'quantizer_state_dict': self.model.quantizer.state_dict(),
            'config': self.config,
        }

        if self.norm_mean is not None:
            export_dict['norm_mean'] = self.norm_mean
            export_dict['norm_std'] = self.norm_std

        torch.save(export_dict, output_path)
        print(f"Encoder exported to: {output_path}")


def visualize_reconstruction(x_orig, x_recon, indices, save_path=None):
    """
    Visualize original vs reconstructed force history.

    Args:
        x_orig: [B, T, num_fingers, force_dim] original
        x_recon: [B, T, num_fingers, force_dim] reconstructed
        indices: [B, num_fingers] tokens
        save_path: Optional path to save figure
    """
    B, T, F, D = x_orig.shape

    # Select first sample
    x_orig_sample = x_orig[0]  # [T, F, D]
    x_recon_sample = x_recon[0]  # [T, F, D]
    indices_sample = indices[0]  # [F]

    channel_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    fig, axes = plt.subplots(F, D, figsize=(20, 8))

    if F == 1:
        axes = axes.reshape(1, -1)

    for f in range(F):
        for d in range(D):
            ax = axes[f, d]

            ax.plot(x_orig_sample[:, f, d], label='Original', linewidth=2, alpha=0.7)
            ax.plot(x_recon_sample[:, f, d], label='Reconstructed', linewidth=2, linestyle='--', alpha=0.7)

            ax.set_title(f'Finger {f} - {channel_names[d]} (Token: {indices_sample[f]})')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Force/Torque')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def compute_reconstruction_metrics(x_orig, x_recon):
    """
    Compute reconstruction metrics.

    Args:
        x_orig: [B, T, num_fingers, force_dim]
        x_recon: [B, T, num_fingers, force_dim]

    Returns:
        dict of metrics
    """
    # MSE
    mse = np.mean((x_orig - x_recon) ** 2)

    # Per-channel MSE
    per_channel_mse = np.mean((x_orig - x_recon) ** 2, axis=(0, 1))  # [F, D]

    # RMSE
    rmse = np.sqrt(mse)

    # Normalized RMSE (NRMSE)
    signal_range = x_orig.max() - x_orig.min()
    nrmse = rmse / (signal_range + 1e-6)

    # R² score per channel
    ss_res = np.sum((x_orig - x_recon) ** 2, axis=(0, 1))  # [F, D]
    ss_tot = np.sum((x_orig - x_orig.mean(axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))  # [F, D]
    r2_per_channel = 1 - ss_res / (ss_tot + 1e-10)

    # Overall R²
    r2_overall = 1 - np.sum(ss_res) / (np.sum(ss_tot) + 1e-10)

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'r2_overall': float(r2_overall),
        'per_channel_mse': per_channel_mse.tolist(),
        'per_channel_r2': r2_per_channel.tolist(),
    }

    return metrics


def evaluate_on_dataset(inference, dataloader, num_batches=None, output_dir=None):
    """
    Evaluate reconstruction quality on a dataset.

    Args:
        inference: TRexVQVAEInference instance
        dataloader: DataLoader
        num_batches: Number of batches to evaluate (None = all)
        output_dir: Optional directory to save results
    """
    all_x_orig = []
    all_x_recon = []
    all_indices = []

    num_batches = num_batches or len(dataloader)

    print(f"Evaluating on {num_batches} batches...")

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if batch_idx >= num_batches:
            break

        # Extract data
        if isinstance(batch, dict):
            if 'force' in batch:
                x = batch['force'].numpy()
            elif 'tactile' in batch:
                x = batch['tactile'].numpy()
            else:
                raise KeyError(f"Expected 'force' or 'tactile', got: {batch.keys()}")
        else:
            x = batch.numpy()

        # Reshape if needed
        if x.ndim == 3 and x.shape[-1] == inference.model_config['num_fingers'] * inference.model_config['force_dim']:
            B, T, _ = x.shape
            x = x.reshape(B, T, inference.model_config['num_fingers'], inference.model_config['force_dim'])

        # Reconstruct
        x_recon, indices = inference.reconstruct(x)

        all_x_orig.append(x)
        all_x_recon.append(x_recon)
        all_indices.append(indices)

    # Concatenate
    all_x_orig = np.concatenate(all_x_orig, axis=0)
    all_x_recon = np.concatenate(all_x_recon, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    # Compute metrics
    metrics = compute_reconstruction_metrics(all_x_orig, all_x_recon)

    print("\nReconstruction Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  NRMSE: {metrics['nrmse']:.6f}")
    print(f"  R² (overall): {metrics['r2_overall']:.4f}")

    # Codebook usage
    all_indices_flat = all_indices.reshape(-1)
    unique_codes = np.unique(all_indices_flat)
    usage_counts = np.bincount(all_indices_flat, minlength=inference.model_config['quantizer']['num_embeddings'])

    print(f"\nCodebook Usage:")
    print(f"  Active codes: {len(unique_codes)}/{inference.model_config['quantizer']['num_embeddings']}")
    print(f"  Usage distribution: min={usage_counts.min()}, max={usage_counts.max()}, mean={usage_counts.mean():.1f}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(output_dir / 'reconstruction_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save codebook usage
        np.save(output_dir / 'codebook_usage.npy', usage_counts)

        # Visualize a few samples
        for i in range(min(5, all_x_orig.shape[0])):
            visualize_reconstruction(
                all_x_orig[i:i+1],
                all_x_recon[i:i+1],
                all_indices[i:i+1],
                save_path=output_dir / f'reconstruction_sample_{i}.png'
            )

        print(f"\nResults saved to: {output_dir}")

    return metrics, all_x_orig, all_x_recon, all_indices


def main():
    parser = argparse.ArgumentParser(description="T-Rex VQ-VAE Inference")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['evaluate', 'export_encoder'],
        default='evaluate',
        help='Inference mode'
    )
    parser.add_argument(
        '--dataloader_config',
        type=str,
        default='dataloader/vqvae_tactile.yaml',
        help='Dataloader config for evaluation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='TactileSelfencoder/trex_vqvae_eval',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=50,
        help='Number of batches to evaluate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--export_path',
        type=str,
        default='TactileSelfencoder/trex_encoder_export.pt',
        help='Path to export encoder'
    )

    args = parser.parse_args()

    # Create inference
    inference = TRexVQVAEInference(args.checkpoint, device=args.device)

    if args.mode == 'evaluate':
        # Load dataloader
        _, val_loader, _ = build_dataloader_from_config(args.dataloader_config)

        # Evaluate
        evaluate_on_dataset(
            inference,
            val_loader,
            num_batches=args.num_batches,
            output_dir=args.output_dir
        )

    elif args.mode == 'export_encoder':
        # Export encoder
        inference.export_encoder(args.export_path)


if __name__ == "__main__":
    main()
