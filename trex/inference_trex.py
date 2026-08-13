"""
Inference and visualization for T-Rex VQ-VAE (2-finger adapted).

Usage:
    python TactileSelfencoder/inference_trex.py \\
        --checkpoint outputs/trex_vqvae/latest.pt \\
        --data_config dataloader/vqvae_tactile.yaml \\
        --output outputs/trex_vqvae/visualizations \\
        --n_samples 10
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloader.dataloader import (
    build_base_dataset,
    build_normal_dataloaders,
    set_seed,
)
from TactileSelfencoder.trex_official import TactileVQVAE, TactileVQVAEConfig


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config
    config_dict = checkpoint['config']
    model_cfg = config_dict['model']

    cfg = TactileVQVAEConfig(
        window=model_cfg['window'],
        in_channels=model_cfg['n_fingers'] * model_cfg['per_finger_dim'],
        hidden_channels=model_cfg['hidden_channels'],
        bottleneck_channels=model_cfg['bottleneck_channels'],
        embed_dim=model_cfg['embed_dim'],
        n_strided_blocks=model_cfg['n_strided_blocks'],
        codebook_size=model_cfg['codebook_size'],
        commitment_weight=model_cfg['commitment_weight'],
        decay=model_cfg['decay'],
        revive_freq=model_cfg['revive_freq'],
        revive_threshold=model_cfg['revive_threshold'],
        use_magnitude_weight=model_cfg['use_magnitude_weight'],
        weight_alpha=model_cfg['weight_alpha'],
        weight_tau=model_cfg['weight_tau'],
        granularity=model_cfg['granularity'],
        n_fingers=model_cfg['n_fingers'],
        per_finger_dim=model_cfg['per_finger_dim'],
        init_mode=model_cfg.get('init_mode', 'uniform'),
    )

    model = TactileVQVAE(cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    stats = checkpoint['stats']

    return model, stats, checkpoint


def normalize_force(force, stats):
    """Normalize force using stored statistics."""
    if isinstance(force, np.ndarray):
        force = torch.from_numpy(force)

    # Handle shape
    if force.dim() == 3:
        B, T, total_dim = force.shape
        if total_dim == 12:
            force = force.reshape(B, T, 2, 6)

    mean = torch.tensor(stats['mean'], dtype=force.dtype, device=force.device)
    std = torch.tensor(stats['std'], dtype=force.dtype, device=force.device)

    normalized = (force - mean) / (std + 1e-6)
    return normalized


def denormalize_force(force_norm, stats):
    """Denormalize force back to original scale."""
    mean = torch.tensor(stats['mean'], dtype=force_norm.dtype, device=force_norm.device)
    std = torch.tensor(stats['std'], dtype=force_norm.dtype, device=force_norm.device)

    return force_norm * std + mean


@torch.no_grad()
def encode_dataset(model, dataloader, stats, device, max_samples=None):
    """Encode entire dataset and collect indices."""
    model.eval()

    all_indices = []
    all_magnitudes = []
    all_recon_errors = []
    n_samples = 0

    for batch in tqdm(dataloader, desc="Encoding"):
        force_raw = batch['tactile_history']  # [B, T, 12]

        # Reshape from [B, T, 12] to [B, T, 2, 6]
        B, T, total_dim = force_raw.shape
        if total_dim != 12:
            raise ValueError(f"Expected tactile_history shape [B, T, 12], got {force_raw.shape}")

        force_raw = force_raw.reshape(B, T, 2, 6).to(device)
        force_norm = normalize_force(force_raw, stats)

        # Encode
        indices = model.encode(force_norm)  # [B, 2] for per-finger

        # Decode and compute error
        recon_norm = model.decode_indices(indices)
        recon_raw = denormalize_force(recon_norm, stats)

        error = torch.mean((force_raw - recon_raw) ** 2, dim=[1, 2, 3])  # [B]
        magnitude = torch.norm(force_raw.reshape(force_raw.shape[0], -1), dim=1)

        all_indices.append(indices.cpu().numpy())
        all_magnitudes.append(magnitude.cpu().numpy())
        all_recon_errors.append(error.cpu().numpy())

        n_samples += force_raw.shape[0]
        if max_samples and n_samples >= max_samples:
            break

    return {
        'indices': np.concatenate(all_indices, axis=0),  # [N, 2]
        'magnitudes': np.concatenate(all_magnitudes),    # [N]
        'recon_errors': np.concatenate(all_recon_errors),  # [N]
    }


def visualize_reconstruction(force_orig, force_recon, indices, save_path):
    """Visualize original vs reconstructed force trajectories."""
    # force_orig, force_recon: [T, 2, 6]
    T, n_fingers, n_dims = force_orig.shape

    channel_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

    fig, axes = plt.subplots(n_fingers, n_dims, figsize=(18, 8))
    fig.suptitle(f'Force Reconstruction (Indices: {indices})', fontsize=14)

    for finger in range(n_fingers):
        for dim in range(n_dims):
            ax = axes[finger, dim] if n_fingers > 1 else axes[dim]

            orig = force_orig[:, finger, dim]
            recon = force_recon[:, finger, dim]

            ax.plot(orig, label='Original', linewidth=2, alpha=0.7)
            ax.plot(recon, label='Reconstructed', linewidth=2, linestyle='--', alpha=0.7)

            mse = np.mean((orig - recon) ** 2)
            ax.set_title(f'Finger {finger} - {channel_names[dim]} (MSE: {mse:.4f})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_codebook_usage(indices, codebook_size, save_path):
    """Visualize codebook usage histogram."""
    # indices: [N, 2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for finger in range(2):
        ax = axes[finger]
        finger_indices = indices[:, finger]

        counts = np.bincount(finger_indices, minlength=codebook_size)
        usage = (counts > 0).sum() / codebook_size * 100

        ax.bar(range(codebook_size), counts)
        ax.set_title(f'Finger {finger} Codebook Usage\n'
                    f'{usage:.1f}% codes active ({(counts > 0).sum()}/{codebook_size})')
        ax.set_xlabel('Code Index')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_magnitude_vs_error(magnitudes, recon_errors, save_path):
    """Scatter plot of magnitude vs reconstruction error."""
    plt.figure(figsize=(10, 6))

    plt.scatter(magnitudes, recon_errors, alpha=0.3, s=10)
    plt.xlabel('Force Magnitude (L2 norm)')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error vs Force Magnitude')
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(magnitudes, recon_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(magnitudes.min(), magnitudes.max(), 100)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="T-Rex VQ-VAE Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to data config YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--max_encode",
        type=int,
        default=10000,
        help="Maximum samples to encode for statistics"
    )
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, stats, checkpoint = load_checkpoint(args.checkpoint, device)

    codebook_size = checkpoint['config']['model']['codebook_size']
    print(f"Codebook size: {codebook_size}")

    # Load data
    print(f"Loading data: {args.data_config}")
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    # Set seed
    set_seed(data_config['split']['seed'])

    # Build dataset and dataloaders
    print("Building dataloaders...")
    base_dataset = build_base_dataset(data_config)
    dataloaders, datasets = build_normal_dataloaders(data_config, base_dataset)

    if 'val' not in dataloaders:
        raise ValueError("No validation split found")

    val_loader = dataloaders['val']
    print(f"Val dataset: {len(datasets['val'])} samples")

    # Encode dataset
    print(f"\nEncoding up to {args.max_encode} samples...")
    encoding_results = encode_dataset(
        model, val_loader, stats, device,
        max_samples=args.max_encode
    )

    indices = encoding_results['indices']
    magnitudes = encoding_results['magnitudes']
    recon_errors = encoding_results['recon_errors']

    print(f"Encoded {len(indices)} samples")
    print(f"Mean reconstruction error: {recon_errors.mean():.6f}")
    print(f"Std reconstruction error: {recon_errors.std():.6f}")

    # Visualize codebook usage
    print("\nVisualizing codebook usage...")
    visualize_codebook_usage(
        indices, codebook_size,
        output_dir / "codebook_usage.png"
    )

    # Visualize magnitude vs error
    print("Visualizing magnitude vs error...")
    visualize_magnitude_vs_error(
        magnitudes, recon_errors,
        output_dir / "magnitude_vs_error.png"
    )

    # Visualize sample reconstructions
    print(f"\nVisualizing {args.n_samples} sample reconstructions...")

    sample_count = 0
    for batch_idx, batch in enumerate(val_loader):
        if sample_count >= args.n_samples:
            break

        force_raw = batch['tactile_history']  # [B, T, 12]

        # Reshape from [B, T, 12] to [B, T, 2, 6]
        B, T, total_dim = force_raw.shape
        if total_dim != 12:
            raise ValueError(f"Expected tactile_history shape [B, T, 12], got {force_raw.shape}")

        force_raw = force_raw.reshape(B, T, 2, 6).to(device)

        # Process each sample in batch
        for i in range(force_raw.shape[0]):
            if sample_count >= args.n_samples:
                break

            sample = force_raw[i:i+1]  # [1, T, 2, 6]
            force_norm = normalize_force(sample, stats)

            # Encode and decode
            idx = model.encode(force_norm)  # [1, 2]
            recon_norm = model.decode_indices(idx)
            recon_raw = denormalize_force(recon_norm, stats)

            # Convert to numpy for visualization
            orig_np = sample[0].cpu().numpy()  # [T, 2, 6]
            recon_np = recon_raw[0].cpu().numpy()
            idx_np = idx[0].cpu().numpy()  # [2]

            # Visualize
            save_path = output_dir / f"reconstruction_sample_{sample_count:03d}.png"
            visualize_reconstruction(orig_np, recon_np, idx_np, save_path)

            sample_count += 1

    # Save statistics
    stats_output = {
        'checkpoint': str(args.checkpoint),
        'n_samples_encoded': int(len(indices)),
        'codebook_size': int(codebook_size),
        'reconstruction_error': {
            'mean': float(recon_errors.mean()),
            'std': float(recon_errors.std()),
            'min': float(recon_errors.min()),
            'max': float(recon_errors.max()),
        },
        'codebook_usage': {
            'finger_0': {
                'active_codes': int((np.bincount(indices[:, 0], minlength=codebook_size) > 0).sum()),
                'usage_ratio': float((np.bincount(indices[:, 0], minlength=codebook_size) > 0).sum() / codebook_size),
            },
            'finger_1': {
                'active_codes': int((np.bincount(indices[:, 1], minlength=codebook_size) > 0).sum()),
                'usage_ratio': float((np.bincount(indices[:, 1], minlength=codebook_size) > 0).sum() / codebook_size),
            },
        },
        'magnitude': {
            'mean': float(magnitudes.mean()),
            'std': float(magnitudes.std()),
            'min': float(magnitudes.min()),
            'max': float(magnitudes.max()),
        },
    }

    stats_path = output_dir / "inference_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)

    print(f"\n✓ Statistics saved: {stats_path}")
    print(f"✓ All visualizations saved to: {output_dir}")

    print("\nCodebook Usage Summary:")
    for finger in range(2):
        usage = stats_output['codebook_usage'][f'finger_{finger}']
        print(f"  Finger {finger}: {usage['active_codes']}/{codebook_size} "
              f"({usage['usage_ratio']*100:.1f}%)")


if __name__ == "__main__":
    main()
