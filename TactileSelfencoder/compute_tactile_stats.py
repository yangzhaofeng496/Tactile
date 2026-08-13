"""
Compute per-finger, per-channel normalization statistics from training set only.

Output: tactile_stats.json with:
- channel_mean: [num_fingers, force_dim]
- channel_std: [num_fingers, force_dim]
- metadata
"""

import json
import torch
import numpy as np
from pathlib import Path
import yaml
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloader.dataloader import build_dataloader_from_config


def compute_tactile_stats(dataloader_config_path, output_path, num_fingers=2, force_dim=6):
    """
    Compute per-finger, per-channel mean and std from training set.

    Args:
        dataloader_config_path: Path to dataloader YAML config
        output_path: Path to save stats JSON
        num_fingers: Number of fingers (default 2)
        force_dim: Force dimension per finger (default 6)
    """

    # Load dataloader
    print(f"Loading dataloader from: {dataloader_config_path}")
    train_loader, val_loader, config = build_dataloader_from_config(dataloader_config_path)

    print(f"Computing statistics from {len(train_loader)} training batches...")

    # Accumulators for Welford's online algorithm
    # We compute stats per (finger, channel)
    count = 0
    mean = np.zeros((num_fingers, force_dim))
    M2 = np.zeros((num_fingers, force_dim))

    for batch_idx, batch in enumerate(train_loader):
        # Extract force data
        # Assuming batch contains 'force' or 'tactile' field
        if isinstance(batch, dict):
            if 'force' in batch:
                force_data = batch['force']
            elif 'tactile' in batch:
                force_data = batch['tactile']
            else:
                raise KeyError(f"Expected 'force' or 'tactile' in batch, got keys: {batch.keys()}")
        else:
            force_data = batch

        # Expected shape: [B, T, 12] or [B, T, num_fingers, force_dim]
        if force_data.ndim == 3 and force_data.shape[-1] == num_fingers * force_dim:
            # Reshape [B, T, 12] -> [B, T, 2, 6]
            B, T, _ = force_data.shape
            force_data = force_data.reshape(B, T, num_fingers, force_dim)

        assert force_data.shape[-2] == num_fingers, f"Expected num_fingers={num_fingers}, got {force_data.shape}"
        assert force_data.shape[-1] == force_dim, f"Expected force_dim={force_dim}, got {force_data.shape}"

        # Convert to numpy
        if torch.is_tensor(force_data):
            force_data = force_data.cpu().numpy()

        # Flatten batch and time dimensions: [B, T, F, D] -> [B*T, F, D]
        B, T, F, D = force_data.shape
        force_flat = force_data.reshape(-1, F, D)  # [N, F, D]

        # Welford's online algorithm for each (finger, channel)
        for sample in force_flat:
            count += 1
            delta = sample - mean
            mean += delta / count
            delta2 = sample - mean
            M2 += delta * delta2

        if (batch_idx + 1) % 100 == 0:
            print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")

    # Compute variance and std
    if count < 2:
        raise ValueError("Need at least 2 samples to compute variance")

    variance = M2 / count
    std = np.sqrt(variance)

    # Avoid division by zero
    std = np.maximum(std, 1e-6)

    # Build stats dictionary
    stats = {
        'channel_mean': mean.tolist(),
        'channel_std': std.tolist(),
        'num_fingers': num_fingers,
        'force_dim': force_dim,
        'num_samples': int(count),
        'channel_order': ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'],
        'finger_order': ['finger_0', 'finger_1'],
        'data_source': str(dataloader_config_path),
        'split': 'train',
    }

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {output_path}")
    print(f"Total samples: {count}")
    print(f"\nPer-finger, per-channel mean:")
    print(mean)
    print(f"\nPer-finger, per-channel std:")
    print(std)

    return stats


def load_tactile_stats(stats_path):
    """Load tactile stats from JSON file."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    mean = np.array(stats['channel_mean'])
    std = np.array(stats['channel_std'])

    return mean, std, stats


def normalize_tactile_data(data, mean, std):
    """
    Normalize tactile data using precomputed statistics.

    Args:
        data: [..., num_fingers, force_dim] array or tensor
        mean: [num_fingers, force_dim] array
        std: [num_fingers, force_dim] array

    Returns:
        Normalized data with same shape as input
    """
    if torch.is_tensor(data):
        mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
        std = torch.tensor(std, dtype=data.dtype, device=data.device)

    return (data - mean) / std


def denormalize_tactile_data(data, mean, std):
    """
    Denormalize tactile data.

    Args:
        data: [..., num_fingers, force_dim] array or tensor
        mean: [num_fingers, force_dim] array
        std: [num_fingers, force_dim] array

    Returns:
        Denormalized data with same shape as input
    """
    if torch.is_tensor(data):
        mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
        std = torch.tensor(std, dtype=data.dtype, device=data.device)

    return data * std + mean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute tactile force statistics")
    parser.add_argument(
        '--dataloader_config',
        type=str,
        default='dataloader/vqvae_tactile.yaml',
        help='Path to dataloader config'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='TactileSelfencoder/tactile_stats.json',
        help='Output path for statistics JSON'
    )
    parser.add_argument(
        '--num_fingers',
        type=int,
        default=2,
        help='Number of fingers'
    )
    parser.add_argument(
        '--force_dim',
        type=int,
        default=6,
        help='Force dimension per finger'
    )

    args = parser.parse_args()

    stats = compute_tactile_stats(
        dataloader_config_path=args.dataloader_config,
        output_path=args.output,
        num_fingers=args.num_fingers,
        force_dim=args.force_dim
    )
