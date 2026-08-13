"""
Compute per-finger per-channel statistics from the training dataset.
Adapted for 2-finger T-Rex VQ-VAE.

Usage:
    python TactileSelfencoder/compute_trex_stats.py \
        --data_config dataloader/vqvae_tactile.yaml \
        --output TactileSelfencoder/trex_tactile_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

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


def compute_statistics(dataloader, n_fingers=2, force_dim=6):
    """Compute per-finger per-channel mean and std from training data.

    Args:
        dataloader: Training dataloader
        n_fingers: Number of fingers (2 for our system)
        force_dim: Dimension of force/torque per finger (6)

    Returns:
        dict with keys: mean, std, min, max, count
    """
    print(f"Computing statistics over {len(dataloader)} batches...")

    # Accumulators
    sum_vals = np.zeros((n_fingers, force_dim))
    sum_sq_vals = np.zeros((n_fingers, force_dim))
    min_vals = np.full((n_fingers, force_dim), np.inf)
    max_vals = np.full((n_fingers, force_dim), -np.inf)
    count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing stats")):
        # Extract tactile_history from batch
        if 'tactile_history' in batch:
            force = batch['tactile_history']  # [B, T, D] where D=12
        else:
            raise ValueError(f"Cannot find 'tactile_history' in batch keys: {batch.keys()}")

        if isinstance(force, torch.Tensor):
            force = force.cpu().numpy()

        # Validate shape
        if force.ndim == 4:
            B, T, F, D = force.shape
            if F != n_fingers or D != force_dim:
                raise ValueError(
                    f"Expected shape [B, T, {n_fingers}, {force_dim}], "
                    f"got {force.shape}"
                )
        elif force.ndim == 3:
            # Might be [B, T, 12] - reshape to [B, T, 2, 6]
            B, T, total_dim = force.shape
            if total_dim == n_fingers * force_dim:
                force = force.reshape(B, T, n_fingers, force_dim)
            else:
                raise ValueError(f"Cannot reshape {force.shape} to [B, T, 2, 6]")
        else:
            raise ValueError(f"Unexpected force shape: {force.shape}")

        # Update statistics
        # Reshape to [B*T, 2, 6] for efficient computation
        force_flat = force.reshape(-1, n_fingers, force_dim)  # [B*T, 2, 6]

        sum_vals += force_flat.sum(axis=0)  # [2, 6]
        sum_sq_vals += (force_flat ** 2).sum(axis=0)  # [2, 6]
        min_vals = np.minimum(min_vals, force_flat.min(axis=0))
        max_vals = np.maximum(max_vals, force_flat.max(axis=0))
        count += force_flat.shape[0]  # B*T

    # Compute mean and std
    mean = sum_vals / count  # [2, 6]
    var = (sum_sq_vals / count) - (mean ** 2)
    var = np.maximum(var, 1e-8)  # Avoid negative variance due to numerical errors
    std = np.sqrt(var)  # [2, 6]

    # Add small epsilon to std to avoid division by zero
    std = np.maximum(std, 1e-6)

    return {
        'mean': mean.tolist(),  # [2, 6]
        'std': std.tolist(),    # [2, 6]
        'min': min_vals.tolist(),  # [2, 6]
        'max': max_vals.tolist(),  # [2, 6]
        'count': int(count),
        'n_samples': int(count),
        'n_fingers': n_fingers,
        'force_dim': force_dim,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-finger per-channel statistics for T-Rex VQ-VAE"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to data config YAML (e.g., dataloader/vqvae_tactile.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save statistics JSON",
    )
    parser.add_argument(
        "--n_fingers",
        type=int,
        default=2,
        help="Number of fingers (default: 2)",
    )
    parser.add_argument(
        "--force_dim",
        type=int,
        default=6,
        help="Dimension of force/torque per finger (default: 6)",
    )
    args = parser.parse_args()

    # Load data config
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"Loading training dataloader from: {args.data_config}")

    # Set seed for reproducibility
    set_seed(data_config['split']['seed'])

    # Build base dataset
    print("Building base dataset...")
    base_dataset = build_base_dataset(data_config)

    # Build dataloaders
    print("Building dataloaders...")
    dataloaders, datasets = build_normal_dataloaders(data_config, base_dataset)

    # Get training dataloader
    if 'train' not in dataloaders:
        raise ValueError("No training split found in dataloaders")

    train_loader = dataloaders['train']

    print(f"Training dataset size: {len(datasets['train'])} samples")
    print(f"Batch size: {data_config['loader']['batch_size']}")
    print(f"Number of batches: {len(train_loader)}")

    # Compute statistics
    stats = compute_statistics(
        train_loader,
        n_fingers=args.n_fingers,
        force_dim=args.force_dim,
    )

    # Add metadata
    stats['source'] = str(Path(args.data_config).resolve())
    stats['data_config'] = data_config
    stats['description'] = (
        "Per-finger per-channel statistics for T-Rex VQ-VAE 2-finger adaptation. "
        "Computed from training set only."
    )
    stats['channel_order'] = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    stats['finger_order'] = ['finger_0', 'finger_1']
    stats['units'] = 'raw sensor units (pre-normalization)'

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Statistics saved to: {output_path}")
    print(f"\nStatistics summary:")
    print(f"  Total samples: {stats['count']:,}")
    print(f"  Fingers: {stats['n_fingers']}")
    print(f"  Channels per finger: {stats['force_dim']}")
    print(f"\nMean (per finger):")
    mean = np.array(stats['mean'])
    for f in range(args.n_fingers):
        print(f"  Finger {f}: {mean[f]}")
    print(f"\nStd (per finger):")
    std = np.array(stats['std'])
    for f in range(args.n_fingers):
        print(f"  Finger {f}: {std[f]}")


if __name__ == "__main__":
    main()
