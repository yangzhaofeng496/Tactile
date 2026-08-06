"""
Visualize VQ-VAE clustering: show which samples map to which codebook entries.

This script creates scatter plots showing:
1. Encoder outputs (z_e) projected to 2D via t-SNE/PCA
2. Color-coded by assigned codebook index
3. Codebook embeddings overlaid on the same plot
"""

import argparse
from pathlib import Path
import yaml

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from vqvae_model import build_vqvae_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE clustering")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to VQ-VAE checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("TactileSelfencoder/vqvae_config.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which data split to visualize"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of samples to visualize (for speed)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["tsne", "pca"],
        help="Dimensionality reduction method"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot (default: auto-generated)"
    )
    return parser.parse_args()


def build_force_dataloader(config):
    """Build dataloader for force history data."""
    import sys
    import os

    original_dir = os.getcwd()
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import dataloader.dataloader as dl

        batch_size = config['training']['batch_size']
        num_workers = config['training']['num_workers']
        history_steps = config['model']['input']['history_steps']

        dataloader_config_path = project_root / config['data']['dataloader_config']
        dataloader_config = dl.load_yaml(dataloader_config_path)

        dataloader_config['loader']['batch_size'] = batch_size
        dataloader_config['loader']['num_workers'] = num_workers
        dataloader_config['sequence']['length'] = history_steps

        base_dataset = dl.build_base_dataset(dataloader_config)

        dataloaders_dict, datasets_dict = dl.build_normal_dataloaders(
            dataloader_config,
            base_dataset
        )

        return dataloaders_dict

    finally:
        os.chdir(original_dir)


def collect_embeddings(model, dataloader, device, max_samples=5000):
    """
    Collect encoder outputs and codebook indices for visualization.

    Returns:
        z_e_all: [N, D] encoder outputs (before quantization)
        indices_all: [N] codebook indices assigned to each sample
        codebook: [K, D] codebook embeddings
    """
    model.eval()

    z_e_list = []
    indices_list = []

    total_samples = 0

    print(f"Collecting embeddings (max {max_samples} samples)...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            if total_samples >= max_samples:
                break

            x = batch["tactile_history"].to(device)

            # Get encoder output
            z_e = model.encoder(x)  # [B, D]
            z_e_normalized = model.quantizer.rms_norm(z_e)

            # Get quantized indices
            indices, _ = model.encode(x)  # [B]

            # Store
            remaining = max_samples - total_samples
            take = min(z_e_normalized.shape[0], remaining)

            z_e_list.append(z_e_normalized[:take].cpu())
            indices_list.append(indices[:take].cpu())

            total_samples += take

    z_e_all = torch.cat(z_e_list, dim=0).numpy()  # [N, D]
    indices_all = torch.cat(indices_list, dim=0).numpy()  # [N]
    codebook = model.quantizer.embedding.cpu().numpy()  # [K, D]

    print(f"Collected {z_e_all.shape[0]} samples")
    print(f"Encoder output dim: {z_e_all.shape[1]}")
    print(f"Codebook size: {codebook.shape[0]}")

    return z_e_all, indices_all, codebook


def reduce_dimensions(z_e, codebook, method="tsne"):
    """
    Reduce encoder outputs and codebook to 2D for visualization.

    Args:
        z_e: [N, D] encoder outputs
        codebook: [K, D] codebook embeddings
        method: "tsne" or "pca"

    Returns:
        z_e_2d: [N, 2]
        codebook_2d: [K, 2]
    """
    # Combine for joint reduction
    combined = np.vstack([z_e, codebook])

    print(f"Reducing dimensions using {method.upper()}...")

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        combined_2d = reducer.fit_transform(combined)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
        combined_2d = reducer.fit_transform(combined)

        explained_var = reducer.explained_variance_ratio_
        print(f"PCA explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {explained_var.sum():.2%}")

    # Split back
    z_e_2d = combined_2d[:len(z_e)]
    codebook_2d = combined_2d[len(z_e):]

    return z_e_2d, codebook_2d


def plot_clustering(z_e_2d, indices, codebook_2d, output_path, method="tsne"):
    """
    Create scatter plot of clustering.

    Args:
        z_e_2d: [N, 2] encoder outputs in 2D
        indices: [N] codebook indices
        codebook_2d: [K, 2] codebook embeddings in 2D
        output_path: where to save the plot
        method: dimensionality reduction method used
    """
    num_codes = codebook_2d.shape[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a colormap with distinct colors
    cmap = plt.cm.get_cmap('tab10' if num_codes <= 10 else 'tab20')

    # Plot samples colored by their assigned codebook
    for code_idx in range(num_codes):
        mask = indices == code_idx
        count = mask.sum()

        if count > 0:
            ax.scatter(
                z_e_2d[mask, 0],
                z_e_2d[mask, 1],
                c=[cmap(code_idx)],
                label=f'Code {code_idx} ({count})',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )

    # Plot codebook embeddings as large stars
    ax.scatter(
        codebook_2d[:, 0],
        codebook_2d[:, 1],
        c=[cmap(i) for i in range(num_codes)],
        marker='*',
        s=500,
        edgecolors='black',
        linewidths=2,
        label='Codebook centers',
        zorder=10
    )

    # Add codebook index labels
    for i, (x, y) in enumerate(codebook_2d):
        ax.annotate(
            f'{i}',
            (x, y),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            color='white',
            zorder=11
        )

    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(
        f'VQ-VAE Clustering Visualization ({method.upper()})\n'
        f'Samples colored by assigned codebook index',
        fontsize=14,
        fontweight='bold'
    )

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        framealpha=0.9
    )

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")

    plt.close(fig)


def plot_usage_histogram(indices, num_codes, output_path):
    """Plot histogram of codebook usage."""
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = np.bincount(indices, minlength=num_codes)

    ax.bar(range(num_codes), counts, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Codebook Index', fontsize=12)
    ax.set_ylabel('Usage Count', fontsize=12)
    ax.set_title('Codebook Usage Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

    used = (counts > 0).sum()
    usage_rate = used / num_codes * 100
    ax.text(
        0.02, 0.98,
        f'Active codes: {used}/{num_codes} ({usage_rate:.1f}%)',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Usage histogram saved to: {output_path}")

    plt.close(fig)


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")

    # Load model
    model, config = build_vqvae_from_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load data
    print("\nBuilding dataloaders...")
    dataloaders = build_force_dataloader(config)
    dataloader = dataloaders[args.split]

    # Collect embeddings
    z_e, indices, codebook = collect_embeddings(
        model, dataloader, device, max_samples=args.max_samples
    )

    # Reduce to 2D
    z_e_2d, codebook_2d = reduce_dimensions(z_e, codebook, method=args.method)

    # Generate output paths
    if args.output is None:
        checkpoint_name = args.checkpoint.stem
        output_dir = Path("TactileSelfencoder/clustering_viz")
        output_dir.mkdir(exist_ok=True, parents=True)

        clustering_path = output_dir / f"{checkpoint_name}_{args.split}_{args.method}_clustering.png"
        usage_path = output_dir / f"{checkpoint_name}_{args.split}_usage.png"
    else:
        clustering_path = args.output
        usage_path = args.output.parent / (args.output.stem + "_usage.png")

    # Create plots
    print("\nCreating visualizations...")
    plot_clustering(z_e_2d, indices, codebook_2d, clustering_path, method=args.method)
    plot_usage_histogram(indices, codebook.shape[0], usage_path)

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
