"""
Training script for T-Rex VQ-VAE adapted to 2-finger system.

Based on T-Rex official implementation:
https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/tactile_vqvae/train.py

Usage:
    python TactileSelfencoder/train_trex_vqvae.py \\
        --config TactileSelfencoder/trex_2finger_config.yaml \\
        --data_config dataloader/vqvae_tactile.yaml \\
        --stats TactileSelfencoder/trex_tactile_stats.json \\
        --output_dir outputs/trex_vqvae
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloader.dataloader import (
    build_base_dataset,
    build_normal_dataloaders,
    set_seed,
)
from trex.trex_official import TactileVQVAE, TactileVQVAEConfig


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_stats(stats_path):
    """Load pre-computed statistics."""
    with open(stats_path, 'r') as f:
        return json.load(f)


def normalize_force(force, stats):
    """Normalize force data using per-finger per-channel statistics.

    Args:
        force: [B, T, n_fingers, 6] or [B, T, n_fingers*6]
        stats: dict with 'mean' and 'std' keys, each [n_fingers, 6]

    Returns:
        normalized: [B, T, n_fingers, 6], normalized to ~N(0, 1)
    """
    if isinstance(force, np.ndarray):
        force = torch.from_numpy(force)

    # Handle shape
    if force.dim() == 3:
        B, T, total_dim = force.shape
        n_fingers, force_dim = stats['mean'].shape
        if total_dim == n_fingers * force_dim:
            force = force.reshape(B, T, n_fingers, force_dim)

    mean = torch.tensor(stats['mean'], dtype=force.dtype, device=force.device)  # [n_fingers, 6]
    std = torch.tensor(stats['std'], dtype=force.dtype, device=force.device)    # [n_fingers, 6]

    # Normalize
    normalized = (force - mean) / (std + 1e-6)
    return normalized


def compute_magnitude(force_raw):
    """Compute L2 magnitude of raw (un-normalized) force windows.

    Args:
        force_raw: [B, T, 2, 6] raw force data

    Returns:
        magnitude: [B] L2 norm over all dimensions
    """
    # Flatten and compute norm
    B = force_raw.shape[0]
    flat = force_raw.reshape(B, -1)  # [B, T*2*6]
    magnitude = torch.norm(flat, dim=1)  # [B]
    return magnitude


def cosine_lr_schedule(step, total_steps, warmup_steps, base_lr, min_lr_ratio):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def visualize_clustering_trex(z_e_list, indices_list, codebook, n_fingers, title="T-Rex Clustering"):
    """
    Visualize T-Rex VQ-VAE clustering with PCA.

    Args:
        z_e_list: list of encoder outputs [N, n_fingers, D]
        indices_list: list of codebook indices [N, n_fingers]
        codebook: [K, D] codebook embeddings
        n_fingers: number of fingers (2 for this system)
        title: plot title

    Returns:
        (fig_pca, fig_usage, stats): PCA scatter figure, usage bar chart, and statistics dict
    """
    # Concatenate all batches
    z_e_all = torch.cat(z_e_list, dim=0).cpu().numpy()  # [N, n_fingers, D]
    indices_all = torch.cat(indices_list, dim=0).cpu().numpy()  # [N, n_fingers]
    codebook_np = codebook.detach().cpu().numpy()  # [K, D]

    N, n_fingers, D = z_e_all.shape
    num_codes = codebook_np.shape[0]

    # Flatten across fingers for PCA: [N*n_fingers, D]
    z_e_flat = z_e_all.reshape(-1, D)
    indices_flat = indices_all.reshape(-1)  # [N*n_fingers]

    # Fit PCA on flattened features
    mean = z_e_flat.mean(axis=0)
    centered = z_e_flat - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    z_e_2d = centered @ eigenvectors[:, :2]
    codebook_2d = (codebook_np - mean) @ eigenvectors[:, :2]

    # Usage statistics (across all fingers)
    counts = np.bincount(indices_flat, minlength=num_codes).astype(np.float64)
    fracs = counts / counts.sum()
    active = int((counts > 0).sum())
    probs = fracs[fracs > 0]
    perplexity = float(np.exp(-(probs * np.log(probs)).sum()))
    max_frac = float(fracs.max())
    stats = {
        'per_code_count': counts,
        'per_code_fraction': fracs,
        'active_codes': active,
        'perplexity': perplexity,
        'max_usage_fraction': max_frac,
    }

    # Plot 1: PCA scatter with finger colors
    fig_pca = plt.figure(figsize=(14, 10))
    ax = fig_pca.add_subplot(1, 1, 1)

    # Get colormap
    if num_codes <= 10:
        cmap = plt.colormaps.get_cmap('tab10') if hasattr(plt, 'colormaps') else plt.cm.tab10
    else:
        cmap = plt.colormaps.get_cmap('tab20') if hasattr(plt, 'colormaps') else plt.cm.tab20

    # Plot samples by code and finger
    finger_markers = ['o', 's', '^', 'D', 'v']  # Different markers for different fingers
    handles = []

    for code_idx in range(num_codes):
        for finger_idx in range(n_fingers):
            # Get samples for this code and finger
            mask = (indices_all[:, finger_idx] == code_idx)
            count = int(mask.sum())

            if count > 0:
                finger_z_e = z_e_all[mask, finger_idx, :]  # [count, D]
                finger_z_e_2d = (finger_z_e - mean) @ eigenvectors[:, :2]

                h = ax.scatter(
                    finger_z_e_2d[:, 0],
                    finger_z_e_2d[:, 1],
                    c=[cmap(code_idx)],
                    marker=finger_markers[finger_idx],
                    label=f'C{code_idx} F{finger_idx} ({count})' if finger_idx == 0 else None,
                    alpha=0.5,
                    s=30,
                    edgecolors='none'
                )
                if finger_idx == 0:
                    handles.append(h)

    # Plot codebook centers
    ax.scatter(
        codebook_2d[:, 0],
        codebook_2d[:, 1],
        c=[cmap(i) for i in range(num_codes)],
        marker='*',
        s=600,
        edgecolors='black',
        linewidths=2,
        zorder=10
    )

    # Add labels
    for i, (x, y) in enumerate(codebook_2d):
        ax.annotate(
            f'{i}',
            (x, y),
            fontsize=11,
            fontweight='bold',
            ha='center',
            va='center',
            color='white',
            zorder=11
        )

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'{title}\nPCA on encoder outputs (per-finger granularity)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Usage bar chart
    fig_usage = plt.figure(figsize=(12, 6))
    ax_usage = fig_usage.add_subplot(1, 1, 1)
    ax_usage.bar(range(num_codes), counts, color=[cmap(i) for i in range(num_codes)])
    ax_usage.set_xlabel('Code index')
    ax_usage.set_ylabel('Sample count (all fingers)')
    ax_usage.set_title(f'{title} — Codebook Usage\nactive={active}/{num_codes} | '
                       f'ppl={perplexity:.2f} | max_frac={max_frac*100:.1f}%',
                       fontsize=12)
    ax_usage.set_xticks(range(num_codes))
    ax_usage.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig_pca, fig_usage, stats


def visualize_clustering_trex_3d(z_e_list, indices_list, codebook, n_fingers, title="T-Rex Clustering 3D"):
    """
    Interactive 3D PCA scatter (top 3 PCs) rendered with Plotly for wandb.

    Same PCA fit as the 2D version: fitted ONLY on the flattened encoder
    outputs (across fingers), codebook centers transformed with the same PCA.

    Args:
        z_e_list: list of encoder outputs [N, n_fingers, D]
        indices_list: list of codebook indices [N, n_fingers]
        codebook: [K, D] codebook embeddings
        n_fingers: number of fingers
        title: plot title

    Returns:
        plotly.graph_objects.Figure: interactive 3D scatter figure
    """
    import numpy as np
    import plotly.graph_objects as go

    z_e_all = torch.cat(z_e_list, dim=0).cpu().numpy()      # [N, n_fingers, D]
    indices_all = torch.cat(indices_list, dim=0).cpu().numpy()  # [N, n_fingers]
    codebook_np = codebook.detach().cpu().numpy()           # [K, D]

    N, n_fingers, D = z_e_all.shape
    num_codes = codebook_np.shape[0]

    z_e_flat = z_e_all.reshape(-1, D)                       # [N*n_fingers, D]
    indices_flat = indices_all.reshape(-1)                  # [N*n_fingers]

    mean = z_e_flat.mean(axis=0)
    centered = z_e_flat - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    z_e_3d = centered @ eigenvectors[:, :3]
    codebook_3d = (codebook_np - mean) @ eigenvectors[:, :3]

    # Color mapping shared with the 2D plot.
    if num_codes <= 10:
        cmap = plt.colormaps.get_cmap('tab10') if hasattr(plt, 'colormaps') else plt.cm.tab10
    else:
        cmap = plt.colormaps.get_cmap('tab20') if hasattr(plt, 'colormaps') else plt.cm.tab20
    rgba = lambda i: f'rgba({int(cmap(i)[0]*255)},{int(cmap(i)[1]*255)},{int(cmap(i)[2]*255)},0.6)'

    finger_markers = ['circle', 'diamond', 'square', 'x', 'cross']  # per-finger symbols

    fig = go.Figure()

    for code_idx in range(num_codes):
        for finger_idx in range(min(n_fingers, len(finger_markers))):
            mask = (indices_all[:, finger_idx] == code_idx)
            count = int(mask.sum())
            if count == 0:
                continue
            pts = z_e_all[mask, finger_idx, :]
            pts_3d = (pts - mean) @ eigenvectors[:, :3]
            fig.add_trace(go.Scatter3d(
                x=pts_3d[:, 0], y=pts_3d[:, 1], z=pts_3d[:, 2],
                mode='markers',
                name=f'C{code_idx} F{finger_idx} ({count})',
                marker=dict(
                    size=2, symbol=finger_markers[finger_idx], color=rgba(code_idx), opacity=0.6),
                hovertemplate=f'Code {code_idx} Finger {finger_idx}<br>'
                              f'x=%{{x:.3f}} y=%{{y:.3f}} z=%{{z:.3f}}',
            ))

    # Codebook centers
    fig.add_trace(go.Scatter3d(
        x=codebook_3d[:, 0], y=codebook_3d[:, 1], z=codebook_3d[:, 2],
        mode='markers+text',
        name='Codebook centers',
        text=[str(i) for i in range(num_codes)],
        textposition='top center',
        textfont=dict(size=10, color='white'),
        marker=dict(size=8, symbol='diamond', color='black',
                    line=dict(width=1, color='white'), opacity=1.0),
    ))

    fig.update_layout(
        title=f'{title}<br>PCA on encoder outputs (top 3 PCs, per-finger granularity)',
        height=800,
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        legend=dict(font=dict(size=9)),
    )

    return fig


def train_epoch(model, dataloader, optimizer, stats, config, epoch, global_step, device, use_wandb=False):
    """Train for one epoch."""
    model.train()

    epoch_metrics = {
        'recon_loss': 0.0,
        'vq_loss': 0.0,
        'total_loss': 0.0,
        'perplexity': 0.0,
        'active_codes': 0.0,
        'revived': 0,
        'n_batches': 0,
    }

    train_cfg = config['train']
    total_steps = train_cfg['epochs'] * len(dataloader)

    # For clustering visualization
    collect_for_clustering = use_wandb
    z_e_list = []
    indices_list = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        # Get tactile_history from batch (dataloader returns dict with 'tactile_history' key)
        force_raw = batch['tactile_history']  # [B, T, n_fingers*6]

        # Reshape from [B, T, n_fingers*6] to [B, T, n_fingers, 6]
        B, T, total_dim = force_raw.shape
        n_fingers = config['model']['n_fingers']
        per_finger_dim = config['model']['per_finger_dim']
        if total_dim != n_fingers * per_finger_dim:
            raise ValueError(
                f"Expected tactile_history shape [B, T, {n_fingers*per_finger_dim}], "
                f"got {force_raw.shape}")

        force_raw = force_raw.reshape(B, T, n_fingers, per_finger_dim).to(device)

        # Normalize
        force_norm = normalize_force(force_raw, stats)  # [B, T, 2, 6]

        # Compute magnitude (for weighted loss)
        magnitude = compute_magnitude(force_raw)  # [B]

        # Learning rate schedule
        lr = cosine_lr_schedule(
            global_step,
            total_steps,
            train_cfg['warmup_steps'],
            train_cfg['lr'],
            train_cfg['min_lr_ratio'],
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        optimizer.zero_grad()
        output = model(force_norm, magnitude)

        loss = output['total_loss']

        # Backward pass
        loss.backward()
        if train_cfg.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_cfg['grad_clip']
            )
        optimizer.step()

        # Collect embeddings for clustering visualization (need to run encoder separately)
        if collect_for_clustering:
            with torch.no_grad():
                z_e = model.encoder(force_norm)  # [B, n_fingers, D] for per-finger mode
                z_e_list.append(z_e.detach().cpu())
                indices_list.append(output['indices'].detach().cpu())  # [B, n_fingers]

        # Update metrics
        epoch_metrics['recon_loss'] += output['recon_loss'].item()
        epoch_metrics['vq_loss'] += output['vq_loss'].item()
        epoch_metrics['total_loss'] += output['total_loss'].item()
        epoch_metrics['perplexity'] += output['perplexity'].item()
        epoch_metrics['active_codes'] += output['active_codes'].item()
        epoch_metrics['revived'] += output['revived'].item()
        epoch_metrics['n_batches'] += 1

        # Log to wandb during training
        wandb_log_every = train_cfg.get('wandb_log_every', 50)
        if use_wandb and (batch_idx + 1) % wandb_log_every == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_recon_loss': output['recon_loss'].item(),
                'train/batch_vq_loss': output['vq_loss'].item(),
                'train/batch_perplexity': output['perplexity'].item(),
                'train/batch_active_codes': output['active_codes'].item(),
                'train/lr': lr,
                'global_step': global_step,
            })

        # Update progress bar
        if batch_idx % train_cfg.get('log_every', 50) == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{output['recon_loss'].item():.4f}",
                'vq': f"{output['vq_loss'].item():.4f}",
                'perp': f"{output['perplexity'].item():.1f}",
                'active': f"{int(output['active_codes'].item())}/{config['model']['codebook_size']}",
                'lr': f"{lr:.2e}",
            })

        global_step += 1

    # Average metrics
    for key in ['recon_loss', 'vq_loss', 'total_loss', 'perplexity', 'active_codes']:
        epoch_metrics[key] /= epoch_metrics['n_batches']

    # Add collected embeddings for visualization
    if collect_for_clustering:
        epoch_metrics['z_e_list'] = z_e_list
        epoch_metrics['indices_list'] = indices_list

    return epoch_metrics, global_step


@torch.no_grad()
def validate(model, dataloader, stats, config, device, collect_embeddings=False, max_samples=2000):
    """Validate the model."""
    model.eval()

    val_metrics = {
        'recon_loss': 0.0,
        'vq_loss': 0.0,
        'total_loss': 0.0,
        'perplexity': 0.0,
        'active_codes': 0.0,
        'n_batches': 0,
    }

    z_e_list = []
    indices_list = []
    n_samples = 0

    for batch in tqdm(dataloader, desc="Validation"):
        # Get tactile_history from batch
        force_raw = batch['tactile_history']  # [B, T, n_fingers*6]

        # Reshape from [B, T, n_fingers*6] to [B, T, n_fingers, 6]
        B, T, total_dim = force_raw.shape
        n_fingers = config['model']['n_fingers']
        per_finger_dim = config['model']['per_finger_dim']
        if total_dim != n_fingers * per_finger_dim:
            raise ValueError(
                f"Expected tactile_history shape [B, T, {n_fingers*per_finger_dim}], "
                f"got {force_raw.shape}")

        force_raw = force_raw.reshape(B, T, n_fingers, per_finger_dim).to(device)

        # Normalize
        force_norm = normalize_force(force_raw, stats)
        magnitude = compute_magnitude(force_raw)

        # Forward pass
        output = model(force_norm, magnitude)

        # Collect embeddings for visualization (need to run encoder separately)
        if collect_embeddings and n_samples < max_samples:
            z_e = model.encoder(force_norm)  # [B, n_fingers, D] for per-finger mode
            z_e_list.append(z_e.detach().cpu())
            indices_list.append(output['indices'].detach().cpu())
            n_samples += B

        # Update metrics
        val_metrics['recon_loss'] += output['recon_loss'].item()
        val_metrics['vq_loss'] += output['vq_loss'].item()
        val_metrics['total_loss'] += output['total_loss'].item()
        val_metrics['perplexity'] += output['perplexity'].item()
        val_metrics['active_codes'] += output['active_codes'].item()
        val_metrics['n_batches'] += 1

    # Average metrics
    for key in val_metrics:
        if key != 'n_batches':
            val_metrics[key] /= val_metrics['n_batches']

    if collect_embeddings and z_e_list:
        val_metrics['z_e_list'] = z_e_list
        val_metrics['indices_list'] = indices_list

    return val_metrics


def save_checkpoint(model, optimizer, stats, config, epoch, global_step, output_dir):
    """Save checkpoint."""
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'config': config,
        'stats': stats,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save epoch checkpoint
    ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, ckpt_path)

    # Save latest
    latest_path = output_dir / "latest.pt"
    torch.save(checkpoint, latest_path)

    print(f"✓ Checkpoint saved: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train T-Rex VQ-VAE (2-finger)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to data config YAML"
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="Path to pre-computed statistics JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load configurations
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    print(f"Loading statistics from: {args.stats}")
    stats = load_stats(args.stats)

    print(f"Loading data from: {args.data_config}")
    data_config = load_config(args.data_config)

    # Set seed
    set_seed(data_config['split']['seed'])

    # Build dataset and dataloaders
    print("Building base dataset...")
    base_dataset = build_base_dataset(data_config)

    print("Building dataloaders...")
    dataloaders, datasets = build_normal_dataloaders(data_config, base_dataset)

    if 'train' not in dataloaders:
        raise ValueError("No training split found")
    if 'val' not in dataloaders:
        raise ValueError("No validation split found")

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    print(f"Train dataset: {len(datasets['train'])} samples")
    print(f"Val dataset: {len(datasets['val'])} samples")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model_cfg = TactileVQVAEConfig(
        window=config['model']['window'],
        in_channels=config['model']['n_fingers'] * config['model']['per_finger_dim'],
        hidden_channels=config['model']['hidden_channels'],
        bottleneck_channels=config['model']['bottleneck_channels'],
        embed_dim=config['model']['embed_dim'],
        n_strided_blocks=config['model']['n_strided_blocks'],
        codebook_size=config['model']['codebook_size'],
        commitment_weight=config['model']['commitment_weight'],
        decay=config['model']['decay'],
        revive_freq=config['model']['revive_freq'],
        revive_threshold=config['model']['revive_threshold'],
        use_magnitude_weight=config['model']['use_magnitude_weight'],
        weight_alpha=config['model']['weight_alpha'],
        weight_tau=config['model']['weight_tau'],
        granularity=config['model']['granularity'],
        n_fingers=config['model']['n_fingers'],
        per_finger_dim=config['model']['per_finger_dim'],
        init_mode=config['model'].get('init_mode', 'uniform'),
    )

    model = TactileVQVAE(model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train'].get('weight_decay', 1e-4),
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']

    # Setup wandb
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', False)

    if use_wandb:
        run_name = wandb_config.get('run_name', None)
        wandb.init(
            project=wandb_config.get('project', 'trex-vqvae'),
            name=run_name,
            config={
                'model': config['model'],
                'train': config['train'],
                'data': data_config,
                'n_params': n_params,
            },
            tags=wandb_config.get('tags', ['trex', '2-finger', 'vqvae']),
        )
        print(f"✓ Wandb initialized: {wandb.run.name}")
    else:
        print("✗ Wandb disabled")

    # Training loop
    print(f"\nStarting training for {config['train']['epochs']} epochs...")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, config['train']['epochs']):
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, stats, config,
            epoch, global_step, device, use_wandb=use_wandb
        )

        print(f"\nEpoch {epoch+1}/{config['train']['epochs']} Summary:")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, "
              f"Perplexity: {train_metrics['perplexity']:.1f}, "
              f"Active: {int(train_metrics['active_codes'])}/{config['model']['codebook_size']}")

        # Validate
        val_metrics = validate(
            model, val_loader, stats, config, device,
            collect_embeddings=False, max_samples=2000
        )
        print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
              f"Recon: {val_metrics['recon_loss']:.4f}, "
              f"VQ: {val_metrics['vq_loss']:.4f}, "
              f"Perplexity: {val_metrics['perplexity']:.1f}, "
              f"Active: {int(val_metrics['active_codes'])}/{config['model']['codebook_size']}")

        # Log epoch metrics to wandb
        if use_wandb:
            log_dict = {
                'train/epoch_loss': train_metrics['total_loss'],
                'train/epoch_recon_loss': train_metrics['recon_loss'],
                'train/epoch_vq_loss': train_metrics['vq_loss'],
                'train/epoch_perplexity': train_metrics['perplexity'],
                'train/epoch_active_codes': train_metrics['active_codes'],
                'train/epoch_revived': train_metrics['revived'],
                'val/epoch_loss': val_metrics['total_loss'],
                'val/epoch_recon_loss': val_metrics['recon_loss'],
                'val/epoch_vq_loss': val_metrics['vq_loss'],
                'val/epoch_perplexity': val_metrics['perplexity'],
                'val/epoch_active_codes': val_metrics['active_codes'],
                'epoch': epoch + 1,
            }
            wandb.log(log_dict)

            # Visualize clustering at end of epoch
            if 'z_e_list' in train_metrics and train_metrics['z_e_list']:
                fig_pca, fig_usage, cluster_stats = visualize_clustering_trex(
                    train_metrics['z_e_list'],
                    train_metrics['indices_list'],
                    model.quantizer.embed,
                    config['model']['n_fingers'],
                    title=f'Training Set - Epoch {epoch+1}'
                )
                wandb.log({
                    'clustering/epoch_train': wandb.Image(fig_pca),
                    'clustering/epoch_train_usage': wandb.Image(fig_usage),
                    'clustering/epoch_train_active_codes': cluster_stats['active_codes'],
                    'clustering/epoch_train_perplexity': cluster_stats['perplexity'],
                    'clustering/epoch_train_max_usage_fraction': cluster_stats['max_usage_fraction'],
                    'epoch': epoch + 1,
                })

                fig_3d = visualize_clustering_trex_3d(
                    train_metrics['z_e_list'],
                    train_metrics['indices_list'],
                    model.quantizer.embed,
                    config['model']['n_fingers'],
                    title=f'Training Set - Epoch {epoch+1}'
                )
                wandb.log({
                    'clustering/epoch_train_3d': wandb.Plotly(fig_3d),
                    'epoch': epoch + 1,
                })

                plt.close(fig_pca)
                plt.close(fig_usage)

                # Clean up
                del train_metrics['z_e_list']
                del train_metrics['indices_list']

        # Save checkpoint
        if (epoch + 1) % config['train'].get('save_every_epoch', 5) == 0:
            save_checkpoint(
                model, optimizer, stats, config,
                epoch, global_step, args.output_dir
            )

        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_path = Path(args.output_dir) / "best.pt"
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'config': config,
                'stats': stats,
                'val_loss': best_val_loss,
            }
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, stats, config,
        config['train']['epochs'] - 1, global_step, args.output_dir
    )

    if use_wandb:
        wandb.finish()

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
