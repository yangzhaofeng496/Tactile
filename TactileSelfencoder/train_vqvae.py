"""
Training script for Temporal VQ-VAE.

Trains the VQ-VAE model on tactile force history data.
"""

import argparse
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vqvae_model import TemporalVQVAE, build_vqvae_from_config


def compute_magnitude(force):
    """L2 norm of the raw force window: [B, T, D] -> [B].

    Mirrors T-Rex's magnitude-weighted recon loss input: magnitude is computed
    on the raw (un-normalized) window so it reflects actual contact strength.
    """
    B = force.shape[0]
    flat = force.reshape(B, -1)  # [B, T*D]
    return torch.norm(flat, dim=1)  # [B]


def print_loss_formulas(config):
    """Print the loss formulas used in training."""
    commitment_cost = config['model']['quantizer']['commitment_cost']

    print("\n" + "="*70)
    print("LOSS FORMULAS")
    print("="*70)

    print("\n1. Reconstruction Loss (MSE):")
    print("   recon_loss = MSE(x_recon, x_original)")
    print("   where:")
    print("     x_original: [B, T, D] input tactile force history")
    print("     x_recon:    [B, T, D] reconstructed from decoder")
    print("   → Measures how well we can reconstruct the input")

    print(f"\n2. Commitment Loss (VQ Loss with EMA):")
    print(f"   commitment_loss = {commitment_cost} × MSE(z_e_normalized, stop_gradient(z_q))")
    print("   where:")
    print("     z_e_normalized: [B, D] RMS-normalized encoder output")
    print("     z_q:            [B, D] quantized embedding from codebook")
    print("     stop_gradient:  no gradient flows to z_q (only encoder learns)")
    print("   → Encourages encoder to 'commit' to codebook entries")

    print("\n3. Codebook Loss (EMA - No Gradient):")
    print("   codebook_loss = 0.0  (always zero with EMA)")
    print("   Codebook is updated via Exponential Moving Average:")
    print(f"     decay = {config['model']['quantizer']['ema_decay']}")
    print("     cluster_size_t = decay × cluster_size_{t-1} + (1-decay) × current_usage")
    print("     embedding_sum_t = decay × embedding_sum_{t-1} + (1-decay) × Σ(z_e_normalized)")
    print("     codebook_t = embedding_sum_t / cluster_size_t")
    print("   → Codebook learns without gradients")

    print("\n4. VQ Loss:")
    print("   vq_loss = commitment_loss + codebook_loss")
    print(f"           = commitment_loss + 0.0")
    print("           = commitment_loss")

    print("\n5. Total Loss:")
    print("   total_loss = recon_loss + vq_loss")
    print("              = recon_loss + commitment_loss")

    print("\n" + "="*70)
    print("GRADIENT FLOW")
    print("="*70)
    print("recon_loss → backprop → decoder + encoder")
    print("commitment_loss → backprop → encoder only (z_q is detached)")
    print("codebook → updated by EMA (no backprop)")
    print("="*70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Temporal VQ-VAE")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("TactileSelfencoder/vqvae_config.yaml"),
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from"
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

        print(f"Loading tactile force data from: {dataloader_config['dataset']['repo_id']}")
        print(f"History steps: {history_steps}")

        base_dataset = dl.build_base_dataset(dataloader_config)

        dataloaders_dict, datasets_dict = dl.build_normal_dataloaders(
            dataloader_config,
            base_dataset
        )

        train_loader = dataloaders_dict.get('train')
        val_loader = dataloaders_dict.get('val')
        test_loader = dataloaders_dict.get('test')

        tactile_channel_names = dataloader_config['dataset']['keys'].get(
            'tactile_force_channel_order', []
        )
        print(f"Tactile channels: {tactile_channel_names}")

        if val_loader is None:
            print("⚠ No validation set available")
        if test_loader is None:
            print("⚠ No test set available")

        return train_loader, val_loader, test_loader

    finally:
        os.chdir(original_dir)


def visualize_clustering_simple(z_e_list, indices_list, codebook, save_path, title="Clustering",
                                magnitudes_list=None):
    """
    PCA-based clustering visualization.

    PCA is fitted ONLY on the collected features (z_e_normalized). The codebook
    centers are transformed with that same fitted PCA, so scatter points and
    star markers live in a single consistent space.

    Args:
        z_e_list: list of normalized encoder outputs [N, D] (z_e_normalized)
        indices_list: list of codebook indices [N]
        codebook: [K, D] codebook embeddings
        save_path: where to save the plot (unused; caller uploads to wandb)
        title: plot title
        magnitudes_list: optional list of raw window magnitudes [N]. When given,
            samples are split into weak/strong by the median and drawn with
            different markers (circle = weak, triangle = strong).

    Returns:
        (fig_pca, fig_usage, stats): PCA scatter figure, a standalone usage
        bar-chart figure, and a dict with usage statistics.
    """
    import numpy as np

    # Concatenate all batches
    z_e_all = torch.cat(z_e_list, dim=0).cpu().numpy()  # [N, D]
    indices_all = torch.cat(indices_list, dim=0).cpu().numpy()  # [N]
    codebook_np = codebook.detach().cpu().numpy()  # [K, D]
    num_codes = codebook_np.shape[0]

    mag_all = None
    mag_thr = None
    if magnitudes_list is not None:
        mag_all = torch.cat(magnitudes_list, dim=0).cpu().numpy()  # [N]
        mag_thr = float(np.percentile(mag_all, 95))  # top 5% = strong

    # ---- Fit PCA on features ONLY (never on codebook+features combined) ----
    mean = z_e_all.mean(axis=0)
    centered = z_e_all - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    z_e_2d = centered @ eigenvectors[:, :2]
    # Transform codebook with the SAME fitted PCA (no separate fit)
    codebook_2d = (codebook_np - mean) @ eigenvectors[:, :2]

    # ---- Usage statistics ----
    counts = np.bincount(indices_all, minlength=num_codes).astype(np.float64)
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

    # ---- Plot 1: PCA scatter (with codebook centers in the SAME space) ----
    fig_pca = plt.figure(figsize=(12, 10))
    ax = fig_pca.add_subplot(1, 1, 1)

    # Get colormap (compatible with newer matplotlib versions)
    if num_codes <= 10:
        cmap = plt.colormaps.get_cmap('tab10') if hasattr(plt, 'colormaps') else plt.cm.tab10
    else:
        cmap = plt.colormaps.get_cmap('tab20') if hasattr(plt, 'colormaps') else plt.cm.tab20

    # Plot samples colored by actual nearest-neighbor assignment
    # Include ALL codes in the legend, even those with 0 samples.
    # When magnitudes are given: circle = weak, triangle = strong.
    handles = []
    for code_idx in range(num_codes):
        mask = indices_all == code_idx
        count = int(mask.sum())
        if count > 0:
            if mag_all is not None:
                weak = mask & (mag_all <= mag_thr)
                strong = mask & (mag_all > mag_thr)
                weak_cnt = int((mag_all[mask] <= mag_thr).sum())
                strong_cnt = count - weak_cnt
                if weak.any():
                    ax.scatter(
                        z_e_2d[weak, 0], z_e_2d[weak, 1],
                        c=[cmap(code_idx)], marker='o',
                        s=18, alpha=0.45, edgecolors='none'
                    )
                if strong.any():
                    ax.scatter(
                        z_e_2d[strong, 0], z_e_2d[strong, 1],
                        c=[cmap(code_idx)], marker='^',
                        s=35, alpha=0.85, edgecolors='none'
                    )
                # Legend proxy (only for codes with samples)
                h = ax.scatter([], [], c=[cmap(code_idx)], marker='o',
                               label=f'Code {code_idx} ({count}) 弱{weak_cnt}/强{strong_cnt}', s=20)
            else:
                h = ax.scatter(
                    z_e_2d[mask, 0],
                    z_e_2d[mask, 1],
                    c=[cmap(code_idx)],
                    label=f'Code {code_idx} ({count})',
                    alpha=0.6,
                    s=20,
                    edgecolors='none'
                )
        else:
            # Empty code: still add a legend entry (transparent proxy)
            h = ax.scatter(
                [], [],
                c=[cmap(code_idx)],
                label=f'Code {code_idx} (0)',
                alpha=0.6,
                s=20,
            )
        handles.append(h)

    # Plot codebook centers (same PCA space)
    ax.scatter(
        codebook_2d[:, 0],
        codebook_2d[:, 1],
        c=[cmap(i) for i in range(num_codes)],
        marker='*',
        s=500,
        edgecolors='black',
        linewidths=2,
        zorder=10
    )

    # Add labels
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

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'{title}\nPCA on z_e_normalized (features only)', fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: standalone usage bar chart (all codes, incl. zero-count) ----
    fig_usage = plt.figure(figsize=(12, 6))
    ax_usage = fig_usage.add_subplot(1, 1, 1)
    ax_usage.bar(range(num_codes), counts, color=[cmap(i) for i in range(num_codes)])
    ax_usage.set_xlabel('Code index')
    ax_usage.set_ylabel('Sample count')
    ax_usage.set_title(f'{title} — Usage\nactive={active}/{num_codes} | '
                       f'ppl={perplexity:.2f} | max_frac={max_frac*100:.1f}%',
                       fontsize=12)
    ax_usage.set_xticks(range(num_codes))
    ax_usage.grid(True, alpha=0.3)

    # Return both figures (caller uploads to wandb) and usage stats
    return fig_pca, fig_usage, stats


def visualize_clustering_3d(z_e_list, indices_list, codebook, title="Clustering 3D",
                            magnitudes_list=None):
    """
    Interactive 3D PCA scatter (top 3 PCs) rendered with Plotly for wandb.

    PCA is fitted ONLY on the collected features (z_e_normalized). The codebook
    centers are transformed with that same fitted PCA, so scatter points and
    star markers live in a single consistent space.

    Args:
        z_e_list: list of normalized encoder outputs [N, D] (z_e_normalized)
        indices_list: list of codebook indices [N]
        codebook: [K, D] codebook embeddings
        title: plot title
        magnitudes_list: optional list of raw window magnitudes [N]. When given,
            samples are split into weak/strong by the P95 and drawn with
            different markers (circle = weak, triangle = strong).

    Returns:
        plotly.graph_objects.Figure: interactive 3D scatter figure
    """
    import numpy as np
    import plotly.graph_objects as go

    z_e_all = torch.cat(z_e_list, dim=0).cpu().numpy()      # [N, D]
    indices_all = torch.cat(indices_list, dim=0).cpu().numpy()  # [N]
    codebook_np = codebook.detach().cpu().numpy()           # [K, D]
    num_codes = codebook_np.shape[0]

    mag_all = None
    mag_thr = None
    if magnitudes_list is not None:
        mag_all = torch.cat(magnitudes_list, dim=0).cpu().numpy()  # [N]
        mag_thr = float(np.percentile(mag_all, 95))  # top 5% = strong

    # Fit PCA on features ONLY.
    mean = z_e_all.mean(axis=0)
    centered = z_e_all - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    z_e_3d = centered @ eigenvectors[:, :3]
    codebook_3d = (codebook_np - mean) @ eigenvectors[:, :3]

    # Color mapping shared with the 2D plot (tab10/tab20).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if num_codes <= 10:
        cmap = plt.colormaps.get_cmap('tab10') if hasattr(plt, 'colormaps') else plt.cm.tab10
    else:
        cmap = plt.colormaps.get_cmap('tab20') if hasattr(plt, 'colormaps') else plt.cm.tab20
    rgba = lambda i: f'rgba({int(cmap(i)[0]*255)},{int(cmap(i)[1]*255)},{int(cmap(i)[2]*255)},0.6)'

    counts = np.bincount(indices_all, minlength=num_codes).astype(np.float64)

    fig = go.Figure()

    for code_idx in range(num_codes):
        mask = indices_all == code_idx
        count = int(mask.sum())
        color = rgba(code_idx)
        if count > 0:
            if mag_all is not None:
                weak = mask & (mag_all <= mag_thr)
                strong = mask & (mag_all > mag_thr)
                weak_cnt = int((mag_all[mask] <= mag_thr).sum())
                strong_cnt = count - weak_cnt
                name = f'Code {code_idx} ({count}) 弱{weak_cnt}/强{strong_cnt}'
                if weak.any():
                    fig.add_trace(go.Scatter3d(
                        x=z_e_3d[weak, 0], y=z_e_3d[weak, 1], z=z_e_3d[weak, 2],
                        mode='markers', name=name,
                        marker=dict(size=2, symbol='circle', color=color, opacity=0.5),
                        hovertemplate=f'Code {code_idx}<br>x=%{{x:.3f}} y=%{{y:.3f}} z=%{{z:.3f}}',
                        showlegend=True,
                    ))
                if strong.any():
                    fig.add_trace(go.Scatter3d(
                        x=z_e_3d[strong, 0], y=z_e_3d[strong, 1], z=z_e_3d[strong, 2],
                        mode='markers', name=name,
                        marker=dict(size=5, symbol='diamond', color=color, opacity=0.9),
                        hovertemplate=f'Code {code_idx}<br>x=%{{x:.3f}} y=%{{y:.3f}} z=%{{z:.3f}}',
                        showlegend=not weak.any(),
                    ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=z_e_3d[mask, 0], y=z_e_3d[mask, 1], z=z_e_3d[mask, 2],
                    mode='markers', name=f'Code {code_idx} ({count})',
                    marker=dict(size=2, color=color, opacity=0.7),
                    hovertemplate=f'Code {code_idx}<br>x=%{{x:.3f}} y=%{{y:.3f}} z=%{{z:.3f}}',
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers', name=f'Code {code_idx} (0)',
                marker=dict(size=2, color=color, opacity=0.7),
            ))

    # Codebook centers as large stars.
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
        title=f'{title}<br>PCA on z_e_normalized (top 3 PCs)',
        height=800,
        scene=dict(
            xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
        ),
        legend=dict(font=dict(size=9)),
    )

    return fig


def train_epoch(model, dataloader, optimizer, device, epoch, config, use_wandb=False):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_codebook_loss = 0
    total_commitment_loss = 0

    log_every = config['training']['log_every']
    wandb_log_every = config['training'].get('wandb_log_every', 10)

    # For clustering visualization - collect all samples throughout epoch
    collect_for_clustering = use_wandb or config['training'].get('save_local_plots', False)
    all_z_e = []
    all_indices = []
    all_magnitudes = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        x = batch["tactile_history"].to(device)

        if batch_idx == 0 and epoch == 1:
            print(f"\n{'='*70}")
            print(f"FIRST BATCH DIAGNOSTICS (Epoch {epoch})")
            print(f"{'='*70}")
            print(f"Input tactile_history shape: {x.shape}")
            print(f"Input mean: {x.mean():.6f}, std: {x.std():.6f}")
            print(f"Input min: {x.min():.6f}, max: {x.max():.6f}")

        history_steps = config['model']['input']['history_steps']
        if x.shape[1] != history_steps:
            x = x[:, :history_steps, :]

        # T-Rex style magnitude-weighted recon loss needs raw window magnitude.
        magnitude = compute_magnitude(x) if config['model'].get('use_magnitude_weight', False) else None
        output = model(x, magnitude)

        if batch_idx == 0 and epoch == 1:
            with torch.no_grad():
                # NOTE: reuse the exact tensors from this forward (output
                # carries the exact normalized encoder input that was fed to
                # the quantizer), instead of re-running model.encoder(x).
                z_e_raw = output['z_e']
                print(f"\nEncoder output (before RMS Norm):")
                print(f"  z_e shape: {z_e_raw.shape}")
                print(f"  z_e mean: {z_e_raw.mean():.6f}, std: {z_e_raw.std():.6f}")
                print(f"  z_e norm (avg): {z_e_raw.norm(dim=1).mean():.6f}")

                z_e_norm = output['z_e_normalized']
                print(f"\nEncoder output (after RMS Norm):")
                print(f"  z_e_normalized mean: {z_e_norm.mean():.6f}, std: {z_e_norm.std():.6f}")
                print(f"  z_e_normalized norm (avg): {z_e_norm.norm(dim=1).mean():.6f}")

                print(f"\nCodebook statistics:")
                print(f"  Codebook shape: {model.quantizer.embedding.shape}")
                print(f"  Codebook mean: {model.quantizer.embedding.mean():.6f}")
                print(f"  Codebook std: {model.quantizer.embedding.std():.6f}")
                print(f"  Codebook norm (avg): {model.quantizer.embedding.norm(dim=1).mean():.6f}")

                distances = torch.cdist(z_e_norm, model.quantizer.embedding)
                min_distances = distances.min(dim=1)[0]
                print(f"\nDistance to nearest codebook:")
                print(f"  Min distance (avg): {min_distances.mean():.6f}")
                print(f"  Min distance (std): {min_distances.std():.6f}")

                z_q = output['z_q']
                mse = F.mse_loss(z_e_norm, z_q.detach())
                print(f"\nCommitment MSE (before multiplying by {config['model']['quantizer']['commitment_cost']}):")
                print(f"  Raw MSE: {mse:.6f}")
                print(f"  Commitment loss: {output['commitment_loss'].item():.6f}")

                if magnitude is not None:
                    w = model._recon_weight(magnitude)
                    print(f"\nMagnitude-weighted recon loss:")
                    print(f"  magnitude: min={magnitude.min().item():.3f} "
                          f"median={magnitude.median().item():.3f} "
                          f"max={magnitude.max().item():.3f}")
                    print(f"  w        : min={w.min().item():.3f} "
                          f"median={w.median().item():.3f} "
                          f"max={w.max().item():.3f}")
                    print(f"  per-sample weights: {w.detach().cpu().numpy().round(3)}", flush=True)
            print(f"{'='*70}\n", flush=True)

        # Quick per-epoch magnitude/weight stats (every epoch, batch 0).
        if batch_idx == 0 and epoch > 1 and magnitude is not None:
            w = model._recon_weight(magnitude)
            print(f"[Epoch {epoch}] magnitude: "
                  f"min={magnitude.min().item():.1f} median={magnitude.median().item():.1f} "
                  f"max={magnitude.max().item():.1f} | w: "
                  f"min={w.min().item():.3f} median={w.median().item():.3f} "
                  f"max={w.max().item():.3f}", flush=True)

        loss = output['total_loss']

        optimizer.zero_grad()
        loss.backward()

        if 'grad_clip_norm' in config['training']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip_norm']
            )

        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += output['recon_loss'].item()
        total_vq_loss += output['vq_loss'].item()
        total_codebook_loss += output['codebook_loss'].item()
        total_commitment_loss += output['commitment_loss'].item()

        # Collect encoder outputs and indices for end-of-epoch clustering visualization
        # NOTE: use output['z_e_normalized'] (the exact tensor used for
        # quantization) so PCA points and codebook centers share the same space.
        if collect_for_clustering:
            all_z_e.append(output['z_e_normalized'].detach().cpu())
            all_indices.append(output['indices'].detach().cpu())
            if magnitude is not None:
                all_magnitudes.append(magnitude.detach().cpu())

        if use_wandb and (batch_idx + 1) % wandb_log_every == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_idx + 1

            # Compute encoder output norms for monitoring (reuse the exact
            # tensor from this forward, do NOT re-run rms_norm externally)
            with torch.no_grad():
                z_e_norm = output['z_e_normalized']
                encoder_norm_mean = z_e_norm.norm(dim=1).mean().item()
                codebook_norm_mean = model.quantizer.embedding.norm(dim=1).mean().item()
                scale_ratio = encoder_norm_mean / (codebook_norm_mean + 1e-10)

            wandb.log({
                'train_batch/total_loss': loss.item(),
                'train_batch/recon_loss': output['recon_loss'].item(),
                'train_batch/vq_loss': output['vq_loss'].item(),
                'train_batch/codebook_loss': output['codebook_loss'].item(),
                'train_batch/commitment_loss': output['commitment_loss'].item(),
                'codebook/encoder_norm_mean': encoder_norm_mean,
                'codebook/encoder_code_scale_ratio': scale_ratio,
                'global_step': global_step,
            }, step=global_step)

        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_recon = total_recon_loss / (batch_idx + 1)
            avg_vq = total_vq_loss / (batch_idx + 1)

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'recon': f'{avg_recon:.4f}',
                'vq': f'{avg_vq:.4f}'
            })

    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'codebook_loss': total_codebook_loss / num_batches,
        'commitment_loss': total_commitment_loss / num_batches,
    }

    # Return clustering data if collected
    if collect_for_clustering and all_z_e:
        metrics['z_e_all'] = torch.cat(all_z_e, dim=0)
        metrics['indices_all'] = torch.cat(all_indices, dim=0)
        if all_magnitudes:
            metrics['magnitudes_all'] = torch.cat(all_magnitudes, dim=0)

    return metrics


def validate(model, dataloader, device, split_name="Val", collect_embeddings=False, max_samples=2000):
    """Validate the model."""
    was_training = model.training
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_codebook_loss = 0
    total_commitment_loss = 0

    all_indices = []
    z_e_list = []
    indices_list = []
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[{split_name}]")

        for batch in pbar:
            x = batch["tactile_history"].to(device)

            history_steps = model.config['input']['history_steps']
            if x.shape[1] != history_steps:
                x = x[:, :history_steps, :]

            magnitude = compute_magnitude(x) if model.config.get('use_magnitude_weight', False) else None
            output = model(x, magnitude)

            total_loss += output['total_loss'].item()
            total_recon_loss += output['recon_loss'].item()
            total_vq_loss += output['vq_loss'].item()
            total_codebook_loss += output['codebook_loss'].item()
            total_commitment_loss += output['commitment_loss'].item()

            all_indices.append(output['indices'].cpu())

            # Collect embeddings for clustering visualization
            if collect_embeddings and total_samples < max_samples:
                z_e_normalized = output['z_e_normalized']

                remaining = max_samples - total_samples
                take = min(z_e_normalized.shape[0], remaining)

                z_e_list.append(z_e_normalized[:take].cpu())
                indices_list.append(output['indices'][:take].cpu())
                total_samples += take

    num_batches = len(dataloader)
    metrics = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'codebook_loss': total_codebook_loss / num_batches,
        'commitment_loss': total_commitment_loss / num_batches,
    }

    all_indices = torch.cat(all_indices, dim=0)
    num_embeddings = model.config['quantizer']['num_embeddings']
    usage_counts = torch.bincount(all_indices, minlength=num_embeddings)
    used_codes = (usage_counts > 0).sum().item()
    usage_rate = used_codes / num_embeddings

    metrics['codebook_usage'] = usage_rate
    metrics['used_codes'] = used_codes
    metrics['usage_counts'] = usage_counts.cpu().numpy()

    if collect_embeddings:
        metrics['z_e_list'] = z_e_list
        metrics['indices_list'] = indices_list

    # Restore the caller's train/eval state: validation must not leak a mode
    # change (nor any EMA update) into the surrounding training loop.
    model.train(was_training)

    return metrics



def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': model.config,
    }

    torch.save(checkpoint, save_path)

    print(f"✓ Checkpoint saved: {save_path}")


def main():
    args = parse_args()

    print(f"Loading config from: {args.config}")
    model, config = build_vqvae_from_config(args.config)

    print_loss_formulas(config)

    device_name = config['training']['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Codebook size: {config['model']['quantizer']['num_embeddings']}")
    print(f"Embedding dim: {config['model']['quantizer']['embedding_dim']}")

    # Dead code replacement status
    if config.get('dead_code_replacement', {}).get('enabled', False):
        print(f"✓ Dead code replacement: enabled")
    else:
        print(f"  Dead code replacement: disabled")

    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', False)

    if use_wandb:
        run_name = wandb_config.get('run_name', None)
        wandb.init(
            project=wandb_config.get('project', 'tactile-vqvae'),
            name=run_name,
            config={
                'model': config['model'],
                'training': config['training'],
                'total_params': total_params,
                'trainable_params': trainable_params,
            }
        )
        print(f"✓ Wandb initialized: {wandb.run.name}")

    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader = build_force_dataloader(config)
    print(f"Train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Val batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"Test batches: {len(test_loader)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    start_epoch = 0
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    num_epochs = config['training']['num_epochs']
    save_dir = Path(config['training']['save_dir'])
    save_every = config['training']['save_every']

    best_val_loss = float('inf')

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch+1, config, use_wandb=use_wandb)

        val_metrics = None
        if val_loader is not None:
            val_metrics = validate(model, val_loader, device, split_name="Val", collect_embeddings=False, max_samples=2000)

        # Visualize clustering at end of epoch
        save_local_plots = config['training'].get('save_local_plots', False)
        if use_wandb or save_local_plots:
            # Visualize training set clustering
            if 'z_e_all' in train_metrics:
                mags = train_metrics.get('magnitudes_all')
                fig_pca, fig_usage, cluster_stats = visualize_clustering_simple(
                    [train_metrics['z_e_all']],
                    [train_metrics['indices_all']],
                    model.quantizer.embedding,
                    None,
                    title=f'Training Set Clustering - Epoch {epoch+1}',
                    magnitudes_list=[mags] if mags is not None else None,
                )
                fig_3d = visualize_clustering_3d(
                    [train_metrics['z_e_all']],
                    [train_metrics['indices_all']],
                    model.quantizer.embedding,
                    title=f'Training Set Clustering - Epoch {epoch+1}',
                    magnitudes_list=[mags] if mags is not None else None,
                )

                if use_wandb:
                    wandb.log({
                        'clustering/epoch_train': wandb.Image(fig_pca),
                        'clustering/epoch_train_usage': wandb.Image(fig_usage),
                        'clustering/epoch_train_active_codes': cluster_stats['active_codes'],
                        'clustering/epoch_train_perplexity': cluster_stats['perplexity'],
                        'clustering/epoch_train_max_usage_fraction': cluster_stats['max_usage_fraction'],
                        'epoch': epoch + 1,
                    })
                    wandb.log({
                        'clustering/epoch_train_3d': wandb.Plotly(fig_3d),
                        'epoch': epoch + 1,
                    })

                # Save locally for offline viewing (VSCode / browser)
                if save_local_plots:
                    cluster_dir = save_dir / "clustering"
                    cluster_dir.mkdir(parents=True, exist_ok=True)
                    fig_3d.write_html(
                        cluster_dir / f"clustering_3d_epoch_{epoch+1}.html",
                        include_plotlyjs=True,
                    )
                    fig_pca.savefig(cluster_dir / f"clustering_2d_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
                    fig_usage.savefig(cluster_dir / f"clustering_usage_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')

                plt.close(fig_pca)
                plt.close(fig_usage)

                # Clean up to save memory
                del train_metrics['z_e_all']
                del train_metrics['indices_all']

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Results:")
        print(f"{'='*60}")
        print(f"Train - Loss: {train_metrics['loss']:.6f} | "
              f"Recon: {train_metrics['recon_loss']:.6f} | "
              f"VQ: {train_metrics['vq_loss']:.6f}")

        if val_metrics is not None:
            print(f"Val   - Loss: {val_metrics['loss']:.6f} | "
                  f"Recon: {val_metrics['recon_loss']:.6f} | "
                  f"VQ: {val_metrics['vq_loss']:.6f}")
            print(f"Codebook usage: {val_metrics['codebook_usage']*100:.2f}% "
                  f"({val_metrics['used_codes']}/{config['model']['quantizer']['num_embeddings']} codes)")

        # Dead code stats
        if model.quantizer.dead_code_enabled:
            usage_stats = model.quantizer.get_usage_stats()
            print(f"Active codes: {usage_stats['active_codes']}, "
                  f"Dead codes: {usage_stats['dead_codes']}, "
                  f"Perplexity: {usage_stats['perplexity']:.2f}")
            print(f"Max usage fraction: {usage_stats['max_usage_fraction']*100:.2f}%, "
                  f"Min nonzero usage: {usage_stats['min_nonzero_usage']:.2f}")
            print(f"Codebook norm - mean: {usage_stats['embedding_norm_mean']:.4f}, "
                  f"min: {usage_stats['embedding_norm_min']:.4f}, "
                  f"max: {usage_stats['embedding_norm_max']:.4f}")

        print(f"Codebook initialized: {model.quantizer.initialized.item()}")

        print(f"{'='*60}")

        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/total_loss': train_metrics['loss'],
                'train/recon_loss': train_metrics['recon_loss'],
                'train/vq_loss': train_metrics['vq_loss'],
                'train/codebook_loss': train_metrics['codebook_loss'],
                'train/commitment_loss': train_metrics['commitment_loss'],
            }

            if val_metrics is not None:
                log_dict['val/total_loss'] = val_metrics['loss']
                log_dict['val/recon_loss'] = val_metrics['recon_loss']
                log_dict['val/vq_loss'] = val_metrics['vq_loss']
                log_dict['val/codebook_loss'] = val_metrics['codebook_loss']
                log_dict['val/commitment_loss'] = val_metrics['commitment_loss']
                log_dict['val/codebook_usage'] = val_metrics['codebook_usage']
                log_dict['val/used_codes'] = val_metrics['used_codes']

            # Dead code stats
            if model.quantizer.dead_code_enabled:
                usage_stats = model.quantizer.get_usage_stats()
                log_dict['codebook/active_codes'] = usage_stats['active_codes']
                log_dict['codebook/dead_codes'] = usage_stats['dead_codes']
                log_dict['codebook/perplexity'] = usage_stats['perplexity']
                log_dict['codebook/max_usage_fraction'] = usage_stats['max_usage_fraction']
                log_dict['codebook/min_nonzero_usage'] = usage_stats['min_nonzero_usage']
                log_dict['codebook/embedding_norm_mean'] = usage_stats['embedding_norm_mean']
                log_dict['codebook/embedding_norm_min'] = usage_stats['embedding_norm_min']
                log_dict['codebook/embedding_norm_max'] = usage_stats['embedding_norm_max']

            log_dict['codebook/initialized'] = int(model.quantizer.initialized.item())

            import numpy as np

            if val_metrics is not None:
                usage_counts = val_metrics['usage_counts']
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(range(len(usage_counts)), usage_counts, alpha=0.6, s=20)
                ax.set_xlabel('Codebook Index')
                ax.set_ylabel('Usage Count')
                ax.set_title(f'Codebook Usage Distribution - Epoch {epoch+1}')
                ax.grid(True, alpha=0.3)

                log_dict['codebook_usage_plot'] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log_dict)

        if (epoch + 1) % save_every == 0:
            save_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, val_metrics, save_path)

        if val_metrics is not None and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = save_dir / "checkpoint_best.pth"
            save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
            print(f"✓ New best model saved (val_loss: {best_val_loss:.6f})")

    print("\n" + "="*60)
    print("Final evaluation on test set...")
    print("="*60)

    if test_loader is not None:
        test_metrics = validate(model, test_loader, device, split_name="Test")

        print(f"Test - Loss: {test_metrics['loss']:.6f} | "
              f"Recon: {test_metrics['recon_loss']:.6f} | "
              f"VQ: {test_metrics['vq_loss']:.6f}")
        print(f"Codebook usage: {test_metrics['codebook_usage']*100:.2f}% "
              f"({test_metrics['used_codes']}/{config['model']['quantizer']['num_embeddings']} codes)")

        if use_wandb:
            wandb.log({
                'test/total_loss': test_metrics['loss'],
                'test/recon_loss': test_metrics['recon_loss'],
                'test/vq_loss': test_metrics['vq_loss'],
                'test/codebook_usage': test_metrics['codebook_usage'],
                'test/used_codes': test_metrics['used_codes'],
            })
    else:
        print("⚠ No test set available")
        test_metrics = None

    final_path = save_dir / "checkpoint_final.pth"
    # Use val_metrics if test_metrics is None
    final_metrics = test_metrics if test_metrics is not None else val_metrics
    save_checkpoint(model, optimizer, num_epochs-1, final_metrics, final_path)

    if use_wandb:
        wandb.finish()

    print("\n✓ Training completed!")


if __name__ == "__main__":
    main()
