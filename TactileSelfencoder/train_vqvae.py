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

        train_loader = dataloaders_dict['train']
        val_loader = dataloaders_dict['val']
        test_loader = dataloaders_dict['test']

        tactile_channel_names = dataloader_config['dataset']['keys'].get(
            'tactile_force_channel_order', []
        )
        print(f"Tactile channels: {tactile_channel_names}")

        return train_loader, val_loader, test_loader

    finally:
        os.chdir(original_dir)


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

        output = model(x)

        if batch_idx == 0 and epoch == 1:
            with torch.no_grad():
                z_e_raw = model.encoder(x)
                print(f"\nEncoder output (before RMS Norm):")
                print(f"  z_e shape: {z_e_raw.shape}")
                print(f"  z_e mean: {z_e_raw.mean():.6f}, std: {z_e_raw.std():.6f}")
                print(f"  z_e norm (avg): {z_e_raw.norm(dim=1).mean():.6f}")

                z_e_norm = model.quantizer.rms_norm(z_e_raw)
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
            print(f"{'='*70}\n")

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

        if use_wandb and (batch_idx + 1) % wandb_log_every == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_idx + 1
            wandb.log({
                'train_batch/total_loss': loss.item(),
                'train_batch/recon_loss': output['recon_loss'].item(),
                'train_batch/vq_loss': output['vq_loss'].item(),
                'train_batch/codebook_loss': output['codebook_loss'].item(),
                'train_batch/commitment_loss': output['commitment_loss'].item(),
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

    return metrics


def validate(model, dataloader, device, split_name="Val"):
    """Validate the model."""
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_codebook_loss = 0
    total_commitment_loss = 0

    all_indices = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[{split_name}]")

        for batch in pbar:
            x = batch["tactile_history"].to(device)

            history_steps = model.config['input']['history_steps']
            if x.shape[1] != history_steps:
                x = x[:, :history_steps, :]

            output = model(x)

            total_loss += output['total_loss'].item()
            total_recon_loss += output['recon_loss'].item()
            total_vq_loss += output['vq_loss'].item()
            total_codebook_loss += output['codebook_loss'].item()
            total_commitment_loss += output['commitment_loss'].item()

            all_indices.append(output['indices'].cpu())

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

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': model.config,
    }, save_path)

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
    if config['model'].get('dead_code_replacement', {}).get('enabled', False):
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
    print(f"Val batches: {len(val_loader)}")
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
        val_metrics = validate(model, val_loader, device, split_name="Val")

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Results:")
        print(f"{'='*60}")
        print(f"Train - Loss: {train_metrics['loss']:.6f} | "
              f"Recon: {train_metrics['recon_loss']:.6f} | "
              f"VQ: {train_metrics['vq_loss']:.6f}")
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

        print(f"{'='*60}")

        if use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/total_loss': train_metrics['loss'],
                'train/recon_loss': train_metrics['recon_loss'],
                'train/vq_loss': train_metrics['vq_loss'],
                'train/codebook_loss': train_metrics['codebook_loss'],
                'train/commitment_loss': train_metrics['commitment_loss'],
                'val/total_loss': val_metrics['loss'],
                'val/recon_loss': val_metrics['recon_loss'],
                'val/vq_loss': val_metrics['vq_loss'],
                'val/codebook_loss': val_metrics['codebook_loss'],
                'val/commitment_loss': val_metrics['commitment_loss'],
                'val/codebook_usage': val_metrics['codebook_usage'],
                'val/used_codes': val_metrics['used_codes'],
            }

            # Dead code stats
            if model.quantizer.dead_code_enabled:
                usage_stats = model.quantizer.get_usage_stats()
                log_dict['val/active_codes'] = usage_stats['active_codes']
                log_dict['val/dead_codes'] = usage_stats['dead_codes']
                log_dict['val/perplexity'] = usage_stats['perplexity']

            import numpy as np

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

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = save_dir / "checkpoint_best.pth"
            save_checkpoint(model, optimizer, epoch, val_metrics, save_path)
            print(f"✓ New best model saved (val_loss: {best_val_loss:.6f})")

    print("\n" + "="*60)
    print("Final evaluation on test set...")
    print("="*60)
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

    final_path = save_dir / "checkpoint_final.pth"
    save_checkpoint(model, optimizer, num_epochs-1, test_metrics, final_path)

    if use_wandb:
        wandb.finish()

    print("\n✓ Training completed!")


if __name__ == "__main__":
    main()
