"""
T-Rex style per-finger tactile force VQ-VAE.

Based on T-Rex paper's description:
- Per-finger 6D force/torque encoding
- Shared encoder/decoder/codebook across fingers
- Finger identity embedding
- EMA-based vector quantization
- Magnitude-weighted reconstruction loss

Adapted for 2 fingers instead of T-Rex's 5 fingers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PerFingerForceEncoder(nn.Module):
    """
    T-Rex style per-finger force encoder with finger identity embedding.

    Processes each finger independently with shared conv weights,
    but adds finger-specific identity embedding.
    """

    def __init__(self, config):
        super().__init__()

        self.num_fingers = config['num_fingers']
        self.force_dim = config['force_dim']
        enc_cfg = config['encoder']

        # Finger identity embeddings
        self.finger_embedding = nn.Embedding(
            self.num_fingers,
            enc_cfg['finger_embed_dim']
        )

        # Conv1: Initial feature extraction
        # Input: [B*num_fingers, 6, 16]
        self.conv1 = nn.Conv1d(
            enc_cfg['conv1_in_channels'],
            enc_cfg['conv1_out_channels'],
            kernel_size=enc_cfg['conv1_kernel_size'],
            stride=enc_cfg['conv1_stride'],
            padding=enc_cfg['conv1_padding']
        )
        if enc_cfg.get('use_group_norm', True):
            self.norm1 = nn.GroupNorm(enc_cfg['num_groups'], enc_cfg['conv1_out_channels'])
        else:
            self.norm1 = nn.Identity()

        # Conv2: First downsampling (16 -> 8)
        self.conv2 = nn.Conv1d(
            enc_cfg['conv2_in_channels'],
            enc_cfg['conv2_out_channels'],
            kernel_size=enc_cfg['conv2_kernel_size'],
            stride=enc_cfg['conv2_stride'],
            padding=enc_cfg['conv2_padding']
        )
        if enc_cfg.get('use_group_norm', True):
            self.norm2 = nn.GroupNorm(enc_cfg['num_groups'], enc_cfg['conv2_out_channels'])
        else:
            self.norm2 = nn.Identity()

        # Conv3: Second downsampling (8 -> 4)
        self.conv3 = nn.Conv1d(
            enc_cfg['conv3_in_channels'],
            enc_cfg['conv3_out_channels'],
            kernel_size=enc_cfg['conv3_kernel_size'],
            stride=enc_cfg['conv3_stride'],
            padding=enc_cfg['conv3_padding']
        )
        if enc_cfg.get('use_group_norm', True):
            self.norm3 = nn.GroupNorm(enc_cfg['num_groups'], enc_cfg['conv3_out_channels'])
        else:
            self.norm3 = nn.Identity()

        # Conv4: Final conv before pooling
        self.conv4 = nn.Conv1d(
            enc_cfg['conv4_in_channels'],
            enc_cfg['conv4_out_channels'],
            kernel_size=enc_cfg['conv4_kernel_size'],
            stride=enc_cfg['conv4_stride'],
            padding=enc_cfg['conv4_padding']
        )
        if enc_cfg.get('use_group_norm', True):
            self.norm4 = nn.GroupNorm(enc_cfg['num_groups'], enc_cfg['conv4_out_channels'])
        else:
            self.norm4 = nn.Identity()

        # Activation
        if enc_cfg.get('activation', 'gelu') == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

        # Temporal pooling
        if enc_cfg.get('use_avg_pool', True):
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)

        # Projection layer to incorporate finger embedding
        self.finger_proj = nn.Linear(
            enc_cfg['finger_embed_dim'],
            enc_cfg['conv4_out_channels']
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, num_fingers, force_dim] force history
               where T=history_steps, num_fingers=2, force_dim=6

        Returns:
            z_e: [B, num_fingers, embedding_dim] continuous embeddings
        """
        B, T, F, D = x.shape
        assert F == self.num_fingers
        assert D == self.force_dim

        # Reshape to [B*num_fingers, force_dim, T]
        x = x.permute(0, 2, 3, 1)  # [B, num_fingers, force_dim, T]
        x = x.reshape(B * F, D, T)  # [B*F, D, T]

        # Conv layers
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act(x)

        # Temporal pooling: [B*F, C, T'] -> [B*F, C]
        x = self.pool(x).squeeze(-1)

        # Add finger identity embedding
        # Create finger indices: [0, 1, 0, 1, ...] for each batch
        finger_ids = torch.arange(F, device=x.device).repeat(B)  # [B*F]
        finger_emb = self.finger_embedding(finger_ids)  # [B*F, finger_embed_dim]
        finger_emb = self.finger_proj(finger_emb)  # [B*F, C]

        # Add finger embedding to features
        z_e = x + finger_emb  # [B*F, embedding_dim]

        # Reshape back to [B, num_fingers, embedding_dim]
        z_e = z_e.reshape(B, F, -1)

        return z_e


class PerFingerForceDecoder(nn.Module):
    """
    T-Rex style per-finger force decoder.

    Reconstructs force history from quantized embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.num_fingers = config['num_fingers']
        self.force_dim = config['force_dim']
        self.history_steps = config['history_steps']

        dec_cfg = config['decoder']

        # MLP decoder
        self.fc1 = nn.Linear(dec_cfg['fc1_in'], dec_cfg['fc1_out'])
        if dec_cfg.get('use_group_norm', True):
            self.norm1 = nn.GroupNorm(dec_cfg['num_groups'], dec_cfg['fc1_out'])
        else:
            self.norm1 = nn.Identity()

        self.fc2 = nn.Linear(dec_cfg['fc1_out'], dec_cfg['fc2_out'])
        if dec_cfg.get('use_group_norm', True):
            self.norm2 = nn.GroupNorm(dec_cfg['num_groups'], dec_cfg['fc2_out'])
        else:
            self.norm2 = nn.Identity()

        self.fc3 = nn.Linear(dec_cfg['fc2_out'], dec_cfg['output_dim'])

        # Activation
        if dec_cfg.get('activation', 'gelu') == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

    def forward(self, z_q):
        """
        Args:
            z_q: [B, num_fingers, embedding_dim] quantized embeddings

        Returns:
            x_recon: [B, T, num_fingers, force_dim] reconstructed force history
        """
        B, F, D = z_q.shape

        # Flatten to [B*F, D]
        z_q = z_q.reshape(B * F, D)

        # MLP decoder
        x = self.fc1(z_q)
        x = self.norm1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.fc3(x)  # [B*F, T*force_dim]

        # Reshape to [B, num_fingers, T, force_dim]
        x = x.reshape(B, F, self.history_steps, self.force_dim)

        # Permute to [B, T, num_fingers, force_dim]
        x_recon = x.permute(0, 2, 1, 3)

        return x_recon


class TRexVectorQuantizer(nn.Module):
    """
    T-Rex style EMA Vector Quantizer.

    Key features:
    - EMA-based codebook updates (no gradient descent on codebook)
    - Dead code replacement
    - Commitment loss
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5, dead_code_config=None, init_config=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialization config
        self.init_config = init_config or {}
        self.init_strategy = self.init_config.get('init_strategy', 'first_batch')
        self.kmeans_init_samples = self.init_config.get('kmeans_init_samples', 4096)
        self.kmeans_iterations = self.init_config.get('kmeans_iterations', 20)

        # Codebook
        embedding = torch.randn(num_embeddings, embedding_dim)
        if self.init_strategy == 'random':
            embedding = F.normalize(embedding, p=2, dim=1) * (embedding_dim ** 0.5)
        else:
            embedding.zero_()

        self.register_buffer('embedding', embedding)

        # EMA statistics
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embedding_sum', torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer('initialized', torch.tensor(self.init_strategy == 'random', dtype=torch.bool))

        # K-means initialization
        if self.init_strategy == 'kmeans':
            self.kmeans_samples = []

        # Dead code replacement
        self.dead_code_config = dead_code_config or {}
        self.dead_code_enabled = self.dead_code_config.get('enabled', False)

        if self.dead_code_enabled:
            self.register_buffer('ema_usage', torch.zeros(num_embeddings))
            self.register_buffer('dead_counter', torch.zeros(num_embeddings, dtype=torch.long))
            self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
            self.usage_decay = self.dead_code_config.get('usage_decay', 0.99)
            self.dead_threshold = self.dead_code_config.get('dead_threshold', 0.1)
            self.dead_patience = self.dead_code_config.get('dead_patience', 1000)
            self.check_interval = self.dead_code_config.get('check_interval', 100)
            self.reset_interval = self.dead_code_config.get('reset_interval', 500)
            self.reset_count = self.dead_code_config.get('reset_count', 1.0)
            self.reset_noise_scale = self.dead_code_config.get('reset_noise_scale', 1e-4)

    @torch.no_grad()
    def _initialize_codebook(self, z_e):
        """Initialize from first batch."""
        # Flatten: [B, F, D] -> [B*F, D]
        z_e_flat = z_e.reshape(-1, self.embedding_dim)
        N = z_e_flat.shape[0]
        K = self.num_embeddings

        if N >= K:
            indices = torch.randperm(N, device=z_e.device)[:K]
            selected = z_e_flat[indices]
        else:
            repeats = (K + N - 1) // N
            selected = z_e_flat.repeat(repeats, 1)[:K]
            selected = selected + torch.randn_like(selected) * 1e-5

        self.embedding.copy_(selected)
        self.ema_cluster_size.fill_(1.0)
        self.ema_embedding_sum.copy_(self.embedding)

        if self.dead_code_enabled:
            self.ema_usage.fill_(1.0)
            self.dead_counter.zero_()

        self.initialized.fill_(True)

    @torch.no_grad()
    def _initialize_codebook_kmeans(self):
        """Initialize with k-means."""
        all_samples = torch.cat(self.kmeans_samples, dim=0)
        K = self.num_embeddings
        device = all_samples.device

        # K-means++ initialization
        centers = torch.zeros(K, self.embedding_dim, device=device)
        centers[0] = all_samples[torch.randint(0, all_samples.shape[0], (1,), device=device)]

        for k in range(1, K):
            dists = torch.cdist(all_samples, centers[:k])
            min_dists = dists.min(dim=1)[0]
            probs = min_dists ** 2
            probs = probs / (probs.sum() + 1e-10)
            centers[k] = all_samples[torch.multinomial(probs, 1)]

        # K-means iterations
        for _ in range(self.kmeans_iterations):
            dists = torch.cdist(all_samples, centers)
            assignments = dists.argmin(dim=1)

            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centers[k] = all_samples[mask].mean(dim=0)
                else:
                    dists_to_centers = torch.cdist(all_samples, centers)
                    min_dists = dists_to_centers.min(dim=1)[0]
                    farthest_idx = min_dists.argmax()
                    centers[k] = all_samples[farthest_idx]

        # Final assignment
        dists = torch.cdist(all_samples, centers)
        assignments = dists.argmin(dim=1)
        cluster_counts = torch.bincount(assignments, minlength=K).float()
        cluster_counts[cluster_counts == 0] = 1.0

        self.embedding.copy_(centers)
        self.ema_cluster_size.copy_(cluster_counts)
        self.ema_embedding_sum.copy_(centers * cluster_counts.unsqueeze(1))

        if self.dead_code_enabled:
            self.ema_usage.copy_(cluster_counts)
            self.dead_counter.zero_()

        self.initialized.fill_(True)
        self.kmeans_samples = []

    def forward(self, z_e, update_ema=None):
        """
        Args:
            z_e: [B, num_fingers, embedding_dim]
            update_ema: Whether to update EMA (defaults to self.training)

        Returns:
            z_q: [B, num_fingers, embedding_dim] quantized
            indices: [B, num_fingers] discrete tokens
            vq_loss: VQ loss
            commitment_loss: Commitment loss
        """
        if update_ema is None:
            update_ema = self.training

        B, F, D = z_e.shape

        # Flatten for quantization
        z_e_flat = z_e.reshape(-1, D)  # [B*F, D]

        # Initialize if needed
        if update_ema and not self.initialized:
            if self.init_strategy == 'first_batch':
                self._initialize_codebook(z_e.detach())
            elif self.init_strategy == 'kmeans':
                self.kmeans_samples.append(z_e_flat.detach().cpu())
                total_samples = sum(s.shape[0] for s in self.kmeans_samples)

                if total_samples >= self.kmeans_init_samples:
                    self.kmeans_samples = [s.to(z_e.device) for s in self.kmeans_samples]
                    self._initialize_codebook_kmeans()

        # Skip if not initialized (during k-means collection)
        if self.init_strategy == 'kmeans' and not self.initialized:
            z_q = z_e
            indices = torch.zeros(B, F, dtype=torch.long, device=z_e.device)
            vq_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = vq_loss
            return z_q, indices, vq_loss, commitment_loss

        # Compute distances
        z_e_sq = (z_e_flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.embedding ** 2).sum(dim=1, keepdim=True).t()
        distances = z_e_sq + e_sq - 2 * torch.matmul(z_e_flat, self.embedding.t())

        # Find nearest
        indices_flat = distances.argmin(dim=1)
        z_q_flat = F.embedding(indices_flat, self.embedding)

        # Update EMA
        if update_ema:
            with torch.no_grad():
                encodings = F.one_hot(indices_flat, self.num_embeddings).float()

                batch_cluster_size = encodings.sum(dim=0)
                self.ema_cluster_size.mul_(self.decay).add_(
                    batch_cluster_size, alpha=1 - self.decay
                )

                batch_embedding_sum = torch.matmul(encodings.t(), z_e_flat)
                self.ema_embedding_sum.mul_(self.decay).add_(
                    batch_embedding_sum, alpha=1 - self.decay
                )

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                smoothed_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )

                self.embedding.copy_(self.ema_embedding_sum / smoothed_cluster_size.unsqueeze(1))

                # Dead code handling
                if self.dead_code_enabled:
                    self._update_usage(encodings, z_e_flat)

        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z_e_flat, z_q_flat.detach())
        vq_loss = commitment_loss

        # Straight-through
        z_q_flat = z_e_flat + (z_q_flat - z_e_flat).detach()

        # Reshape back
        z_q = z_q_flat.reshape(B, F, D)
        indices = indices_flat.reshape(B, F)

        return z_q, indices, vq_loss, commitment_loss

    def _update_usage(self, encodings, z_e_flat):
        """Update usage statistics and handle dead codes."""
        batch_usage = encodings.sum(dim=0)
        self.ema_usage.mul_(self.usage_decay).add_(batch_usage, alpha=1 - self.usage_decay)
        self.global_step += 1

        if self.global_step % self.check_interval == 0:
            self._check_and_reset_dead_codes(z_e_flat)

    @torch.no_grad()
    def _check_and_reset_dead_codes(self, z_e_flat):
        """Check and reset dead codes."""
        dead_mask = self.ema_usage < self.dead_threshold
        self.dead_counter[dead_mask] += self.check_interval
        self.dead_counter[~dead_mask] = 0

        if self.global_step % self.reset_interval == 0:
            really_dead = self.dead_counter > self.dead_patience

            if really_dead.any():
                num_dead = really_dead.sum().item()
                dead_indices = torch.where(really_dead)[0]

                # Reset with farthest samples
                for i, dead_idx in enumerate(dead_indices):
                    if i < z_e_flat.shape[0]:
                        replacement = z_e_flat[i].clone()
                    else:
                        idx = i % z_e_flat.shape[0]
                        replacement = z_e_flat[idx].clone()
                        replacement = replacement + torch.randn_like(replacement) * self.reset_noise_scale

                    self.embedding[dead_idx] = replacement
                    self.ema_cluster_size[dead_idx] = self.reset_count
                    self.ema_embedding_sum[dead_idx] = replacement * self.reset_count
                    self.ema_usage[dead_idx] = self.reset_count
                    self.dead_counter[dead_idx] = 0


class TRexTactileVQVAE(nn.Module):
    """
    Complete T-Rex style per-finger tactile VQ-VAE.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_fingers = config['num_fingers']

        self.encoder = PerFingerForceEncoder(config)

        self.quantizer = TRexVectorQuantizer(
            num_embeddings=config['quantizer']['num_embeddings'],
            embedding_dim=config['quantizer']['embedding_dim'],
            commitment_cost=config['quantizer']['commitment_cost'],
            decay=config['quantizer'].get('ema_decay', 0.99),
            epsilon=config['quantizer'].get('ema_epsilon', 1e-5),
            dead_code_config=config.get('dead_code_replacement', {}),
            init_config=config['quantizer']
        )

        self.decoder = PerFingerForceDecoder(config)

        # Magnitude-weighted reconstruction loss config
        self.use_magnitude_weighted = config.get('loss', {}).get('use_magnitude_weighted_recon', True)
        self.magnitude_eps = config.get('loss', {}).get('magnitude_weight_eps', 1e-6)

    def forward(self, x):
        """
        Args:
            x: [B, T, num_fingers, force_dim] force history

        Returns:
            dict with outputs and losses
        """
        # Encode
        z_e = self.encoder(x)  # [B, F, D]

        # Quantize
        z_q, indices, vq_loss, commitment_loss = self.quantizer(z_e, update_ema=self.training)

        # Decode
        x_recon = self.decoder(z_q)  # [B, T, F, force_dim]

        # Reconstruction loss
        if self.use_magnitude_weighted:
            # Magnitude-weighted MSE
            magnitude = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True) + self.magnitude_eps)  # [B, T, F, 1]
            weights = magnitude / (magnitude.mean() + self.magnitude_eps)
            recon_loss = ((x - x_recon) ** 2 * weights).mean()
        else:
            recon_loss = F.mse_loss(x_recon, x)

        total_loss = recon_loss + vq_loss

        return {
            'z_e': z_e,
            'z_q': z_q,
            'indices': indices,
            'x_recon': x_recon,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'total_loss': total_loss,
        }

    def encode(self, x):
        """Encode to discrete tokens (read-only)."""
        if not self.quantizer.initialized:
            raise RuntimeError("Quantizer not initialized")

        z_e = self.encoder(x)
        z_q, indices, _, _ = self.quantizer(z_e, update_ema=False)
        return indices, z_q

    def decode(self, z_q):
        """Decode from quantized embeddings."""
        return self.decoder(z_q)

    def decode_from_indices(self, indices):
        """Decode from token indices."""
        B, F = indices.shape
        indices_flat = indices.reshape(-1)
        z_q_flat = F.embedding(indices_flat, self.quantizer.embedding)
        z_q = z_q_flat.reshape(B, F, -1)
        return self.decoder(z_q)


def build_trex_vqvae_from_config(config_path):
    """Build T-Rex VQ-VAE from config file."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = TRexTactileVQVAE(config['model'])

    return model, config


if __name__ == "__main__":
    from pathlib import Path

    config_path = Path(__file__).parent / "trex_vqvae_config.yaml"

    if config_path.exists():
        model, config = build_trex_vqvae_from_config(config_path)

        # Test with [B, T, F, D] = [4, 16, 2, 6]
        B, T, F, D = 4, 16, 2, 6
        x = torch.randn(B, T, F, D)

        output = model(x)

        print("T-Rex VQ-VAE test successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output['x_recon'].shape}")
        print(f"Indices shape: {output['indices'].shape}")
        print(f"Indices: {output['indices']}")
        print(f"Total loss: {output['total_loss'].item():.6f}")
