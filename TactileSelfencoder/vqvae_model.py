"""
Temporal VQ-VAE for tactile force history encoding.

Encodes tactile force history [B, T, D] into discrete tokens via vector quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, D]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class VectorQuantizer(nn.Module):
    """
    EMA Vector Quantizer with straight-through estimator and dead code replacement.

    Maps continuous embeddings to discrete codebook entries.
    Codebook is updated via EMA, not gradient descent.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5,
                 dead_code_config=None):
        """
        Args:
            num_embeddings: Size of the codebook (number of discrete tokens)
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss (beta)
            decay: EMA decay rate
            epsilon: Small constant for numerical stability
            dead_code_config: Dict with dead code replacement settings
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook: [num_embeddings, embedding_dim]
        embedding = torch.empty(num_embeddings, embedding_dim)
        embedding.normal_(0, 1.0 / embedding_dim)
        self.register_buffer('embedding', embedding)

        # EMA statistics
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embedding_sum', embedding.clone())

        # RMS Norm for quantized embeddings
        self.rms_norm = RMSNorm(embedding_dim)

        # Dead code replacement
        self.dead_code_config = dead_code_config or {}
        self.dead_code_enabled = self.dead_code_config.get('enabled', False)

        if self.dead_code_enabled:
            usage_decay = self.dead_code_config.get('usage_decay', 0.99)
            self.register_buffer('ema_usage', torch.zeros(num_embeddings))
            self.register_buffer('dead_counter', torch.zeros(num_embeddings, dtype=torch.long))
            self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
            self.usage_decay = usage_decay
            self.dead_threshold = self.dead_code_config.get('dead_threshold', 1.0)
            self.dead_patience = self.dead_code_config.get('dead_patience', 1000)
            self.check_interval = self.dead_code_config.get('check_interval', 100)
            self.reset_interval = self.dead_code_config.get('reset_interval', 500)

    def forward(self, z_e):
        """
        Args:
            z_e: Encoder output [B, embedding_dim]

        Returns:
            z_q: Quantized embedding [B, embedding_dim]
            indices: Token indices [B]
            vq_loss: VQ loss (only commitment loss)
            codebook_loss: Set to 0 (for compatibility)
            commitment_loss: Commitment loss
        """
        # Apply RMS Norm to encoder output BEFORE quantization
        z_e_normalized = self.rms_norm(z_e)

        # Compute L2 distances
        z_e_sq = torch.sum(z_e_normalized ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding ** 2, dim=1, keepdim=True).transpose(0, 1)
        distances = z_e_sq + e_sq - 2 * torch.matmul(z_e_normalized, self.embedding.t())

        # Find nearest codebook entry
        indices = torch.argmin(distances, dim=1)

        # Get quantized embeddings
        z_q = F.embedding(indices, self.embedding)

        # Update EMA statistics during training
        if self.training:
            encodings = F.one_hot(indices, self.num_embeddings).float()

            # Update cluster size
            self.ema_cluster_size.mul_(self.decay).add_(
                encodings.sum(dim=0), alpha=1 - self.decay
            )

            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size.add_(self.epsilon).div_(
                n + self.num_embeddings * self.epsilon
            ).mul_(n)

            # Update embedding sum
            dw = torch.matmul(encodings.t(), z_e_normalized)
            self.ema_embedding_sum.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Normalize to get updated codebook
            self.embedding.copy_(self.ema_embedding_sum / self.ema_cluster_size.unsqueeze(1))

            # Dead code replacement
            if self.dead_code_enabled:
                self._update_usage_statistics(encodings, z_e_normalized)

        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z_e_normalized, z_q.detach())

        codebook_loss = torch.tensor(0.0, device=z_e.device)
        vq_loss = commitment_loss

        # Straight-through estimator
        z_q_st = z_e_normalized + (z_q - z_e_normalized).detach()

        return z_q_st, indices, vq_loss, codebook_loss, commitment_loss

    def _update_usage_statistics(self, encodings, z_e_normalized):
        """Update usage statistics and check for dead codes."""
        # Update EMA usage
        batch_usage = encodings.sum(dim=0)
        self.ema_usage.mul_(self.usage_decay).add_(batch_usage, alpha=1 - self.usage_decay)

        self.global_step += 1

        # Check dead codes periodically
        if self.global_step % self.check_interval == 0:
            self._check_and_reset_dead_codes(z_e_normalized)

    def _check_and_reset_dead_codes(self, z_e_normalized):
        """Check for dead codes and reset them."""
        # Update dead counter
        dead_mask = self.ema_usage < self.dead_threshold
        self.dead_counter[dead_mask] += self.check_interval
        self.dead_counter[~dead_mask] = 0

        # Reset dead codes periodically
        if self.global_step % self.reset_interval == 0:
            really_dead = self.dead_counter > self.dead_patience

            if really_dead.any():
                num_dead = really_dead.sum().item()

                # Randomly select encoder outputs to replace dead codes
                batch_size = z_e_normalized.shape[0]
                for i in range(self.num_embeddings):
                    if really_dead[i]:
                        # Random replacement
                        random_idx = torch.randint(0, batch_size, (1,)).item()
                        self.embedding[i] = z_e_normalized[random_idx].detach()

                # Reset statistics for dead codes
                avg_usage = self.ema_usage[~really_dead].mean()
                self.ema_usage[really_dead] = avg_usage
                self.dead_counter[really_dead] = 0

    def get_usage_stats(self):
        """Get codebook usage statistics."""
        if not self.dead_code_enabled:
            return {}

        active_codes = (self.ema_usage > self.dead_threshold).sum().item()
        dead_codes = self.num_embeddings - active_codes

        # Compute perplexity
        usage_probs = self.ema_usage / (self.ema_usage.sum() + 1e-10)
        perplexity = torch.exp(-torch.sum(usage_probs * torch.log(usage_probs + 1e-10)))

        return {
            'active_codes': active_codes,
            'dead_codes': dead_codes,
            'perplexity': perplexity.item(),
            'usage': self.ema_usage.cpu().numpy(),
        }


class TemporalEncoder(nn.Module):
    """
    1D CNN encoder for temporal force history.
    """

    def __init__(self, config):
        super().__init__()

        input_dim = config['input']['force_dim']
        enc_cfg = config['encoder']

        self.conv1 = nn.Conv1d(
            input_dim,
            enc_cfg['conv1_out'],
            kernel_size=enc_cfg['conv1_kernel'],
            padding=enc_cfg['conv1_kernel'] // 2
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            enc_cfg['conv2_in'],
            enc_cfg['conv2_out'],
            kernel_size=enc_cfg['conv2_kernel'],
            padding=enc_cfg['conv2_kernel'] // 2
        )
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(enc_cfg['conv2_out'], enc_cfg['fc_out'])

    def forward(self, x):
        """
        Args:
            x: [B, T, D] force history

        Returns:
            z_e: [B, embedding_dim] continuous embedding
        """
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        z_e = self.fc(x)
        return z_e


class TemporalDecoder(nn.Module):
    """
    MLP decoder for reconstructing force history.
    """

    def __init__(self, config):
        super().__init__()

        dec_cfg = config['decoder']
        history_steps = config['input']['history_steps']
        force_dim = config['input']['force_dim']

        self.history_steps = history_steps
        self.force_dim = force_dim

        self.fc1 = nn.Linear(dec_cfg['fc1_in'], dec_cfg['fc1_out'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dec_cfg['fc1_out'], dec_cfg['fc2_out'])

    def forward(self, z_q):
        """
        Args:
            z_q: [B, embedding_dim] quantized embedding

        Returns:
            x_recon: [B, T, D] reconstructed force history
        """
        x = self.fc1(z_q)
        x = self.relu(x)
        x = self.fc2(x)
        x_recon = x.view(-1, self.history_steps, self.force_dim)
        return x_recon


class TemporalVQVAE(nn.Module):
    """
    Complete Temporal VQ-VAE model for tactile force history.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.encoder = TemporalEncoder(config)

        dead_code_config = config.get('dead_code_replacement', {})
        self.quantizer = VectorQuantizer(
            num_embeddings=config['quantizer']['num_embeddings'],
            embedding_dim=config['quantizer']['embedding_dim'],
            commitment_cost=config['quantizer']['commitment_cost'],
            decay=config['quantizer'].get('ema_decay', 0.99),
            epsilon=config['quantizer'].get('ema_epsilon', 1e-5),
            dead_code_config=dead_code_config
        )

        self.decoder = TemporalDecoder(config)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] force history tensor

        Returns:
            Dictionary with all outputs and losses
        """
        z_e = self.encoder(x)
        z_q, indices, vq_loss, codebook_loss, commitment_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)

        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss

        return {
            'z_e': z_e,
            'z_q': z_q,
            'indices': indices,
            'x_recon': x_recon,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'total_loss': total_loss,
        }

    def encode(self, x):
        """Encode to discrete tokens."""
        z_e = self.encoder(x)
        z_q, indices, _, _, _ = self.quantizer(z_e)
        return indices, z_q

    def decode(self, z_q):
        """Decode from quantized embeddings."""
        return self.decoder(z_q)

    def decode_from_indices(self, indices):
        """Decode from token indices."""
        z_q = F.embedding(indices, self.quantizer.embedding)
        return self.decoder(z_q)


def build_vqvae_from_config(config_path):
    """Build VQ-VAE model from YAML config file."""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = TemporalVQVAE(config['model'])

    return model, config


if __name__ == "__main__":
    from pathlib import Path

    config_path = Path(__file__).parent / "vqvae_config.yaml"

    if config_path.exists():
        model, config = build_vqvae_from_config(config_path)

        B, T, D = 4, 15, 12
        x = torch.randn(B, T, D)

        output = model(x)

        print("Model test successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output['x_recon'].shape}")
        print(f"Token indices: {output['indices']}")
        print(f"Total loss: {output['total_loss'].item():.6f}")
