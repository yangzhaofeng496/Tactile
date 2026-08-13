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
                 dead_code_config=None, init_config=None):
        """
        Args:
            num_embeddings: Size of the codebook (number of discrete tokens)
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss (beta)
            decay: EMA decay rate
            epsilon: Small constant for numerical stability
            dead_code_config: Dict with dead code replacement settings
            init_config: Dict with initialization settings
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialization config
        self.init_config = init_config or {}
        self.init_strategy = self.init_config.get('init_strategy', 'first_batch')
        self.init_eps = self.init_config.get('init_eps', 1e-5)
        self.kmeans_init_samples = self.init_config.get('kmeans_init_samples', 4096)
        self.kmeans_iterations = self.init_config.get('kmeans_iterations', 20)

        # Codebook: [num_embeddings, embedding_dim]
        embedding = torch.empty(num_embeddings, embedding_dim)

        # Initialize based on strategy
        if self.init_strategy == 'random':
            # Scale-aware random initialization (not 1/embedding_dim!)
            # Use unit norm initialization for RMSNorm compatibility
            embedding.normal_(0, 1.0)
            embedding = F.normalize(embedding, p=2, dim=1) * (embedding_dim ** 0.5)
        else:
            # For first_batch and kmeans, initialize to zeros (will be set later)
            embedding.zero_()

        self.register_buffer('embedding', embedding)

        # EMA statistics
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embedding_sum', torch.zeros(num_embeddings, embedding_dim))

        # Initialization state
        self.register_buffer('initialized', torch.tensor(self.init_strategy == 'random', dtype=torch.bool))

        # For kmeans: collected samples buffer
        if self.init_strategy == 'kmeans':
            self.kmeans_samples = []

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
            self.reset_count = self.dead_code_config.get('reset_count', 1.0)
            self.reset_noise_scale = self.dead_code_config.get('reset_noise_scale', 1e-4)

    @torch.no_grad()
    def _initialize_codebook(self, z_e_normalized):
        """Initialize codebook from first batch of encoder outputs.

        Args:
            z_e_normalized: [B, embedding_dim] normalized encoder output
        """
        B = z_e_normalized.shape[0]
        K = self.num_embeddings

        if B >= K:
            # Sample K distinct indices
            indices = torch.randperm(B, device=z_e_normalized.device)[:K]
            selected = z_e_normalized[indices]
        else:
            # B < K: repeat with small noise
            repeats = (K + B - 1) // B  # ceil division
            repeated = z_e_normalized.repeat(repeats, 1)[:K]

            # Add small noise to avoid identical codes
            noise = torch.randn_like(repeated) * self.init_eps
            selected = repeated + noise

        # Initialize codebook
        self.embedding.copy_(selected)

        # Initialize EMA statistics
        initial_cluster_size = 1.0
        self.ema_cluster_size.fill_(initial_cluster_size)
        self.ema_embedding_sum.copy_(self.embedding * initial_cluster_size)

        # Initialize dead code tracking
        if self.dead_code_enabled:
            self.ema_usage.fill_(initial_cluster_size)
            self.dead_counter.zero_()

        self.initialized.fill_(True)

    @torch.no_grad()
    def _initialize_codebook_kmeans(self):
        """Initialize codebook using k-means on collected samples."""
        # Concatenate all collected samples
        all_samples = torch.cat(self.kmeans_samples, dim=0)  # [N, D]
        K = self.num_embeddings
        device = all_samples.device

        # Initialize centers: k-means++ style
        centers = torch.zeros(K, self.embedding_dim, device=device)

        # First center: random sample
        centers[0] = all_samples[torch.randint(0, all_samples.shape[0], (1,), device=device)]

        # Remaining centers: k-means++
        for k in range(1, K):
            # Compute distances to nearest existing center
            dists = torch.cdist(all_samples, centers[:k])  # [N, k]
            min_dists = dists.min(dim=1)[0]  # [N]

            # Sample proportional to squared distance
            probs = min_dists ** 2
            probs = probs / (probs.sum() + 1e-10)
            centers[k] = all_samples[torch.multinomial(probs, 1)]

        # Run k-means iterations
        for _ in range(self.kmeans_iterations):
            # Assign to nearest center
            dists = torch.cdist(all_samples, centers)  # [N, K]
            assignments = dists.argmin(dim=1)  # [N]

            # Update centers
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centers[k] = all_samples[mask].mean(dim=0)
                else:
                    # Empty cluster: reinitialize with farthest sample
                    dists_to_centers = torch.cdist(all_samples, centers)
                    min_dists = dists_to_centers.min(dim=1)[0]
                    farthest_idx = min_dists.argmax()
                    centers[k] = all_samples[farthest_idx]

        # Final assignment to get cluster sizes
        dists = torch.cdist(all_samples, centers)
        assignments = dists.argmin(dim=1)
        cluster_counts = torch.bincount(assignments, minlength=K).float()

        # Handle empty clusters
        empty_mask = cluster_counts == 0
        if empty_mask.any():
            cluster_counts[empty_mask] = 1.0

        # Initialize codebook and EMA
        self.embedding.copy_(centers)
        self.ema_cluster_size.copy_(cluster_counts)
        self.ema_embedding_sum.copy_(centers * cluster_counts.unsqueeze(1))

        if self.dead_code_enabled:
            self.ema_usage.copy_(cluster_counts)
            self.dead_counter.zero_()

        self.initialized.fill_(True)

        # Clear collected samples to free memory
        self.kmeans_samples = []

    def forward(self, z_e, update_ema=None):
        """
        Args:
            z_e: Encoder output [B, embedding_dim]
            update_ema: Optional override. If None, defaults to self.training.
                When True, codebook init / EMA / usage / dead-code reset run.
                When False, the quantizer is strictly read-only (no buffer is
                modified). Used by encode()/PCA so that inference never leaks
                EMA updates even if the model is still in train mode.

        Returns:
            z_q: Quantized embedding [B, embedding_dim]
            indices: Token indices [B]
            vq_loss: VQ loss (only commitment loss)
            codebook_loss: Set to 0 (for compatibility)
            commitment_loss: Commitment loss
            z_e_normalized: RMS-normalized encoder output [B, embedding_dim]
                — the exact tensor used for nearest-neighbor distance, so PCA
                scatter points and codebook centers live in the same space.
        """
        if update_ema is None:
            update_ema = self.training

        # Apply RMS Norm to encoder output BEFORE quantization
        z_e_normalized = self.rms_norm(z_e)

        # Initialize codebook if needed (only when updates are allowed)
        if update_ema and not self.initialized:
            if self.init_strategy == 'first_batch':
                self._initialize_codebook(z_e_normalized.detach())
            elif self.init_strategy == 'kmeans':
                # Collect samples
                self.kmeans_samples.append(z_e_normalized.detach().cpu())
                total_samples = sum(s.shape[0] for s in self.kmeans_samples)

                if total_samples >= self.kmeans_init_samples:
                    # Move samples to device and initialize
                    self.kmeans_samples = [s.to(z_e.device) for s in self.kmeans_samples]
                    self._initialize_codebook_kmeans()

        # If kmeans and not yet initialized, skip quantization
        if self.init_strategy == 'kmeans' and not self.initialized:
            # Return dummy values during collection phase
            z_q = z_e_normalized
            indices = torch.zeros(z_e.shape[0], dtype=torch.long, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            vq_loss = commitment_loss
            z_q_st = z_e_normalized
            return z_q_st, indices, vq_loss, codebook_loss, commitment_loss, z_e_normalized

        # Compute L2 distances
        z_e_sq = torch.sum(z_e_normalized ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding ** 2, dim=1, keepdim=True).transpose(0, 1)
        distances = z_e_sq + e_sq - 2 * torch.matmul(z_e_normalized, self.embedding.t())

        # Find nearest codebook entry
        indices = torch.argmin(distances, dim=1)

        # Get quantized embeddings
        z_q = F.embedding(indices, self.embedding)

        # Update EMA statistics only when explicitly allowed
        if update_ema:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.num_embeddings).float()

                # Update cluster size (preserve original for EMA)
                batch_cluster_size = encodings.sum(dim=0)
                self.ema_cluster_size.mul_(self.decay).add_(
                    batch_cluster_size, alpha=1 - self.decay
                )

                # Update embedding sum
                batch_embedding_sum = torch.matmul(encodings.t(), z_e_normalized)
                self.ema_embedding_sum.mul_(self.decay).add_(
                    batch_embedding_sum, alpha=1 - self.decay
                )

                # Laplace smoothing (use local variable, don't pollute ema_cluster_size)
                n = self.ema_cluster_size.sum()
                smoothed_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )

                # Normalize to get updated codebook
                self.embedding.copy_(self.ema_embedding_sum / smoothed_cluster_size.unsqueeze(1))

                # Dead code replacement
                if self.dead_code_enabled:
                    self._update_usage_statistics(encodings, z_e_normalized)

        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(z_e_normalized, z_q.detach())

        codebook_loss = torch.tensor(0.0, device=z_e.device)
        vq_loss = commitment_loss

        # Straight-through estimator
        z_q_st = z_e_normalized + (z_q - z_e_normalized).detach()

        return z_q_st, indices, vq_loss, codebook_loss, commitment_loss, z_e_normalized

    def _update_usage_statistics(self, encodings, z_e_normalized):
        """Update usage statistics and check for dead codes."""
        # Update EMA usage
        batch_usage = encodings.sum(dim=0)
        self.ema_usage.mul_(self.usage_decay).add_(batch_usage, alpha=1 - self.usage_decay)

        self.global_step += 1

        # Check dead codes periodically
        if self.global_step % self.check_interval == 0:
            self._check_and_reset_dead_codes(z_e_normalized)

    @torch.no_grad()
    def _reset_dead_codes(self, dead_mask, z_e_normalized):
        """Reset dead codes with current batch samples.

        Args:
            dead_mask: [num_embeddings] boolean mask of dead codes
            z_e_normalized: [B, embedding_dim] current batch encoder outputs
        """
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return 0

        batch_size = z_e_normalized.shape[0]
        dead_indices = torch.where(dead_mask)[0]

        # Select replacement features
        for i, dead_idx in enumerate(dead_indices):
            if i < batch_size:
                # Use unique sample
                replacement = z_e_normalized[i].clone()
            else:
                # Reuse samples with small noise
                sample_idx = i % batch_size
                replacement = z_e_normalized[sample_idx].clone()
                noise = torch.randn_like(replacement) * self.reset_noise_scale
                replacement = replacement + noise

            # Update codebook
            self.embedding[dead_idx] = replacement

            # Reset EMA statistics
            self.ema_cluster_size[dead_idx] = self.reset_count
            self.ema_embedding_sum[dead_idx] = replacement * self.reset_count
            self.ema_usage[dead_idx] = self.reset_count
            self.dead_counter[dead_idx] = 0

        return num_dead

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
                num_reset = self._reset_dead_codes(really_dead, z_e_normalized)
                # Log will be handled by caller

    def get_usage_stats(self):
        """Get codebook usage statistics."""
        if not self.dead_code_enabled:
            return {}

        active_codes = (self.ema_usage > self.dead_threshold).sum().item()
        dead_codes = self.num_embeddings - active_codes

        # Compute perplexity
        usage_probs = self.ema_usage / (self.ema_usage.sum() + 1e-10)
        perplexity = torch.exp(-torch.sum(usage_probs * torch.log(usage_probs + 1e-10)))

        # Additional metrics
        max_usage = self.ema_usage.max().item()
        total_usage = self.ema_usage.sum().item()
        max_usage_fraction = max_usage / (total_usage + 1e-10)

        nonzero_usage = self.ema_usage[self.ema_usage > 0]
        min_nonzero_usage = nonzero_usage.min().item() if len(nonzero_usage) > 0 else 0.0

        embedding_norms = self.embedding.norm(dim=1)

        return {
            'active_codes': active_codes,
            'dead_codes': dead_codes,
            'perplexity': perplexity.item(),
            'max_usage_fraction': max_usage_fraction,
            'min_nonzero_usage': min_nonzero_usage,
            'embedding_norm_mean': embedding_norms.mean().item(),
            'embedding_norm_min': embedding_norms.min().item(),
            'embedding_norm_max': embedding_norms.max().item(),
            'usage': self.ema_usage.cpu().numpy(),
        }


class TRexConvBlock(nn.Module):
    """
    T-Rex style convolution block: Conv1d + GroupNorm + GELU.
    """

    def __init__(self, in_ch, out_ch, kernel=5, stride=1, num_groups=8):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TRexEncoder(nn.Module):
    """
    T-Rex style temporal encoder (hand-level mode, mirror of the official
    T-Rex F6Encoder).

    Architecture:
        stem:    Conv1d(force_dim -> hidden, k=5) + GroupNorm + GELU
        strided: stack of down-sampling Conv1d blocks (stride=2 while T >= 4)
        proj:    Conv1d(bottleneck -> embed_dim, k=3)
        pool:    temporal mean pooling -> [B, embed_dim]

    Input  : [B, T, D]
    Output : [B, embed_dim]
    """

    def __init__(self, config):
        super().__init__()

        input_cfg = config['input']
        enc_cfg = config['encoder']

        self.history_steps = input_cfg['history_steps']
        self.force_dim = input_cfg['force_dim']
        self.embed_dim = config['quantizer']['embedding_dim']

        hidden_channels = enc_cfg.get('hidden_channels', 128)
        bottleneck_channels = enc_cfg.get('bottleneck_channels', 256)
        n_strided_blocks = enc_cfg.get('n_strided_blocks', 2)
        n_extra_blocks = enc_cfg.get('n_extra_blocks', 0)
        kernel = enc_cfg.get('kernel_size', 5)
        num_groups = enc_cfg.get('num_groups', 8)

        self.stem = TRexConvBlock(
            self.force_dim, hidden_channels, kernel=kernel, stride=1, num_groups=num_groups)

        # Optional stride-1 depth blocks (added after the stem at full time
        # resolution; mirrored in the decoder before the head).
        self.extra = nn.Sequential(*[
            TRexConvBlock(hidden_channels, hidden_channels, kernel=kernel, stride=1, num_groups=num_groups)
            for _ in range(n_extra_blocks)
        ])

        blocks = []
        cur_T = self.history_steps
        cur_ch = hidden_channels
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            out_ch = bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels
            blocks.append(TRexConvBlock(cur_ch, out_ch, kernel=kernel, stride=stride, num_groups=num_groups))
            cur_ch = out_ch
            cur_T = cur_T // stride if stride > 1 else cur_T
        self.strided = nn.Sequential(*blocks)
        self._bottleneck_T = cur_T

        self.proj = nn.Conv1d(cur_ch, self.embed_dim, kernel_size=3, padding=1)

    @property
    def bottleneck_T(self) -> int:
        return self._bottleneck_T

    def forward(self, x):
        """
        Args:
            x: [B, T, D] force history

        Returns:
            z_e: [B, embedding_dim] continuous embedding
        """
        B, T, D = x.shape
        if T != self.history_steps:
            raise ValueError(f"Encoder built for window={self.history_steps}, got T={T}")
        if D != self.force_dim:
            raise ValueError(f"Encoder built for force_dim={self.force_dim}, got D={D}")

        x = x.transpose(1, 2).contiguous()  # [B, D, T]
        x = self.stem(x)
        x = self.extra(x)
        x = self.strided(x)
        x = self.proj(x)                    # [B, E, T_bn]
        z_e = x.mean(dim=2)                 # [B, E]
        return z_e


class TRexDecoder(nn.Module):
    """
    T-Rex style decoder mirroring TRexEncoder (transposed conv upsampling,
    mirror of the official T-Rex F6Decoder).

    Architecture:
        from_embed:  Conv1d(embed_dim -> bottleneck, k=3) over bottleneck frames
        up_strided:  transposed Conv1d blocks reversing the encoder's strides
        head:        Conv1d(hidden -> force_dim, k=5)

    Input  : [B, embed_dim]
    Output : [B, T, D]
    """

    def __init__(self, config):
        super().__init__()

        input_cfg = config['input']
        enc_cfg = config['encoder']

        self.history_steps = input_cfg['history_steps']
        self.force_dim = input_cfg['force_dim']
        self.embed_dim = config['quantizer']['embedding_dim']

        hidden_channels = enc_cfg.get('hidden_channels', 128)
        bottleneck_channels = enc_cfg.get('bottleneck_channels', 256)
        n_strided_blocks = enc_cfg.get('n_strided_blocks', 2)
        n_extra_blocks = enc_cfg.get('n_extra_blocks', 0)
        kernel = enc_cfg.get('kernel_size', 5)
        num_groups = enc_cfg.get('num_groups', 8)

        # Reproduce the encoder's strides to recover the bottleneck length.
        cur_T = self.history_steps
        strides = []
        cur_ch_chain = [hidden_channels]
        for i in range(n_strided_blocks):
            stride = 2 if cur_T >= 4 else 1
            strides.append(stride)
            cur_ch_chain.append(
                bottleneck_channels if i == n_strided_blocks - 1 else hidden_channels)
            if stride > 1:
                cur_T //= stride
        self._bottleneck_T = cur_T

        self.from_embed = nn.Conv1d(self.embed_dim, bottleneck_channels, kernel_size=3, padding=1)

        # Reverse the strided stack.
        blocks = []
        in_ch = bottleneck_channels
        rev_strides = list(reversed(strides))
        rev_ch_chain = list(reversed(cur_ch_chain))  # [bottleneck, ..., hidden]
        for i, st in enumerate(rev_strides):
            out_ch = rev_ch_chain[i + 1]
            blocks.append(self._upconv_block(in_ch, out_ch, kernel=kernel, stride=st, num_groups=num_groups))
            in_ch = out_ch
        self.up_strided = nn.Sequential(*blocks)

        # Mirror of the encoder's stride-1 depth blocks (before the head).
        self.extra = nn.Sequential(*[
            self._upconv_block(hidden_channels, hidden_channels, kernel=kernel, stride=1, num_groups=num_groups)
            for _ in range(n_extra_blocks)
        ])

        self.head = nn.Conv1d(hidden_channels, self.force_dim, kernel_size=kernel, padding=kernel // 2)

    def _upconv_block(self, in_ch, out_ch, kernel, stride, num_groups):
        pad = kernel // 2
        if stride > 1:
            layer = nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=kernel, stride=stride,
                padding=pad, output_padding=stride - 1,
            )
        else:
            layer = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=pad)
        return nn.Sequential(
            layer,
            nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, z_q):
        """
        Args:
            z_q: [B, embedding_dim] quantized embedding

        Returns:
            x_recon: [B, T, D] reconstructed force history
        """
        B = z_q.shape[0]
        x = z_q.unsqueeze(2).expand(-1, -1, self._bottleneck_T)  # [B, E, T_bn]
        x = self.from_embed(x)                                    # [B, bn, T_bn]
        x = self.up_strided(x)                                    # [B, hidden, T]
        x = self.extra(x)                                         # [B, hidden, T]
        x = self.head(x)                                          # [B, D, T]

        # Some stride/output_padding combos may yield T+1 or T-1; trim/pad.
        if x.shape[-1] != self.history_steps:
            if x.shape[-1] > self.history_steps:
                x = x[..., :self.history_steps]
            else:
                x = F.pad(x, (0, self.history_steps - x.shape[-1]))

        x = x.transpose(1, 2).contiguous()                        # [B, T, D]
        return x


class MLPDecoder(nn.Module):
    """
    Original MLP decoder for reconstructing force history.

    Input  : [B, embed_dim]
    Output : [B, T, D]
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

        self.encoder = TRexEncoder(config)

        dead_code_config = config.get('dead_code_replacement', {})
        init_config = config.get('quantizer', {})

        self.quantizer = VectorQuantizer(
            num_embeddings=config['quantizer']['num_embeddings'],
            embedding_dim=config['quantizer']['embedding_dim'],
            commitment_cost=config['quantizer']['commitment_cost'],
            decay=config['quantizer'].get('ema_decay', 0.99),
            epsilon=config['quantizer'].get('ema_epsilon', 1e-5),
            dead_code_config=dead_code_config,
            init_config=init_config
        )

        # Decoder: mirror (transposed conv) by default, or the original MLP.
        decoder_type = config.get('decoder', {}).get('type', 'mirror')
        if decoder_type == 'mlp':
            self.decoder = MLPDecoder(config)
        elif decoder_type in ('mirror', 'trex'):
            self.decoder = TRexDecoder(config)
        else:
            raise ValueError(
                f"Unknown decoder type {decoder_type!r}; expected 'mirror' or 'mlp'")

        # T-Rex style magnitude-weighted reconstruction loss.
        self.use_magnitude_weight = config.get('use_magnitude_weight', False)
        self.weight_alpha = config.get('weight_alpha', 2.0)
        self.weight_tau = config.get('weight_tau', 4.0)

    def _recon_weight(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Per-sample weight = 1 + α·sigmoid(magnitude/τ − 1).

        magnitude is the L2 norm of the raw (un-normalized) force window.
        Weak / free-air windows (magnitude ≈ 0) get a low weight; strong
        contact windows approach 1 + α. Mirrors the official T-Rex design.
        """
        return 1.0 + self.weight_alpha * torch.sigmoid(
            magnitude / self.weight_tau - 1.0)

    def forward(self, x, magnitude=None):
        """
        Args:
            x: [B, T, D] force history tensor
            magnitude: [B] optional L2 norm of the *raw* force window. Used
                for magnitude-weighted reconstruction loss when enabled.

        Returns:
            Dictionary with all outputs and losses
        """
        z_e = self.encoder(x)
        z_q, indices, vq_loss, codebook_loss, commitment_loss, z_e_normalized = self.quantizer(
            z_e,
            update_ema=self.training,
        )
        x_recon = self.decoder(z_q)

        if self.use_magnitude_weight and magnitude is not None:
            per_sample = (x_recon - x).pow(2).mean(dim=[1, 2])        # [B]
            w = self._recon_weight(magnitude.to(per_sample.device))   # [B]
            recon_loss = (per_sample * w).sum() / (w.sum() + 1e-8)
        else:
            recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss

        return {
            'z_e': z_e,
            'z_e_normalized': z_e_normalized,
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
        """Encode to discrete tokens.

        This is a READ-ONLY interface: the quantizer runs with
        ``update_ema=False``, so calling encode() never mutates the codebook,
        EMA buffers, usage counters or global_step even if the model is still
        in train mode.
        """
        if not self.quantizer.initialized:
            raise RuntimeError(
                "VectorQuantizer codebook is not initialized; cannot encode. "
                "Run at least one training forward (or initialize the codebook) "
                "before calling encode()."
            )

        z_e = self.encoder(x)
        z_q, indices, _, _, _, _ = self.quantizer(
            z_e,
            update_ema=False,
        )
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

        B = 4
        T = config['model']['input']['history_steps']
        D = config['model']['input']['force_dim']
        x = torch.randn(B, T, D)

        output = model(x)

        print("Model test successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output['x_recon'].shape}")
        print(f"Token indices: {output['indices']}")
        print(f"Total loss: {output['total_loss'].item():.6f}")
