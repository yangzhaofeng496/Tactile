import json
from pathlib import Path

import torch
import torch.nn as nn


class TactileImageEncoder(nn.Module):
    """
    使用2D卷积处理触觉图像

    输入:
        B,10,6,288,384  (6通道 = 左触觉3通道 + 右触觉3通道)
    输出:
        B,256
    """

    def __init__(self, in_channels=6, output_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)


    def forward(self,x):

        # B,T,C,H,W
        B,T,C,H,W=x.shape

        x=x.reshape(B*T,C,H,W)

        x=self.encoder(x)

        x=x.flatten(1)

        x=self.fc(x)
        x=torch.relu(self.output_norm(x))

        # temporal aggregation
        x=x.reshape(B,T,-1)

        x=x.mean(dim=1)

        return x


class TactileForceEncoder(nn.Module):
    """
    使用1D时间卷积网络处理历史合力数据

    输入:
        B, 16, 6  (16个时间步，每步6维合力)
    输出:
        B, 256
    """

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=256):
        super().__init__()

        # 1D时间卷积层 + 下采样
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
        )

        # 时间维度平均池化
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层输出256维
        self.fc = nn.Linear(hidden_dim * 4, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)


    def forward(self, x):
        """
        x: [B, T, D] = [B, 16, 6]
        """
        # 转换为卷积格式 [B, D, T]
        x = x.transpose(1, 2)  # [B, 6, 16]

        # 1D时间卷积 + 下采样
        x = self.conv1(x)  # [B, 64, 8]
        x = self.conv2(x)  # [B, 128, 4]
        x = self.conv3(x)  # [B, 256, 2]

        # 时间维度平均池化
        x = self.temporal_pool(x)  # [B, 256, 1]

        # 展平
        x = x.squeeze(-1)  # [B, 256]

        # 全连接层
        x = self.fc(x)  # [B, 256]
        x = torch.relu(self.output_norm(x))

        return x


class VQVAEForceEncoder(nn.Module):
    """
    使用冻结的VQ-VAE编码器处理历史合力数据

    输入:
        B, T, D  (T个时间步，每步D维合力)
    输出:
        B, 256  (码本向量)
    """

    def __init__(self, vqvae_checkpoint_path, freeze=True):
        super().__init__()

        # 加载VQ-VAE模型
        from TactileSelfencoder.vqvae_model import build_vqvae_from_config
        import os

        config_path = os.path.join(os.path.dirname(vqvae_checkpoint_path), "../vqvae_config.yaml")
        if not os.path.exists(config_path):
            config_path = "TactileSelfencoder/vqvae_config.yaml"

        self.vqvae, _ = build_vqvae_from_config(config_path)

        # 加载预训练权重 (使用weights_only=False以兼容旧格式)
        checkpoint = torch.load(vqvae_checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.vqvae.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.vqvae.load_state_dict(checkpoint)

        # 冻结VQ-VAE参数
        if freeze:
            for param in self.vqvae.parameters():
                param.requires_grad = False
            self.vqvae.eval()

        self.freeze = freeze

    def forward(self, x, return_token_id=False):
        """
        x: [B, T, D] 历史力数据
        返回:
            - 如果 return_token_id=False: [B, 256] 量化后的码本向量
            - 如果 return_token_id=True: (z_q, indices) 其中 indices 是 [B] token ID
        """
        if self.freeze:
            self.vqvae.eval()
            with torch.no_grad():
                indices, z_q = self.vqvae.encode(x)
        else:
            indices, z_q = self.vqvae.encode(x)

        if return_token_id:
            return z_q, indices
        return z_q


class CurrentForceEncoder(nn.Module):
    """
    使用MLP处理当前6维力数据

    输入:
        B, 6  (当前时刻的6维力)
    输出:
        B, 128
    """

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        x: [B, D] = [B, 6]
        """
        return torch.relu(self.output_norm(self.net(x)))


class ForceFiLM(nn.Module):
    """Condition fusion features on force context via feature-wise modulation."""

    def __init__(self, condition_dim, feature_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return features * (1.0 + gamma) + beta



class StateEncoder(nn.Module):

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=128):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)


    def forward(self,x):
        return torch.relu(self.output_norm(self.net(x)))



class ActionEncoder(nn.Module):
    """
    ACT chunk encoder

    输入:
        B,T,D

    输出:
        B,256
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=256,
    ):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim,output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)


    def forward(self,x):

        B,T,D=x.shape

        x=x.reshape(B,T*D)

        return torch.relu(self.output_norm(self.net(x)))



class FusionEncoder(nn.Module):

    def __init__(self, input_dim=640, hidden_dim=512, output_dim=256):

        super().__init__()

        self.encoder=nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                output_dim
            ),

            nn.ReLU()
        )


    def forward(self,x):

        return self.encoder(x)


class MultiHeadFusionEncoder(nn.Module):
    """Single-layer normalized multi-head modality fusion."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_heads=4,
        dropout=0.1,
        use_gate=True,
        modality_dims=None,
        use_modality_gate=False,
        modality_gate_temperature=1.0,
        modality_gate_residual_scale=1.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("fusion hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.use_gate = bool(use_gate)
        self.modality_dims = tuple(int(dim) for dim in (modality_dims or ()))
        self.use_modality_gate = bool(use_modality_gate)
        self.modality_gate_temperature = float(modality_gate_temperature)
        self.modality_gate_residual_scale = float(modality_gate_residual_scale)
        if self.modality_gate_temperature <= 0:
            raise ValueError("modality_gate_temperature must be positive")
        if self.modality_gate_residual_scale < 0:
            raise ValueError("modality_gate_residual_scale must be non-negative")
        if self.use_modality_gate:
            if not self.modality_dims or sum(self.modality_dims) != int(input_dim):
                raise ValueError(
                    "dynamic modality gate requires modality_dims to sum to fusion input_dim"
                )
            self.modality_gate = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.GELU(),
                nn.Linear(32, len(self.modality_dims)),
            )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ) for _ in range(self.num_heads)
        ])
        self.head_weights = nn.Parameter(torch.zeros(self.num_heads))
        gate_hidden_dim = 32
        self.head_gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, self.num_heads),
        )
        self.modality_weights = nn.Parameter(torch.zeros(self.num_heads, 4))
        self.merge_norm = nn.LayerNorm(hidden_dim)
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_skip = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_attention=False):
        modality_weights = torch.softmax(self.modality_weights, dim=-1)
        if self.use_modality_gate:
            dynamic_modality_weights = torch.softmax(
                self.modality_gate(x) / self.modality_gate_temperature,
                dim=-1,
            )
            # Residual gating preserves the original feature when weights are
            # uniform and prevents a modality from being fully suppressed.
            normalized_weight = (
                len(self.modality_dims) * dynamic_modality_weights
            )
            feature_scale = 1.0 + self.modality_gate_residual_scale * (
                normalized_weight - 1.0
            )
            chunks = torch.split(x, self.modality_dims, dim=-1)
            x = torch.cat(
                [chunk * feature_scale[:, index:index + 1]
                 for index, chunk in enumerate(chunks)],
                dim=-1,
            )
            modality_weights = dynamic_modality_weights
        heads = torch.stack([head(x) for head in self.heads], dim=1)
        static_weights = torch.softmax(self.head_weights, dim=0)
        if self.use_gate:
            gate_weights = torch.softmax(
                self.head_gate(x) + self.head_weights.view(1, self.num_heads),
                dim=-1,
            )
        else:
            gate_weights = static_weights.expand(x.shape[0], -1)
        weights = gate_weights.unsqueeze(-1)
        merged = self.merge_norm((heads * weights).sum(dim=1))
        residual = self.skip(x)
        fused = self.merge_norm(merged + residual)
        output = self.output_norm(self.ffn(fused) + self.output_skip(fused))
        if return_attention:
            return output, {
                "head_weights": static_weights,
                "gate_weights": gate_weights,
                "modality_weights": modality_weights,
                "head_outputs": heads,
            }
        return output



class ModalitySelfAttentionFusionEncoder(nn.Module):
    """One Transformer-style self-attention block over modality tokens."""

    def __init__(self, modality_dims, embed_dim, output_dim, num_heads=4, ffn_dim=192, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("fusion embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, embed_dim), nn.LayerNorm(embed_dim))
            for dim in modality_dims
        ])
        self.modality_embeddings = nn.Parameter(torch.zeros(1, len(modality_dims), embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, modality_features, return_attention=False):
        tokens = torch.stack(
            [projection(feature) for projection, feature in zip(self.projections, modality_features)], dim=1
        ) + self.modality_embeddings
        attended, weights = self.attention(
            tokens, tokens, tokens, need_weights=return_attention, average_attn_weights=False
        )
        tokens = self.attn_norm(tokens + attended)
        tokens = self.ffn_norm(tokens + self.ffn(tokens))
        output = self.output(tokens.mean(dim=1))
        if return_attention:
            return output, {"attention_weights": weights}
        return output


class PairwiseCrossAttentionFusionEncoder(nn.Module):
    """Bidirectional pairwise cross-attention over learned modality tokens."""

    def __init__(self, modality_dims, embed_dim, output_dim, num_heads=4, num_tokens=4, ffn_dim=192, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("fusion embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.num_tokens = int(num_tokens)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, num_tokens * embed_dim), nn.LayerNorm(num_tokens * embed_dim))
            for dim in modality_dims
        ])
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        self.pair_weights = nn.Parameter(torch.zeros(len(self.pairs)))
        self.merge_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Sequential(nn.Linear(embed_dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, modality_features, return_attention=False):
        tokens = [
            projection(feature).view(feature.shape[0], self.num_tokens, -1)
            for projection, feature in zip(self.projections, modality_features)
        ]
        pair_outputs = []
        pair_attention = []
        for left, right in self.pairs:
            left_out, left_weights = self.attention(
                tokens[left], tokens[right], tokens[right],
                need_weights=return_attention, average_attn_weights=False,
            )
            right_out, right_weights = self.attention(
                tokens[right], tokens[left], tokens[left],
                need_weights=return_attention, average_attn_weights=False,
            )
            pair_outputs.append((left_out.mean(dim=1) + right_out.mean(dim=1)) / 2.0)
            if return_attention:
                pair_attention.append(torch.stack([left_weights, right_weights], dim=1))
        pair_outputs = torch.stack(pair_outputs, dim=1)
        pair_weights = torch.softmax(self.pair_weights, dim=0).view(1, len(self.pairs), 1)
        fused = self.merge_norm((pair_outputs * pair_weights).sum(dim=1))
        residual = torch.stack(tokens, dim=1).mean(dim=(1, 2))
        fused = self.merge_norm(fused + residual)
        fused = self.ffn_norm(fused + self.ffn(fused))
        output = self.output(fused)
        if return_attention:
            return output, {
                "pair_attention_weights": torch.stack(pair_attention, dim=1),
                "pair_weights": torch.softmax(self.pair_weights, dim=0),
            }
        return output


class VisualTokenEncoder(nn.Module):
    """
    将ACT Transformer encoder输出的视觉token聚合为固定维度特征。

    输入:
        - [B, S, D] (S个视觉token，来自冻结ACT的transformer encoder输出)
        - 或 [B, D] (已mean-pooled的视觉特征，来自离线预处理缓存)
    输出:
        B, output_dim
    """

    def __init__(
        self,
        input_dim=512,
        output_dim=256,
        pooling="mean",
    ):
        super().__init__()

        if pooling not in {"mean"}:
            raise ValueError(f"Unknown visual token pooling: {pooling!r}. Choose 'mean'.")
        self.pooling = pooling

        self.fc = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        x: [B, S, D] 视觉token序列，或 [B, D] 已pooled特征
        """
        if x.ndim == 3:
            if self.pooling == "mean":
                x = x.mean(dim=1)  # [B, D]
        elif x.ndim != 2:
            raise ValueError(
                "Visual input must be [B, S, D] or [B, D], got "
                f"{tuple(x.shape)}."
            )
        x = self.fc(x)
        return torch.relu(self.output_norm(x))


class ResidualDecoder(nn.Module):

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        action_horizon=30,
        action_dim=6,
        per_step=False,
        temporal=False,
        temporal_num_heads=4,
        temporal_ffn_dim=128,
        dropout=0.1,
    ):

        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.per_step = bool(per_step)
        self.temporal = bool(temporal)
        if self.temporal:
            if input_dim % int(temporal_num_heads) != 0:
                raise ValueError("decoder input_dim must be divisible by temporal_num_heads")
            self.temporal_pos = nn.Parameter(
                torch.zeros(1, self.action_horizon, int(input_dim))
            )
            layer = nn.TransformerEncoderLayer(
                d_model=int(input_dim),
                nhead=int(temporal_num_heads),
                dim_feedforward=int(temporal_ffn_dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)

        self.decoder=nn.Sequential(

            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                self.action_dim
                if self.per_step or self.temporal
                else self.action_horizon * self.action_dim,
            )

        )


    def forward(self,x):
        if self.temporal:
            if x.ndim == 2:
                x = x.unsqueeze(1).expand(-1, self.action_horizon, -1)
            if x.ndim != 3 or x.shape[1] != self.action_horizon:
                raise ValueError(
                    f"temporal decoder expects [B, {self.action_horizon}, D], got {tuple(x.shape)}"
                )
            x = self.temporal_encoder(x + self.temporal_pos)
            x = self.decoder(x)
            return x
        if self.per_step:
            if x.ndim != 3:
                raise ValueError(
                    f"per-step decoder expects [B, T, D], got {tuple(x.shape)}"
                )
            return self.decoder(x)
        B=x.shape[0]
        x=self.decoder(x)
        return x.reshape(B, self.action_horizon, self.action_dim)


class TimestepModalityGate(nn.Module):
    """Generate a modality mixture separately for every action step."""

    def __init__(
        self,
        modality_dims,
        context_dim,
        output_dim,
        action_horizon,
        action_dim=6,
        hidden_dim=32,
        step_embedding_dim=16,
        temperature=1.5,
        residual_scale=0.5,
    ):
        super().__init__()
        self.modality_dims = tuple(int(dim) for dim in modality_dims)
        self.num_modalities = len(self.modality_dims)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        if self.temperature <= 0 or self.residual_scale < 0:
            raise ValueError("Invalid timestep modality gate temperature/scale")
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim))
            for dim in self.modality_dims
        ])
        self.step_embeddings = nn.Parameter(
            torch.zeros(1, self.action_horizon, step_embedding_dim)
        )
        self.action_condition = nn.Sequential(
            nn.Linear(self.action_dim, step_embedding_dim),
            nn.GELU(),
        )
        self.router = nn.Sequential(
            nn.Linear(context_dim + 2 * step_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_modalities),
        )

    def forward(self, context, modality_features, action_chunk=None):
        if len(modality_features) != self.num_modalities:
            raise ValueError("Timestep modality feature count does not match gate")
        if action_chunk is None or action_chunk.ndim != 3:
            raise ValueError("Action-conditioned timestep gate requires [B, T, action_dim]")
        if action_chunk.shape[1] != self.action_horizon or action_chunk.shape[2] != self.action_dim:
            raise ValueError("Action chunk shape does not match timestep gate configuration")
        projected = torch.stack(
            [projection(feature) for projection, feature in zip(self.projections, modality_features)],
            dim=1,
        )
        router_input = torch.cat([
            context.unsqueeze(1).expand(-1, self.action_horizon, -1),
            self.step_embeddings.expand(context.shape[0], -1, -1),
            self.action_condition(action_chunk),
        ], dim=-1)
        weights = torch.softmax(self.router(router_input) / self.temperature, dim=-1)
        fused = (weights.unsqueeze(-1) * projected.unsqueeze(1)).sum(dim=2)
        gated_context = context.unsqueeze(1) + self.residual_scale * fused
        return gated_context, weights


class ResidualActionTransformerFusion(nn.Module):
    """Token-based residual fusion with per-step action queries."""

    def __init__(
        self,
        modality_dims,
        action_dim,
        action_horizon,
        output_dim=80,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("residual transformer model_dim must be divisible by num_heads")
        self.modality_dims = tuple(int(dim) for dim in modality_dims)
        self.action_horizon = int(action_horizon)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, model_dim), nn.LayerNorm(model_dim))
            for dim in self.modality_dims
        ])
        self.action_projection = nn.Sequential(
            nn.Linear(int(action_dim), model_dim),
            nn.LayerNorm(model_dim),
        )
        token_count = len(self.modality_dims) + self.action_horizon
        self.token_embeddings = nn.Parameter(torch.zeros(1, token_count, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output = nn.Sequential(
            nn.Linear(model_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, modality_features, act_chunk, return_attention=False):
        if len(modality_features) != len(self.projections):
            raise ValueError("Residual transformer modality count mismatch")
        modality_tokens = torch.stack(
            [projection(feature) for projection, feature in zip(self.projections, modality_features)],
            dim=1,
        )
        action_tokens = self.action_projection(act_chunk)
        tokens = torch.cat([modality_tokens, action_tokens], dim=1)
        tokens = self.encoder(tokens + self.token_embeddings)
        action_context = self.output(tokens[:, len(self.modality_dims):])
        if return_attention:
            return action_context, {}
        return action_context



class TactileResidualACT(nn.Module):

    def __init__(
        self,
        tactile_encoder_type="force",
        action_horizon=30,
        action_dim=6,
        use_tactile_history=True,
        tactile_encoder_cfg=None,
        state_encoder_cfg=None,
        action_encoder_cfg=None,
        fusion_cfg=None,
        decoder_cfg=None,
        tactile_channel_mean=None,
        tactile_channel_std=None,
        tactile_channel_names=None,
        normalize_tactile_input=False,
        state_mean=None,
        state_std=None,
        state_channel_names=None,
        normalize_state_input=False,
        current_force_encoder_cfg=None,
        current_force_mean=None,
        current_force_std=None,
        normalize_current_force_input=False,
        force_film_cfg=None,
        use_act_visual=False,
        visual_encoder_cfg=None,
    ):
        """
        Args:
            tactile_encoder_type: "force", "image", 或 "vqvae"
                - "force": 使用 TactileForceEncoder (1D卷积，处理12维合力)
                - "image": 使用 TactileImageEncoder (2D卷积，处理触觉图像)
                - "vqvae": 使用 VQVAEForceEncoder (冻结的VQ-VAE编码器)
        """

        super().__init__()

        tactile_encoder_cfg = tactile_encoder_cfg or {}
        state_encoder_cfg = state_encoder_cfg or {}
        action_encoder_cfg = action_encoder_cfg or {}
        fusion_cfg = fusion_cfg or {}
        decoder_cfg = decoder_cfg or {}
        current_force_encoder_cfg = current_force_encoder_cfg or {}
        force_film_cfg = force_film_cfg or {}
        visual_encoder_cfg = visual_encoder_cfg or {}

        if bool(fusion_cfg.get("use_modality_gate", False)) and bool(
            fusion_cfg.get("use_timestep_modality_gate", False)
        ):
            raise ValueError(
                "use_modality_gate and use_timestep_modality_gate are mutually exclusive "
                "for a clean ablation."
            )

        # 是否启用ACT视觉token输入
        self.use_act_visual = bool(use_act_visual)

        # 是否启用触觉历史输入（false=完全移除触觉历史分支）
        self.use_tactile_history = bool(use_tactile_history)
        tactile_input_dim = 0
        tactile_output_dim = 0

        # 根据配置选择触觉编码器
        if self.use_tactile_history:
            if tactile_encoder_type == "force":
                tactile_input_dim = int(tactile_encoder_cfg.get("input_dim", 12))
                tactile_output_dim = int(tactile_encoder_cfg.get("output_dim", 256))
                self.tactile_encoder = TactileForceEncoder(
                    input_dim=tactile_input_dim,
                    hidden_dim=int(tactile_encoder_cfg.get("hidden_dim", 64)),
                    output_dim=tactile_output_dim,
                )
            elif tactile_encoder_type == "image":
                tactile_input_dim = int(tactile_encoder_cfg.get("in_channels", 6))
                tactile_output_dim = int(tactile_encoder_cfg.get("output_dim", 256))
                self.tactile_encoder = TactileImageEncoder(
                    in_channels=tactile_input_dim,
                    output_dim=tactile_output_dim,
                )
            elif tactile_encoder_type == "vqvae":
                tactile_input_dim = int(tactile_encoder_cfg.get("input_dim", 12))
                tactile_output_dim = int(tactile_encoder_cfg.get("output_dim", 256))
                vqvae_checkpoint = tactile_encoder_cfg.get("vqvae_checkpoint_path")
                if vqvae_checkpoint is None:
                    raise ValueError("vqvae_checkpoint_path is required when tactile_encoder_type='vqvae'")
                self.tactile_encoder = VQVAEForceEncoder(
                    vqvae_checkpoint_path=vqvae_checkpoint,
                    freeze=tactile_encoder_cfg.get("freeze", True),
                )
            else:
                raise ValueError(f"Unknown tactile_encoder_type: {tactile_encoder_type}. Choose 'force', 'image', or 'vqvae'.")

        self.tactile_encoder_type = tactile_encoder_type
        self.normalize_tactile_input = bool(normalize_tactile_input)
        mean = torch.as_tensor(
            tactile_channel_mean if tactile_channel_mean is not None else [],
            dtype=torch.float32,
        ).reshape(-1)
        std = torch.as_tensor(
            tactile_channel_std if tactile_channel_std is not None else [],
            dtype=torch.float32,
        ).reshape(-1)
        channel_names = list(tactile_channel_names or [])
        if self.normalize_tactile_input and not self.use_tactile_history:
            self.normalize_tactile_input = False
        if self.normalize_tactile_input:
            if mean.numel() != tactile_input_dim or std.numel() != tactile_input_dim:
                raise ValueError(
                    "Input Z-score statistics must match tactile channels: "
                    f"mean={mean.numel()}, std={std.numel()}, expected={tactile_input_dim}."
                )
            if len(channel_names) != tactile_input_dim:
                raise ValueError(
                    "tactile_channel_names must identify every input channel in order: "
                    f"got {len(channel_names)}, expected {tactile_input_dim}."
                )
            if torch.any(std <= 0):
                raise ValueError("All tactile channel standard deviations must be positive.")
            if len(set(channel_names)) != len(channel_names):
                raise ValueError("tactile_channel_names must be unique.")
        self.tactile_channel_names = tuple(channel_names)
        self.register_buffer("tactile_channel_mean", mean)
        self.register_buffer("tactile_channel_std", std)

        # Current force encoder (新增)
        current_force_input_dim = int(current_force_encoder_cfg.get("input_dim", 6))
        current_force_output_dim = int(current_force_encoder_cfg.get("output_dim", 128))
        self.current_force_encoder = CurrentForceEncoder(
            input_dim=current_force_input_dim,
            hidden_dim=int(current_force_encoder_cfg.get("hidden_dim", 64)),
            output_dim=current_force_output_dim,
        )

        # Current force normalization (新增)
        self.normalize_current_force_input = bool(normalize_current_force_input)
        current_force_mean_tensor = torch.as_tensor(
            current_force_mean if current_force_mean is not None else [],
            dtype=torch.float32,
        ).reshape(-1)
        current_force_std_tensor = torch.as_tensor(
            current_force_std if current_force_std is not None else [],
            dtype=torch.float32,
        ).reshape(-1)
        if self.normalize_current_force_input:
            if (
                current_force_mean_tensor.numel() != current_force_input_dim
                or current_force_std_tensor.numel() != current_force_input_dim
            ):
                raise ValueError(
                    "Current force Z-score statistics must match current force input dimensions: "
                    f"mean={current_force_mean_tensor.numel()}, "
                    f"std={current_force_std_tensor.numel()}, expected={current_force_input_dim}."
                )
            if torch.any(current_force_std_tensor <= 0):
                raise ValueError("All current force standard deviations must be positive.")
        self.register_buffer("current_force_mean", current_force_mean_tensor)
        self.register_buffer("current_force_std", current_force_std_tensor)

        state_output_dim = int(state_encoder_cfg.get("output_dim", 128))
        state_input_dim = int(state_encoder_cfg.get("input_dim", 6))
        self.normalize_state_input = bool(normalize_state_input)
        state_mean_tensor = torch.as_tensor(
            state_mean if state_mean is not None else [],
            dtype=torch.float32,
        ).reshape(-1)
        state_std_tensor = torch.as_tensor(
            state_std if state_std is not None else [],
            dtype=torch.float32,
        ).reshape(-1)
        state_names = list(state_channel_names or [])
        if self.normalize_state_input:
            if (
                state_mean_tensor.numel() != state_input_dim
                or state_std_tensor.numel() != state_input_dim
            ):
                raise ValueError(
                    "State Z-score statistics must match state input dimensions: "
                    f"mean={state_mean_tensor.numel()}, "
                    f"std={state_std_tensor.numel()}, expected={state_input_dim}."
                )
            if len(state_names) != state_input_dim:
                raise ValueError(
                    "state_channel_names must identify every state dimension in order."
                )
            if torch.any(state_std_tensor <= 0):
                raise ValueError("All state standard deviations must be positive.")
        self.state_channel_names = tuple(state_names)
        self.register_buffer("state_mean", state_mean_tensor)
        self.register_buffer("state_std", state_std_tensor)
        self.state_encoder = StateEncoder(
            input_dim=state_input_dim,
            hidden_dim=int(state_encoder_cfg.get("hidden_dim", 64)),
            output_dim=state_output_dim,
        )

        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)

        expected_action_input_dim = (
            self.action_horizon * self.action_dim
        )
        action_input_dim = int(
            action_encoder_cfg.get(
                "input_dim",
                expected_action_input_dim,
            )
        )
        if action_input_dim != expected_action_input_dim:
            raise ValueError(
                "action_encoder.input_dim must equal "
                "decoder.action_horizon * decoder.action_dim: "
                f"{action_input_dim} vs "
                f"{self.action_horizon} * {self.action_dim}."
            )

        self.action_encoder = ActionEncoder(
            input_dim=action_input_dim,
            hidden_dim=int(
                action_encoder_cfg.get("hidden_dim", 128)
            ),
            output_dim=int(
                action_encoder_cfg.get("output_dim", 256)
            ),
        )

        action_output_dim = int(action_encoder_cfg.get("output_dim", 256))
        expected_fusion_input_dim = (
            tactile_output_dim + current_force_output_dim + state_output_dim + action_output_dim
        )

        # ACT视觉token编码器 (可选)
        self.visual_encoder = None
        visual_output_dim = 0
        if self.use_act_visual:
            visual_output_dim = int(visual_encoder_cfg.get("output_dim", 256))
            self.visual_encoder = VisualTokenEncoder(
                input_dim=int(visual_encoder_cfg.get("input_dim", 512)),
                output_dim=visual_output_dim,
                pooling=visual_encoder_cfg.get("pooling", "mean"),
            )
            expected_fusion_input_dim += visual_output_dim

        fusion_input_dim = int(
            fusion_cfg.get("input_dim", expected_fusion_input_dim)
        )
        if not self.use_tactile_history:
            # 关闭触觉历史分支时，自动适配融合输入维度
            fusion_input_dim = expected_fusion_input_dim
        if fusion_input_dim != expected_fusion_input_dim:
            raise ValueError(
                "fusion.input_dim must equal tactile + current_force + state + action "
                "(+ visual if enabled) feature dimensions: "
                f"{fusion_input_dim} vs {expected_fusion_input_dim}."
            )
        fusion_output_dim = int(fusion_cfg.get("output_dim", 256))
        fusion_type = fusion_cfg.get("type", "mlp")
        self.fusion_returns_sequence = fusion_type == "residual_transformer"
        fusion_cls = MultiHeadFusionEncoder if fusion_type == "multihead" else FusionEncoder
        fusion_kwargs = {
            "input_dim": fusion_input_dim,
            "hidden_dim": int(fusion_cfg.get("hidden_dim", 512)),
            "output_dim": fusion_output_dim,
        }
        if fusion_cls is MultiHeadFusionEncoder:
            modality_dims = []
            if tactile_output_dim:
                modality_dims.append(tactile_output_dim)
            modality_dims += [
                current_force_output_dim,
                state_output_dim,
                action_output_dim,
            ]
            if visual_output_dim:
                modality_dims.append(visual_output_dim)
            fusion_kwargs.update(
                num_heads=int(fusion_cfg.get("num_heads", 4)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
                use_gate=bool(fusion_cfg.get("use_gate", True)),
                modality_dims=modality_dims,
                use_modality_gate=bool(
                    fusion_cfg.get("use_modality_gate", False)
                ),
                modality_gate_temperature=float(
                    fusion_cfg.get("modality_gate_temperature", 1.0)
                ),
                modality_gate_residual_scale=float(
                    fusion_cfg.get("modality_gate_residual_scale", 1.0)
                ),
            )
        if fusion_type == "residual_transformer":
            transformer_modality_dims = []
            if tactile_output_dim:
                transformer_modality_dims.append(tactile_output_dim)
            transformer_modality_dims += [
                current_force_output_dim,
                state_output_dim,
                action_output_dim,
            ]
            if visual_output_dim:
                transformer_modality_dims.append(visual_output_dim)
            self.fusion = ResidualActionTransformerFusion(
                modality_dims=transformer_modality_dims,
                action_dim=self.action_dim,
                action_horizon=self.action_horizon,
                output_dim=fusion_output_dim,
                model_dim=int(fusion_cfg.get("model_dim", 128)),
                num_heads=int(fusion_cfg.get("num_heads", 4)),
                num_layers=int(fusion_cfg.get("num_layers", 2)),
                ffn_dim=int(fusion_cfg.get("ffn_dim", 256)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
            )
        elif fusion_type == "self_attention":
            if tactile_output_dim:
                raise ValueError("self_attention fusion currently expects tactile history to be disabled")
            self.fusion = ModalitySelfAttentionFusionEncoder(
                modality_dims=[current_force_output_dim, state_output_dim, action_output_dim, visual_output_dim],
                embed_dim=int(fusion_cfg.get("embed_dim", fusion_cfg.get("hidden_dim", 96))),
                output_dim=fusion_output_dim,
                num_heads=int(fusion_cfg.get("num_heads", 4)),
                ffn_dim=int(fusion_cfg.get("ffn_dim", 192)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
            )
        elif fusion_type == "pairwise_cross_attention":
            if tactile_output_dim:
                raise ValueError("pairwise_cross_attention currently expects tactile history to be disabled")
            self.fusion = PairwiseCrossAttentionFusionEncoder(
                modality_dims=[current_force_output_dim, state_output_dim, action_output_dim, visual_output_dim],
                embed_dim=int(fusion_cfg.get("embed_dim", fusion_cfg.get("hidden_dim", 96))),
                output_dim=fusion_output_dim,
                num_heads=int(fusion_cfg.get("num_heads", 4)),
                num_tokens=int(fusion_cfg.get("num_tokens", 4)),
                ffn_dim=int(fusion_cfg.get("ffn_dim", 192)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
            )
        else:
            self.fusion = fusion_cls(**fusion_kwargs)
        self.force_film_enabled = bool(force_film_cfg.get("enabled", False))
        self.force_film = None
        if self.force_film_enabled:
            self.force_film = ForceFiLM(
                condition_dim=current_force_output_dim,
                feature_dim=fusion_output_dim,
                hidden_dim=int(force_film_cfg.get("hidden_dim", 64)),
            )

        self.single_step_delta = bool(decoder_cfg.get("single_step_delta", False))
        decoder_output_dim = int(
            decoder_cfg.get(
                "output_dim",
                expected_action_input_dim,
            )
        )
        if not self.single_step_delta and decoder_output_dim != expected_action_input_dim:
            raise ValueError(
                "decoder.output_dim must equal "
                "decoder.action_horizon * decoder.action_dim: "
                f"{decoder_output_dim} vs "
                f"{self.action_horizon} * {self.action_dim}."
            )
        decoder_input_dim = int(decoder_cfg.get("input_dim", fusion_output_dim))
        if decoder_input_dim != fusion_output_dim:
            raise ValueError(
                "decoder.input_dim must equal fusion.output_dim: "
                f"{decoder_input_dim} vs {fusion_output_dim}."
            )

        self.use_timestep_modality_gate = bool(
            fusion_cfg.get("use_timestep_modality_gate", False)
        )
        self.timestep_modality_gate = None
        if self.use_timestep_modality_gate:
            timestep_modality_dims = []
            if tactile_output_dim:
                timestep_modality_dims.append(tactile_output_dim)
            timestep_modality_dims += [
                current_force_output_dim,
                state_output_dim,
                action_output_dim,
            ]
            if visual_output_dim:
                timestep_modality_dims.append(visual_output_dim)
            self.timestep_modality_gate = TimestepModalityGate(
                modality_dims=timestep_modality_dims,
                context_dim=fusion_output_dim,
                output_dim=decoder_input_dim,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                hidden_dim=int(fusion_cfg.get("timestep_gate_hidden_dim", 32)),
                step_embedding_dim=int(
                    fusion_cfg.get("timestep_gate_embedding_dim", 16)
                ),
                temperature=float(
                    fusion_cfg.get("timestep_gate_temperature", 1.5)
                ),
                residual_scale=float(
                    fusion_cfg.get("timestep_gate_residual_scale", 0.5)
                ),
            )

        self.decoder = ResidualDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=int(decoder_cfg.get("hidden_dim", 256)),
            action_horizon=1 if self.single_step_delta else self.action_horizon,
            action_dim=self.action_dim,
            per_step=(self.use_timestep_modality_gate or self.fusion_returns_sequence),
            temporal=bool(decoder_cfg.get("use_temporal_decoder", False)),
            temporal_num_heads=int(decoder_cfg.get("temporal_num_heads", 4)),
            temporal_ffn_dim=int(decoder_cfg.get("temporal_ffn_dim", 128)),
            dropout=float(decoder_cfg.get("dropout", 0.1)),
        )


    def forward(
        self,
        tactile_history,
        current_force,
        state,
        act_chunk,
        act_visual_tokens=None,
        return_feature_metrics=False,
    ):

        vqvae_token_id = None
        tactile_feature = None

        if self.use_tactile_history:
            if self.normalize_tactile_input:
                if self.tactile_encoder_type == "force" or self.tactile_encoder_type == "vqvae":
                    expected_channels = self.tactile_channel_mean.numel()
                    if tactile_history.ndim != 3 or tactile_history.shape[-1] != expected_channels:
                        raise ValueError(
                            "Force tactile input must have shape [B, T, C] with "
                            f"C={expected_channels}, got {tuple(tactile_history.shape)}."
                        )
                    tactile_history = (
                        tactile_history.to(torch.float32)
                        - self.tactile_channel_mean.view(1, 1, -1)
                    ) / self.tactile_channel_std.view(1, 1, -1)
                else:
                    expected_channels = self.tactile_channel_mean.numel()
                    if tactile_history.ndim != 5 or tactile_history.shape[2] != expected_channels:
                        raise ValueError(
                            "Image tactile input must have shape [B, T, C, H, W] with "
                            f"C={expected_channels}, got {tuple(tactile_history.shape)}."
                        )
                    tactile_history = (
                        tactile_history.to(torch.float32)
                        - self.tactile_channel_mean.view(1, 1, -1, 1, 1)
                    ) / self.tactile_channel_std.view(1, 1, -1, 1, 1)

            # 获取触觉特征，VQ-VAE模式下同时获取token ID
            if self.tactile_encoder_type == "vqvae":
                tactile_feature, vqvae_token_id = self.tactile_encoder(tactile_history, return_token_id=True)
            else:
                tactile_feature = self.tactile_encoder(tactile_history)

        # 处理当前力数据
        if self.normalize_current_force_input:
            expected_current_force_dim = self.current_force_mean.numel()
            if current_force.ndim != 2 or current_force.shape[-1] != expected_current_force_dim:
                raise ValueError(
                    "Current force input must have shape [B, D] with "
                    f"D={expected_current_force_dim}, got {tuple(current_force.shape)}."
                )
            current_force = (
                current_force.to(torch.float32) - self.current_force_mean.view(1, -1)
            ) / self.current_force_std.view(1, -1)

        current_force_feature = self.current_force_encoder(current_force)

        if self.normalize_state_input:
            expected_state_dim = self.state_mean.numel()
            if state.ndim != 2 or state.shape[-1] != expected_state_dim:
                raise ValueError(
                    "State input must have shape [B, D] with "
                    f"D={expected_state_dim}, got {tuple(state.shape)}."
                )
            state = (
                state.to(torch.float32) - self.state_mean.view(1, -1)
            ) / self.state_std.view(1, -1)

        state_feature=self.state_encoder(state)

        action_feature=self.action_encoder(
            act_chunk
        )

        # ACT视觉token编码 (可选)
        visual_feature = None
        if self.use_act_visual:
            if act_visual_tokens is None:
                raise ValueError(
                    "use_act_visual=True but act_visual_tokens is None."
                )
            if act_visual_tokens.ndim not in (2, 3):
                raise ValueError(
                    "act_visual_tokens must have shape [B, S, D] or [B, D], got "
                    f"{tuple(act_visual_tokens.shape)}."
                )
            visual_feature = self.visual_encoder(act_visual_tokens)


        feature_list = []
        if tactile_feature is not None:
            feature_list.append(tactile_feature)
        feature_list += [
            current_force_feature,
            state_feature,
            action_feature,
        ]
        if visual_feature is not None:
            feature_list.append(visual_feature)
        feature = torch.cat(feature_list, dim=-1)


        # Contributions live in the fusion hidden space, whose width is configurable.
        is_token_fusion = isinstance(self.fusion, (MultiHeadFusionEncoder, ModalitySelfAttentionFusionEncoder, PairwiseCrossAttentionFusionEncoder, ResidualActionTransformerFusion))
        # Initialize contribution tensors for every fusion type.  The multihead
        # branch uses the concatenated input directly, so it previously skipped
        # the assignments below and failed when metrics were requested.
        c_t = tactile_feature
        c_cf = current_force_feature
        c_s = state_feature
        c_a = action_feature
        c_v = visual_feature
        if isinstance(self.fusion, MultiHeadFusionEncoder):
            fusion_result = self.fusion(feature, return_attention=return_feature_metrics)
        elif isinstance(self.fusion, ModalitySelfAttentionFusionEncoder):
            fusion_result = self.fusion(
                [current_force_feature, state_feature, action_feature, visual_feature],
                return_attention=return_feature_metrics,
            )
        elif isinstance(self.fusion, PairwiseCrossAttentionFusionEncoder):
            fusion_result = self.fusion(
                [current_force_feature, state_feature, action_feature, visual_feature],
                return_attention=return_feature_metrics,
            )
        elif isinstance(self.fusion, ResidualActionTransformerFusion):
            transformer_features = []
            if tactile_feature is not None:
                transformer_features.append(tactile_feature)
            transformer_features += [
                current_force_feature,
                state_feature,
                action_feature,
            ]
            if visual_feature is not None:
                transformer_features.append(visual_feature)
            fusion_result = self.fusion(
                transformer_features,
                act_chunk=act_chunk,
                return_attention=return_feature_metrics,
            )
        else:
            fusion_input_layer = self.fusion.encoder[0]
        if is_token_fusion:
            if return_feature_metrics:
                z, fusion_attention = fusion_result
            else:
                z, fusion_attention = fusion_result, None
        if is_token_fusion:
            tactile_dim = tactile_feature.shape[-1] if tactile_feature is not None else 0
            expected_fusion_dim = feature.shape[-1]
        else:
            tactile_dim = tactile_feature.shape[-1] if tactile_feature is not None else 0
        current_force_dim = current_force_feature.shape[-1]
        state_dim = state_feature.shape[-1]
        action_dim = action_feature.shape[-1]
        expected_fusion_dim = tactile_dim + current_force_dim + state_dim + action_dim
        if visual_feature is not None:
            expected_fusion_dim += visual_feature.shape[-1]
        if not is_token_fusion and fusion_input_layer.in_features != expected_fusion_dim:
            raise RuntimeError(
                "Fusion input dimensions do not match encoded modality dimensions: "
                f"{fusion_input_layer.in_features} vs {expected_fusion_dim}."
            )

        if is_token_fusion:
            fusion_weight = None
        else:
            fusion_weight = fusion_input_layer.weight
        if is_token_fusion:
            pass
        else:
            c_t = None
        if not is_token_fusion and tactile_feature is not None:
            c_t = torch.nn.functional.linear(
                tactile_feature,
                fusion_weight[:, :tactile_dim],
            )
        if not is_token_fusion:
          c_cf = torch.nn.functional.linear(
            current_force_feature,
            fusion_weight[
                :,
                tactile_dim:tactile_dim + current_force_dim,
            ],
          )
        if not is_token_fusion:
          c_s = torch.nn.functional.linear(
            state_feature,
            fusion_weight[
                :,
                tactile_dim + current_force_dim:tactile_dim + current_force_dim + state_dim,
            ],
          )
        if not is_token_fusion:
          c_a = torch.nn.functional.linear(
            action_feature,
            fusion_weight[
                :,
                tactile_dim + current_force_dim + state_dim:
                tactile_dim + current_force_dim + state_dim + action_dim,
            ],
          )
        if not is_token_fusion:
          h_pre = (c_t if c_t is not None else 0.0) + c_cf + c_s + c_a + fusion_input_layer.bias
        if not is_token_fusion and visual_feature is not None:
            c_v = torch.nn.functional.linear(
                visual_feature,
                fusion_weight[
                    :,
                    tactile_dim + current_force_dim + state_dim + action_dim:,
                ],
            )
            h_pre = h_pre + c_v

        if not is_token_fusion:
          z=self.fusion.encoder[1:](h_pre)
        timestep_gate_weights = None
        if self.use_timestep_modality_gate:
            timestep_features = []
            if tactile_feature is not None:
                timestep_features.append(tactile_feature)
            timestep_features += [
                current_force_feature,
                state_feature,
                action_feature,
            ]
            if visual_feature is not None:
                timestep_features.append(visual_feature)
            z, timestep_gate_weights = self.timestep_modality_gate(
                z, timestep_features, action_chunk=act_chunk
            )

        if self.force_film_enabled:
            force_condition = current_force_feature
            if z.ndim == 3:
                force_condition = force_condition.unsqueeze(1).expand(
                    -1, z.shape[1], -1
                )
            z = self.force_film(z, force_condition)


        delta_action=self.decoder(z)
        if self.single_step_delta:
            delta_action = delta_action[:, 0, :]

        if return_feature_metrics:
            if isinstance(self.fusion, MultiHeadFusionEncoder):
                feature_metrics = {
                    "fusion_head_weights": fusion_attention["head_weights"],
                    "fusion_gate_weights": fusion_attention["gate_weights"],
                    "fusion_modality_weights": fusion_attention["modality_weights"],
                    "fusion_head_outputs": fusion_attention["head_outputs"],
                }
                if timestep_gate_weights is not None:
                    feature_metrics["fusion_timestep_modality_gate_weights"] = (
                        timestep_gate_weights
                    )
                    feature_metrics["fusion_timestep_modality_gate_mean"] = (
                        timestep_gate_weights.mean(dim=(0, 1))
                    )
                    feature_metrics["fusion_timestep_modality_gate_entropy"] = (
                        -(timestep_gate_weights.clamp_min(1e-8)
                          * timestep_gate_weights.clamp_min(1e-8).log())
                        .sum(dim=-1)
                        .mean()
                    )
            elif isinstance(self.fusion, ModalitySelfAttentionFusionEncoder):
                feature_metrics = {
                    "fusion_attention_weights": fusion_attention["attention_weights"],
                }
            elif isinstance(self.fusion, PairwiseCrossAttentionFusionEncoder):
                feature_metrics = {
                    "fusion_pair_attention_weights": fusion_attention["pair_attention_weights"],
                    "fusion_pair_weights": fusion_attention["pair_weights"],
                }
            elif isinstance(self.fusion, ResidualActionTransformerFusion):
                feature_metrics = {}
            else:
                feature_metrics = {}
            contribution_list = []
            if c_t is not None:
                contribution_list.append(c_t.norm(p=2, dim=-1))
            contribution_list += [
                c_cf.norm(p=2, dim=-1),
                c_s.norm(p=2, dim=-1),
                c_a.norm(p=2, dim=-1),
            ]
            if visual_feature is not None:
                contribution_list.append(c_v.norm(p=2, dim=-1))
            contribution_norms = torch.stack(
                contribution_list,
                dim=-1,
            )
            contribution_ratios = contribution_norms / (
                contribution_norms.sum(dim=-1, keepdim=True) + 1e-12
            )
            contribution_index = 0
            if tactile_feature is not None:
                feature_metrics["tactile_encoder_rms"] = tactile_feature.square().mean(
                    dim=-1
                ).sqrt().mean()
                feature_metrics["tactile_contribution_ratio"] = contribution_ratios[:, contribution_index].mean()
                contribution_index += 1
            feature_metrics["current_force_encoder_rms"] = current_force_feature.square().mean(
                dim=-1
            ).sqrt().mean()
            feature_metrics["state_encoder_rms"] = state_feature.square().mean(
                dim=-1
            ).sqrt().mean()
            feature_metrics["action_encoder_rms"] = action_feature.square().mean(
                dim=-1
            ).sqrt().mean()
            feature_metrics["current_force_contribution_ratio"] = contribution_ratios[:, contribution_index].mean()
            contribution_index += 1
            feature_metrics["state_contribution_ratio"] = contribution_ratios[:, contribution_index].mean()
            contribution_index += 1
            feature_metrics["action_contribution_ratio"] = contribution_ratios[:, contribution_index].mean()
            contribution_index += 1
            if visual_feature is not None:
                feature_metrics["visual_encoder_rms"] = (
                    visual_feature.square().mean(dim=-1).sqrt().mean()
                )
                feature_metrics["visual_contribution_ratio"] = (
                    contribution_ratios[:, contribution_index].mean()
                )
            # VQ-VAE模式下添加token ID
            if vqvae_token_id is not None:
                feature_metrics["vqvae_token_id"] = vqvae_token_id
            return delta_action, feature_metrics

        return delta_action



def compute_target_delta(
    expert_action,
    act_chunk
):

    return expert_action - act_chunk


def residual_loss(
    pred_delta,
    expert_action,
    act_chunk,
    alpha=3.0,
    weight_max=5.0,
    return_difficulty=False,
):
    """
    Difficulty-based weighted SmoothL1 loss for residual policy.

    Args:
        pred_delta: [B, T, D] predicted residual action
        expert_action: [B, T, D] expert action
        act_chunk: [B, T, D] ACT action
        alpha: difficulty scaling factor (default: 3.0)
        weight_max: maximum sample weight (default: 5.0)
        return_difficulty: whether to return difficulty and weight info

    Returns:
        loss: weighted SmoothL1 loss
        (optional) difficulty, sample_weight if return_difficulty=True
    """
    # Calculate residual target
    target_delta = compute_target_delta(expert_action, act_chunk)

    # Calculate difficulty (L2 norm, detached to prevent backprop to ACT/expert)
    difficulty = torch.norm(expert_action - act_chunk, p=2, dim=-1).detach()  # [B, T]

    # Generate sample weights
    sample_weight = torch.clamp(1.0 + alpha * difficulty, min=1.0, max=weight_max)  # [B, T]

    # Compute SmoothL1 loss per element
    element_loss = nn.functional.smooth_l1_loss(pred_delta, target_delta, reduction="none")  # [B, T, D]

    # Average over action dimensions
    per_step_loss = element_loss.mean(dim=-1)  # [B, T]

    # Apply weights
    weighted_loss = per_step_loss * sample_weight  # [B, T]

    # Normalize by sum of weights
    loss = weighted_loss.sum() / (sample_weight.sum() + 1e-8)

    if return_difficulty:
        return loss, difficulty, sample_weight

    return loss


class TactileMagnitudeWeightedMSE(nn.Module):

    def __init__(
        self,
        tactile_type,
        channel_mean,
        channel_std,
        tau,
        action_horizon=30,
        action_dim=6,
        alpha=2.0,
        slope=5.0,
        eps=1e-6,
        tactile_input_already_normalized=False,
        use_weighted_loss=True,
    ):
        super().__init__()

        mean = torch.as_tensor(
            channel_mean,
            dtype=torch.float32,
        ).reshape(-1)
        std = torch.as_tensor(
            channel_std,
            dtype=torch.float32,
        ).reshape(-1)

        if mean.numel() != std.numel():
            raise ValueError(
                "channel_mean and channel_std must have the same length."
            )

        tactile_type = str(tactile_type).lower()
        if tactile_type not in {"force", "image", "vqvae"}:
            raise ValueError(
                "tactile_type must be 'force', 'image', or 'vqvae', got "
                f"{tactile_type!r}."
            )
        if tactile_type in {"force", "vqvae"} and mean.numel() != 12:
            raise ValueError(
                "Force/VQ-VAE tactile weighting expects 12 tactile channels, got "
                f"{mean.numel()}."
            )

        self.register_buffer("channel_mean", mean)
        self.register_buffer("channel_std", std)
        self.register_buffer(
            "tau",
            torch.tensor(float(tau), dtype=torch.float32),
        )
        self.register_buffer(
            "alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
        )
        self.register_buffer(
            "slope",
            torch.tensor(float(slope), dtype=torch.float32),
        )
        self.register_buffer(
            "eps",
            torch.tensor(float(eps), dtype=torch.float32),
        )
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.tactile_type = tactile_type

        self.tactile_input_already_normalized = bool(
            tactile_input_already_normalized
        )
        self.use_weighted_loss = bool(use_weighted_loss)

    @staticmethod
    def from_stats_file(
        stats_path,
        tactile_type="force",
        tau=None,
        action_horizon=30,
        action_dim=6,
        alpha=2.0,
        slope=5.0,
        eps=1e-6,
        tactile_input_already_normalized=False,
        use_weighted_loss=True,
    ):
        stats_path = Path(stats_path)
        with stats_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        resolved_tau = payload.get("tau_value") if tau is None else tau
        if resolved_tau is None:
            raise KeyError(
                f"tau_value is missing in {stats_path}."
            )

        return TactileMagnitudeWeightedMSE(
            tactile_type=tactile_type,
            channel_mean=payload["channel_mean"],
            channel_std=payload["channel_std"],
            tau=resolved_tau,
            action_horizon=action_horizon,
            action_dim=action_dim,
            alpha=alpha,
            slope=slope,
            eps=eps,
            tactile_input_already_normalized=tactile_input_already_normalized,
            use_weighted_loss=use_weighted_loss,
        )

    @staticmethod
    def reduce_weighted_losses(
        loss_per_window,
        window_weights,
        eps,
    ):
        return (
            (window_weights * loss_per_window).sum()
            / (window_weights.sum() + eps)
        )

    def _check_tensor(
        self,
        name,
        tensor,
        expected_ndim,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"{name} must have {expected_ndim} dims, got "
                f"{tensor.ndim} with shape {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or Inf values."
            )

    def _validate_tactile_history(
        self,
        tactile_history,
    ):
        expected_ndim = 3 if self.tactile_type in {"force", "vqvae"} else 5
        self._check_tensor(
            "tactile_history",
            tactile_history,
            expected_ndim=expected_ndim,
        )

        if self.tactile_type in {"force", "vqvae"}:
            tactile_channels = tactile_history.shape[-1]
            if tactile_channels != self.channel_mean.numel():
                raise ValueError(
                    "Force tactile_history last dim must match tactile stats: "
                    f"{tactile_channels} vs {self.channel_mean.numel()}."
                )
        else:
            tactile_channels = tactile_history.shape[2]
            if tactile_channels != self.channel_mean.numel():
                raise ValueError(
                    "Image tactile_history channel dim must match tactile stats: "
                    f"{tactile_channels} vs {self.channel_mean.numel()}."
                )

        return tactile_channels

    def _compute_force_window_magnitude(
        self,
        tactile_float,
    ):
        if self.tactile_input_already_normalized:
            tactile_for_weight = tactile_float
        else:
            tactile_for_weight = (
                tactile_float
                - self.channel_mean.view(1, 1, -1)
            ) / (
                self.channel_std.view(1, 1, -1)
                + self.eps
            )

        magnitude = torch.sqrt(
            tactile_for_weight.square().mean(dim=(1, 2))
            + self.eps
        )

        return magnitude

    def _compute_image_window_magnitude(
        self,
        tactile_float,
    ):
        if self.tactile_input_already_normalized:
            second_moment = tactile_float.square().mean(
                dim=(1, 2, 3, 4)
            )
        else:
            normalizer = self.channel_std + self.eps
            sum_of_squares = torch.zeros(
                tactile_float.shape[0],
                dtype=torch.float32,
                device=tactile_float.device,
            )
            for channel_idx in range(tactile_float.shape[2]):
                normalized_channel = (
                    tactile_float[:, :, channel_idx]
                    - self.channel_mean[channel_idx]
                ) / normalizer[channel_idx]
                sum_of_squares = sum_of_squares + normalized_channel.square().sum(
                    dim=(1, 2, 3)
                )

            sample_size = (
                tactile_float.shape[1]
                * tactile_float.shape[2]
                * tactile_float.shape[3]
                * tactile_float.shape[4]
            )
            second_moment = sum_of_squares / float(sample_size)

        magnitude = torch.sqrt(
            second_moment + self.eps
        )
        return magnitude

    def compute_window_magnitude(
        self,
        tactile_history,
    ):
        self._validate_tactile_history(tactile_history)

        tactile_float = tactile_history.detach().to(torch.float32)

        if self.tactile_type in {"force", "vqvae"}:
            magnitude = self._compute_force_window_magnitude(
                tactile_float
            )
        else:
            magnitude = self._compute_image_window_magnitude(
                tactile_float
            )

        return magnitude

    def compute_window_weights(
        self,
        tactile_history,
    ):
        with torch.no_grad():
            magnitude = self.compute_window_magnitude(
                tactile_history
            )
            weights = 1.0 + self.alpha * torch.sigmoid(
                self.slope
                * (magnitude - self.tau)
                / (self.tau + self.eps)
            )
        return magnitude.detach(), weights.detach()

    def forward(
        self,
        pred_delta,
        target_delta,
        tactile_history,
        act_chunk=None,
        expert_action=None,
        feature_metrics=None,
    ):
        self._check_tensor(
            "pred_delta",
            pred_delta,
            expected_ndim=3,
        )
        self._check_tensor(
            "target_delta",
            target_delta,
            expected_ndim=3,
        )

        if pred_delta.shape != target_delta.shape:
            raise ValueError(
                "pred_delta and target_delta must match: "
                f"{tuple(pred_delta.shape)} vs {tuple(target_delta.shape)}."
            )

        if pred_delta.shape[1:] != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "Expected pred_delta/target_delta shape "
                f"[B, {self.action_horizon}, {self.action_dim}], got "
                f"{tuple(pred_delta.shape)}."
            )

        if tactile_history is not None:
            if tactile_history.shape[0] != pred_delta.shape[0]:
                raise ValueError(
                    "Batch size mismatch between tactile_history and pred_delta."
                )
            self._validate_tactile_history(tactile_history)

        if act_chunk is not None:
            self._check_tensor(
                "act_chunk",
                act_chunk,
                expected_ndim=3,
            )
            if act_chunk.shape != pred_delta.shape:
                raise ValueError(
                    "act_chunk must match pred_delta shape: "
                    f"{tuple(act_chunk.shape)} vs {tuple(pred_delta.shape)}."
                )

        if expert_action is not None:
            self._check_tensor(
                "expert_action",
                expert_action,
                expected_ndim=3,
            )
            if expert_action.shape != pred_delta.shape:
                raise ValueError(
                    "expert_action must match pred_delta shape: "
                    f"{tuple(expert_action.shape)} vs {tuple(pred_delta.shape)}."
                )

        pred_float = pred_delta.to(torch.float32)
        target_float = target_delta.to(torch.float32)

        squared_error = (
            pred_float - target_float
        ).square()
        loss_per_window = squared_error.mean(
            dim=(1, 2)
        )

        # VQ-VAE模式：直接使用unweighted loss，只返回token ID和modality contribution
        if self.tactile_type == "vqvae":
            objective_loss = loss_per_window.mean()

            metrics = {}

            # 从feature_metrics复制modality contribution和token ID
            if feature_metrics is not None:
                # Modality contribution
                if "tactile_contribution_ratio" in feature_metrics:
                    metrics["tactile_contribution_ratio"] = feature_metrics["tactile_contribution_ratio"]
                if "current_force_contribution_ratio" in feature_metrics:
                    metrics["current_force_contribution_ratio"] = feature_metrics["current_force_contribution_ratio"]
                if "state_contribution_ratio" in feature_metrics:
                    metrics["state_contribution_ratio"] = feature_metrics["state_contribution_ratio"]
                if "action_contribution_ratio" in feature_metrics:
                    metrics["action_contribution_ratio"] = feature_metrics["action_contribution_ratio"]
                if "visual_contribution_ratio" in feature_metrics:
                    metrics["visual_contribution_ratio"] = feature_metrics["visual_contribution_ratio"]
                if "visual_encoder_rms" in feature_metrics:
                    metrics["visual_encoder_rms"] = feature_metrics["visual_encoder_rms"]
                if "fusion_head_weights" in feature_metrics:
                    metrics["fusion_head_weights"] = feature_metrics["fusion_head_weights"]
                if "fusion_gate_weights" in feature_metrics:
                    metrics["fusion_gate_weights"] = feature_metrics["fusion_gate_weights"]
                if "fusion_modality_weights" in feature_metrics:
                    metrics["fusion_modality_weights"] = feature_metrics["fusion_modality_weights"]
                if "fusion_head_outputs" in feature_metrics:
                    metrics["fusion_head_outputs"] = feature_metrics["fusion_head_outputs"]
                if "fusion_timestep_modality_gate_weights" in feature_metrics:
                    metrics["fusion_timestep_modality_gate_weights"] = feature_metrics[
                        "fusion_timestep_modality_gate_weights"
                    ]
                for key in (
                    "fusion_timestep_modality_gate_mean",
                    "fusion_timestep_modality_gate_entropy",
                ):
                    if key in feature_metrics:
                        metrics[key] = feature_metrics[key]
                if "fusion_attention_weights" in feature_metrics:
                    metrics["fusion_attention_weights"] = feature_metrics["fusion_attention_weights"]
                if "fusion_pair_attention_weights" in feature_metrics:
                    metrics["fusion_pair_attention_weights"] = feature_metrics["fusion_pair_attention_weights"]
                if "fusion_pair_weights" in feature_metrics:
                    metrics["fusion_pair_weights"] = feature_metrics["fusion_pair_weights"]

                # VQ-VAE token ID
                if "vqvae_token_id" in feature_metrics:
                    metrics["vqvae_token_id"] = feature_metrics["vqvae_token_id"]

            return objective_loss, metrics

        # Force/Image模式：使用原有的magnitude-based weighting
        if tactile_history is None:
            raise ValueError(
                "tactile_history is required for force/image tactile loss."
            )
        tactile_magnitude, window_weights = self.compute_window_weights(
            tactile_history
        )

        if self.use_weighted_loss:
            objective_loss = self.reduce_weighted_losses(
                loss_per_window=loss_per_window,
                window_weights=window_weights,
                eps=self.eps,
            )
        else:
            objective_loss = loss_per_window.mean()

        # 只在use_weighted_loss=True时计算完整的指标
        metrics = {}

        if self.use_weighted_loss:
            weighted_loss = self.reduce_weighted_losses(
                loss_per_window=loss_per_window,
                window_weights=window_weights,
                eps=self.eps,
            )
            unweighted_loss = loss_per_window.mean()

            high_mask = tactile_magnitude >= self.tau
            low_mask = ~high_mask

            if act_chunk is not None and expert_action is not None:
                final_action_error = (
                    pred_float
                    + act_chunk.to(torch.float32)
                    - expert_action.to(torch.float32)
                ).square().mean(dim=(1, 2))
                final_action_mse = final_action_error.mean()
            else:
                final_action_mse = unweighted_loss

            zero = torch.zeros(
                (),
                dtype=torch.float32,
                device=pred_delta.device,
            )
            high_loss = (
                loss_per_window[high_mask].mean()
                if high_mask.any()
                else zero
            )
            low_loss = (
                loss_per_window[low_mask].mean()
                if low_mask.any()
                else zero
            )

            metrics = {
                "weighted_loss": weighted_loss.detach(),
                "unweighted_loss": unweighted_loss.detach(),
                "tactile_magnitude_mean": tactile_magnitude.mean().detach(),
                "tactile_magnitude_min": tactile_magnitude.min().detach(),
                "tactile_magnitude_max": tactile_magnitude.max().detach(),
                "weight_mean": window_weights.mean().detach(),
                "weight_min": window_weights.min().detach(),
                "weight_max": window_weights.max().detach(),
                "weight_p50": torch.quantile(
                    window_weights,
                    0.50,
                ).detach(),
                "weight_p90": torch.quantile(
                    window_weights,
                    0.90,
                ).detach(),
                "weight_p95": torch.quantile(
                    window_weights,
                    0.95,
                ).detach(),
                "fraction_above_tau": high_mask.to(
                    torch.float32
                ).mean().detach(),
                "high_weight_loss": high_loss.detach(),
                "low_weight_loss": low_loss.detach(),
                "high_magnitude_mse": high_loss.detach(),
                "low_magnitude_mse": low_loss.detach(),
                "final_action_mse": final_action_mse.detach(),
                "tau": self.tau.detach(),
            }

        # 添加modality contribution（如果有）
        if feature_metrics is not None:
            for key in (
                "fusion_head_weights",
                "fusion_gate_weights",
                "fusion_modality_weights",
                "fusion_head_outputs",
                "fusion_timestep_modality_gate_weights",
                "fusion_timestep_modality_gate_mean",
                "fusion_timestep_modality_gate_entropy",
            ):
                if key in feature_metrics:
                    metrics[key] = feature_metrics[key]
            metrics["tactile_contribution_ratio"] = feature_metrics.get("tactile_contribution_ratio", torch.tensor(0.0))
            metrics["current_force_contribution_ratio"] = feature_metrics.get("current_force_contribution_ratio", torch.tensor(0.0))
            metrics["state_contribution_ratio"] = feature_metrics.get("state_contribution_ratio", torch.tensor(0.0))
            metrics["action_contribution_ratio"] = feature_metrics.get("action_contribution_ratio", torch.tensor(0.0))
            metrics["visual_contribution_ratio"] = feature_metrics.get("visual_contribution_ratio", torch.tensor(0.0))
            metrics["visual_encoder_rms"] = feature_metrics.get("visual_encoder_rms", torch.tensor(0.0))

        return objective_loss, metrics

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 16

    print("=" * 50)
    print("测试 TactileForceEncoder (1D卷积)")
    print("=" * 50)

    # 测试合力编码器
    tactile_history_force = torch.randn(B, 16, 6).to(device)
    state = torch.randn(B, 6).to(device)
    action_horizon = 30
    action_dim = 6
    act_chunk = torch.randn(B, action_horizon, action_dim).to(device)
    expert_action = torch.randn(B, action_horizon, action_dim).to(device)

    model_force = TactileResidualACT(
        tactile_encoder_type="force",
        action_horizon=action_horizon,
        action_dim=action_dim,
    ).to(device)
    print("device:", device)

    with torch.no_grad():
        delta_pred = model_force(tactile_history_force, state, act_chunk)

    print("输入触觉:", tactile_history_force.shape)
    print("delta_pred:", delta_pred.shape)

    loss = residual_loss(delta_pred, expert_action, act_chunk)
    print("loss:", loss.item())

    params = sum(p.numel() for p in model_force.parameters() if p.requires_grad)
    print("params:", params / 1e6, "M")

    print("\n" + "=" * 50)
    print("测试 TactileImageEncoder (2D卷积)")
    print("=" * 50)

    # 测试图像编码器
    tactile_history_image = torch.randn(B, 10, 6, 288, 384).to(device)

    model_image = TactileResidualACT(
        tactile_encoder_type="image",
        action_horizon=action_horizon,
        action_dim=action_dim,
    ).to(device)

    with torch.no_grad():
        delta_pred_image = model_image(tactile_history_image, state, act_chunk)

    print("输入触觉:", tactile_history_image.shape)
    print("delta_pred:", delta_pred_image.shape)

    loss_image = residual_loss(delta_pred_image, expert_action, act_chunk)
    print("loss:", loss_image.item())

    params_image = sum(p.numel() for p in model_image.parameters() if p.requires_grad)
    print("params:", params_image / 1e6, "M")
