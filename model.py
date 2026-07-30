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

    def __init__(self, in_channels=6):
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

        self.fc = nn.Linear(128, 256)


    def forward(self,x):

        # B,T,C,H,W
        B,T,C,H,W=x.shape

        x=x.reshape(B*T,C,H,W)

        x=self.encoder(x)

        x=x.flatten(1)

        x=self.fc(x)

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

        return x



class StateEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(6,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU()
        )


    def forward(self,x):
        return self.net(x)



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
            nn.ReLU()
        )


    def forward(self,x):

        B,T,D=x.shape

        x=x.reshape(B,T*D)

        return self.net(x)



class FusionEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder=nn.Sequential(

            nn.Linear(
                256+128+256,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU()
        )


    def forward(self,x):

        return self.encoder(x)



class ResidualDecoder(nn.Module):

    def __init__(
        self,
        input_dim=256,
        hidden_dim=256,
        action_horizon=30,
        action_dim=6,
    ):

        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)

        self.decoder=nn.Sequential(

            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                self.action_horizon * self.action_dim
            )

        )


    def forward(self,x):

        B=x.shape[0]

        x=self.decoder(x)

        return x.reshape(
            B,
            self.action_horizon,
            self.action_dim,
        )



class TactileResidualACT(nn.Module):

    def __init__(
        self,
        tactile_encoder_type="force",
        action_horizon=30,
        action_dim=6,
        action_encoder_cfg=None,
        decoder_cfg=None,
    ):
        """
        Args:
            tactile_encoder_type: "force" 或 "image"
                - "force": 使用 TactileForceEncoder (1D卷积，处理12维合力)
                - "image": 使用 TactileImageEncoder (2D卷积，处理触觉图像)
        """

        super().__init__()

        # 根据配置选择触觉编码器
        if tactile_encoder_type == "force":
            # 左右合力各6维，拼接后12维
            self.tactile_encoder = TactileForceEncoder(input_dim=12)
        elif tactile_encoder_type == "image":
            self.tactile_encoder = TactileImageEncoder()
        else:
            raise ValueError(f"Unknown tactile_encoder_type: {tactile_encoder_type}. Choose 'force' or 'image'.")

        self.state_encoder = StateEncoder()

        action_encoder_cfg = action_encoder_cfg or {}
        decoder_cfg = decoder_cfg or {}
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

        self.fusion = FusionEncoder()

        decoder_output_dim = int(
            decoder_cfg.get(
                "output_dim",
                expected_action_input_dim,
            )
        )
        if decoder_output_dim != expected_action_input_dim:
            raise ValueError(
                "decoder.output_dim must equal "
                "decoder.action_horizon * decoder.action_dim: "
                f"{decoder_output_dim} vs "
                f"{self.action_horizon} * {self.action_dim}."
            )

        self.decoder = ResidualDecoder(
            input_dim=int(decoder_cfg.get("input_dim", 256)),
            hidden_dim=int(decoder_cfg.get("hidden_dim", 256)),
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
        )


    def forward(
        self,
        tactile_history,
        state,
        act_chunk
    ):

        tactile_feature=self.tactile_encoder(
            tactile_history
        )

        state_feature=self.state_encoder(
            state
        )

        action_feature=self.action_encoder(
            act_chunk
        )


        feature=torch.cat(
            [
                tactile_feature,
                state_feature,
                action_feature
            ],
            dim=-1
        )


        z=self.fusion(feature)


        delta_action=self.decoder(z)


        return delta_action



def compute_target_delta(
    expert_action,
    act_chunk
):

    return expert_action - act_chunk


def residual_loss(
    pred_delta,
    expert_action,
    act_chunk
):

    target_delta = compute_target_delta(
        expert_action,
        act_chunk
    )

    loss=nn.functional.mse_loss(
        pred_delta,
        target_delta
    )

    return loss


class TactileMagnitudeWeightedMSE(nn.Module):

    def __init__(
        self,
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

        if mean.numel() != 12:
            raise ValueError(
                f"Expected 12 tactile channels, got {mean.numel()}."
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

        self.tactile_input_already_normalized = bool(
            tactile_input_already_normalized
        )
        self.use_weighted_loss = bool(use_weighted_loss)

    @staticmethod
    def from_stats_file(
        stats_path,
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

    def compute_window_magnitude(
        self,
        tactile_history,
    ):
        self._check_tensor(
            "tactile_history",
            tactile_history,
            expected_ndim=3,
        )

        if tactile_history.shape[-1] != self.channel_mean.numel():
            raise ValueError(
                "tactile_history last dim must match tactile stats: "
                f"{tactile_history.shape[-1]} vs {self.channel_mean.numel()}."
            )

        tactile_float = tactile_history.detach().to(torch.float32)

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

        return magnitude, tactile_for_weight

    def compute_window_weights(
        self,
        tactile_history,
    ):
        with torch.no_grad():
            magnitude, _ = self.compute_window_magnitude(
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

        if tactile_history.shape[0] != pred_delta.shape[0]:
            raise ValueError(
                "Batch size mismatch between tactile_history and pred_delta."
            )

        pred_float = pred_delta.to(torch.float32)
        target_float = target_delta.to(torch.float32)

        squared_error = (
            pred_float - target_float
        ).square()
        loss_per_window = squared_error.mean(
            dim=(1, 2)
        )

        tactile_magnitude, window_weights = (
            self.compute_window_weights(
                tactile_history
            )
        )

        if self.use_weighted_loss:
            weighted_loss = self.reduce_weighted_losses(
                loss_per_window=loss_per_window,
                window_weights=window_weights,
                eps=self.eps,
            )
        else:
            window_weights = torch.ones_like(
                tactile_magnitude
            )
            weighted_loss = loss_per_window.mean()

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

        return weighted_loss, metrics

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
