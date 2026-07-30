import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None


class DeformEncoder(nn.Module):
    """
    多通道形变图编码器。

    输入:
        [B, C, 240, 240]
    输出:
        [B, 128, 15, 15]
    """

    def __init__(self, input_channels=3, latent_channels=128):
        super().__init__()

        if resnet18 is None:
            raise ImportError("torchvision is required to build DeformEncoder.")

        if latent_channels != 128:
            raise ValueError("The blueprint fixes latent_channels to 128.")
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1.")

        backbone = resnet18(weights=None)

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.proj2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = backbone.layer3
        self.proj3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)    # [B, 64, 60, 60]
        x = self.layer1(x)  # [B, 64, 60, 60]
        x = self.layer2(x)  # [B, 128, 30, 30]
        x = self.proj2(x)   # [B, 128, 30, 30]
        x = self.layer3(x)  # [B, 256, 15, 15]
        x = self.proj3(x)   # [B, 128, 15, 15]
        return x


class DeformDecoder(nn.Module):
    """
    形变图解码器，仅用于自编码器预训练。

    输入:
        [B, 128, 15, 15]
    输出:
        [B, C, 240, 240]
    """

    def __init__(self, output_channels=3, output_activation="identity"):
        super().__init__()

        if output_activation not in {"identity", "sigmoid", "tanh"}:
            raise ValueError(
                "output_activation must be one of: identity, sigmoid, tanh."
            )
        if output_channels < 1:
            raise ValueError("output_channels must be >= 1.")

        self.up1 = nn.Upsample(scale_factor=4, mode="nearest")
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Upsample(scale_factor=4, mode="nearest")
        self.conv2 = nn.Conv2d(
            64,
            output_channels,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.output_activation = output_activation

    def forward(self, z):
        x = self.up1(z)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)

        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)

        return x


class DeformAutoencoder(nn.Module):
    """
    基于图纸重建的 deformation autoencoder。
    """

    def __init__(
        self,
        input_channels=3,
        latent_channels=128,
        output_activation="identity",
    ):
        super().__init__()
        self.encoder = DeformEncoder(
            input_channels=input_channels,
            latent_channels=latent_channels,
        )
        self.decoder = DeformDecoder(
            output_channels=input_channels,
            output_activation=output_activation,
        )

    def forward(self, x, return_latent=False):
        z = self.encoder(x)
        recon = self.decoder(z)
        if return_latent:
            return recon, z
        return recon


def _spatial_gradient(x):
    grad_x = x[..., :, 1:] - x[..., :, :-1]
    grad_y = x[..., 1:, :] - x[..., :-1, :]
    return grad_x, grad_y


def deform_reconstruction_loss(
    pred,
    target,
    pixel_loss_type="mse",
    edge_weight=0.0,
    sample_weight=None,
):
    """
    图纸中的重建损失:
        L = L_pixel + lambda_edge * L_edge
    """

    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} vs {target.shape}."
        )

    if pixel_loss_type == "mse":
        pixel_map = (pred - target) ** 2
    elif pixel_loss_type == "l1":
        pixel_map = (pred - target).abs()
    elif pixel_loss_type == "huber":
        pixel_map = nn.functional.smooth_l1_loss(
            pred, target, reduction="none"
        )
    else:
        raise ValueError("pixel_loss_type must be one of: mse, l1, huber.")

    pixel_loss = pixel_map.mean(dim=(1, 2, 3))

    if sample_weight is not None:
        sample_weight = sample_weight.to(pixel_loss.device).reshape(-1)
        if sample_weight.shape[0] != pixel_loss.shape[0]:
            raise ValueError("sample_weight must have shape [B].")
        pixel_loss = pixel_loss * sample_weight

    pixel_loss = pixel_loss.mean()

    edge_loss = pred.new_tensor(0.0)
    if edge_weight > 0:
        pred_grad_x, pred_grad_y = _spatial_gradient(pred)
        target_grad_x, target_grad_y = _spatial_gradient(target)
        edge_map_x = (pred_grad_x - target_grad_x).abs().mean(dim=(1, 2, 3))
        edge_map_y = (pred_grad_y - target_grad_y).abs().mean(dim=(1, 2, 3))
        edge_loss = edge_map_x + edge_map_y
        if sample_weight is not None:
            edge_loss = edge_loss * sample_weight
        edge_loss = edge_loss.mean()

    total_loss = pixel_loss + edge_weight * edge_loss
    return {
        "loss": total_loss,
        "pixel_loss": pixel_loss,
        "edge_loss": edge_loss,
    }
