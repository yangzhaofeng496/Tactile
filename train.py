import argparse
import copy
import csv
import json
import os
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import wandb
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from dataloader.dataloader import (
    build_augmented_loaders,
    build_base_dataset,
    build_normal_dataloaders,
    load_lerobot_policy,
    load_yaml,
    set_seed,
)
from model import (
    TactileMagnitudeWeightedMSE,
    TactileResidualACT,
    compute_target_delta,
    residual_loss,
)


def safe_wandb_log(payload):
    if wandb.run is not None:
        wandb.log(payload)


def safe_wandb_save(path):
    if wandb.run is not None:
        wandb.save(str(path))


def safe_wandb_watch(model, **kwargs):
    if wandb.run is not None:
        wandb.watch(model, **kwargs)


def safe_wandb_finish():
    if wandb.run is not None:
        wandb.finish()


def log_compact_wandb_metrics(step, metrics):
    if wandb.run is None:
        return

    payload = {}

    # Objective loss
    if "objective_loss" in metrics:
        payload["train/objective_loss"] = float(metrics["objective_loss"])

    # 只在指标存在时才添加
    if "weighted_loss" in metrics:
        payload["train/weight_loss"] = float(metrics["weighted_loss"])
    if "unweighted_loss" in metrics:
        payload["train/unweight_loss"] = float(metrics["unweighted_loss"])

    if "fusion_head_weights" in metrics:
        for i, value in enumerate(metrics["fusion_head_weights"]):
            payload[f"fusion/head_weight/head_{i}"] = float(value)
    if "fusion_modality_weights" in metrics:
        names = ["current_force", "state", "action_chunk", "visual"]
        weights = metrics["fusion_modality_weights"]
        for head_idx, row in enumerate(weights):
            for name, value in zip(names, row):
                payload[f"fusion/head_modality_weight/head_{head_idx}/{name}"] = float(value)
    if "visual_encoder_rms" in metrics:
        payload["train/act_encoder_latent_rms"] = float(metrics["visual_encoder_rms"])

    # VQ-VAE token ID - 统计每个token ID的出现次数
    if "vqvae_token_id" in metrics:
        token_id = metrics["vqvae_token_id"]
        if isinstance(token_id, torch.Tensor):
            # 统计每个token ID的出现次数
            unique_ids, counts = torch.unique(token_id, return_counts=True)
            for token_id_val, count in zip(unique_ids.cpu().tolist(), counts.cpu().tolist()):
                payload[f"train/vqvae_token_id/token_{token_id_val}"] = int(count)

    # Action dimension MSE
    for action_dim in range(6):
        key = f"action_dim{action_dim}_mse"
        if key in metrics:
            payload[f"train/action_dim_mse/axis_{action_dim}"] = float(metrics[key])

    # Delta action means
    for action_dim in range(6):
        key = f"pred_delta_dim{action_dim}_mean"
        if key in metrics:
            payload[f"train/delta_action/axis_{action_dim}"] = float(metrics[key])

    wandb.log(payload, step=int(step))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the tactile residual ACT model."
    )
    parser.add_argument(
        "--dataloader-config",
        type=Path,
        default=Path("dataloader/tactile_dataloader.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model_config.yaml"),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run one diagnostic pass without full training.",
    )
    parser.add_argument(
        "--ablate-modalities",
        nargs="*",
        default=None,
        choices=["current_force", "state", "visual"],
        help="Fixed modalities to zero during train/val/test.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override training.checkpoint_dir for parallel experiments.",
    )
    return parser.parse_args()


def move_batch_to_device(batch, device):
    keys = [
        "tactile_history",
        "current_force",
        "observation.state",
        "act_chunk",
        "expert_action",
        "delta_action_target",
        "act_visual_tokens",
    ]
    output = dict(batch)
    for key in keys:
        value = output.get(key)
        if isinstance(value, torch.Tensor):
            output[key] = value.to(
                device=device,
                non_blocking=True,
            )
    return output


def apply_fixed_modality_ablation(batch, ablate_modalities):
    """Zero selected conditioning modalities for a controlled ablation run."""
    modalities = set(ablate_modalities or ())
    if "current_force" in modalities:
        batch["current_force"] = torch.zeros_like(batch["current_force"])
    if "state" in modalities:
        batch["observation.state"] = torch.zeros_like(batch["observation.state"])
    if "visual" in modalities and batch.get("act_visual_tokens") is not None:
        batch["act_visual_tokens"] = torch.zeros_like(batch["act_visual_tokens"])
    return batch


def init_metric_accumulator():
    return {
        "count": 0,
        "objective_loss_sum": 0.0,
        "weighted_loss_sum": 0.0,
        "unweighted_loss_sum": 0.0,
        "tactile_magnitude_mean_sum": 0.0,
        "tactile_magnitude_min_sum": 0.0,
        "tactile_magnitude_max_sum": 0.0,
        "weight_mean_sum": 0.0,
        "weight_min_sum": 0.0,
        "weight_max_sum": 0.0,
        "weight_p50_sum": 0.0,
        "weight_p90_sum": 0.0,
        "weight_p95_sum": 0.0,
        "fraction_above_tau_sum": 0.0,
        "high_weight_loss_sum": 0.0,
        "low_weight_loss_sum": 0.0,
        "high_magnitude_mse_sum": 0.0,
        "low_magnitude_mse_sum": 0.0,
        "final_action_mse_sum": 0.0,
        "grad_norm_sum": 0.0,
        "pred_delta_abs_mean_sum": 0.0,
        "pred_delta_abs_max_sum": 0.0,
        "pred_delta_std_sum": 0.0,
        "target_delta_abs_mean_sum": 0.0,
        "target_delta_abs_max_sum": 0.0,
        "target_delta_std_sum": 0.0,
        "loss_per_window_mean_sum": 0.0,
        "loss_per_window_max_sum": 0.0,
        "loss_per_window_p90_sum": 0.0,
        "loss_per_window_p95_sum": 0.0,
        "act_expert_mse_sum": 0.0,
        "high_magnitude_final_action_mse_sum": 0.0,
        "low_magnitude_final_action_mse_sum": 0.0,
        "high_magnitude_target_delta_abs_mean_sum": 0.0,
        "low_magnitude_target_delta_abs_mean_sum": 0.0,
        "tactile_input_abs_mean_sum": 0.0,
        "tactile_input_abs_max_sum": 0.0,
        "target_delta_dim0_std_sum": 0.0,
        "target_delta_dim1_std_sum": 0.0,
        "target_delta_dim2_std_sum": 0.0,
        "target_delta_dim3_std_sum": 0.0,
        "target_delta_dim4_std_sum": 0.0,
        "target_delta_dim5_std_sum": 0.0,
        "pred_delta_dim0_std_sum": 0.0,
        "pred_delta_dim1_std_sum": 0.0,
        "pred_delta_dim2_std_sum": 0.0,
        "pred_delta_dim3_std_sum": 0.0,
        "pred_delta_dim4_std_sum": 0.0,
        "pred_delta_dim5_std_sum": 0.0,
        "pred_delta_dim0_mean_sum": 0.0,
        "pred_delta_dim1_mean_sum": 0.0,
        "pred_delta_dim2_mean_sum": 0.0,
        "pred_delta_dim3_mean_sum": 0.0,
        "pred_delta_dim4_mean_sum": 0.0,
        "pred_delta_dim5_mean_sum": 0.0,
        "action_dim0_mse_sum": 0.0,
        "action_dim1_mse_sum": 0.0,
        "action_dim2_mse_sum": 0.0,
        "action_dim3_mse_sum": 0.0,
        "action_dim4_mse_sum": 0.0,
        "action_dim5_mse_sum": 0.0,
    }


def update_metric_accumulator(accumulator, metrics, batch_size):
    accumulator["count"] += batch_size
    for key in list(accumulator.keys()):
        if key == "count":
            continue
        metric_name = key[:-4]
        # 只累加实际存在的指标，且跳过tensor类型（如vqvae_token_id）
        if metric_name in metrics:
            value = metrics[metric_name]
            # 跳过多元素tensor（如vqvae_token_id batch）
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                continue
            accumulator[key] += float(value) * batch_size


def finalize_metric_accumulator(accumulator):
    count = max(accumulator["count"], 1)
    result = {}
    for key, value in accumulator.items():
        if key == "count":
            continue
        result[key[:-4]] = value / count
    return result


def format_metrics_for_log(prefix, metrics):
    return {
        f"{prefix}/{key}": float(value)
        for key, value in metrics.items()
    }


def resolve_device(training_cfg):
    device_name = training_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_tactile_stats(stats_path):
    stats_path = Path(stats_path)
    with stats_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_tactile_stats_path(
    training_cfg,
    tactile_type,
):
    stats_paths = training_cfg.get("tactile_stats_paths")
    if stats_paths is not None:
        if not isinstance(stats_paths, dict):
            raise ValueError(
                "training.tactile_stats_paths must be a mapping with "
                "'force'/'image' keys."
            )
        stats_path = stats_paths.get(tactile_type)
        if stats_path:
            return stats_path, "typed"
        raise ValueError(
            "training.tactile_stats_paths is configured, but "
            f"training.tactile_stats_paths.{tactile_type} is missing or empty."
        )

    legacy_path = training_cfg.get("tactile_stats_path")
    if legacy_path:
        warnings.warn(
            "training.tactile_stats_path is deprecated; "
            "prefer training.tactile_stats_paths.<tactile_type>. "
            f"Using legacy path for tactile_type={tactile_type!r}.",
            stacklevel=2,
        )
        return legacy_path, "legacy"

    raise ValueError(
        "Missing tactile stats path. Set "
        "training.tactile_stats_paths."
        f"{tactile_type} or legacy training.tactile_stats_path."
    )


def build_tactile_criterion(
    training_cfg,
    tactile_type,
    tactile_channels,
    action_horizon,
    action_dim,
):
    use_weighted = bool(
        training_cfg.get("use_tactile_weighted_loss", False)
    )
    stats_path, stats_path_source = resolve_tactile_stats_path(
        training_cfg=training_cfg,
        tactile_type=tactile_type,
    )

    stats_payload = load_tactile_stats(stats_path)
    stats_channels = len(stats_payload["channel_mean"])
    if len(stats_payload["channel_std"]) != stats_channels:
        raise ValueError(
            f"Tactile stats file {stats_path} has mismatched channel_mean/channel_std lengths."
        )
    if stats_channels != tactile_channels:
        raise ValueError(
            f"Tactile stats file {stats_path} has {stats_channels} channels, "
            f"but tactile_type={tactile_type!r} expects {tactile_channels}."
        )

    criterion = TactileMagnitudeWeightedMSE(
        tactile_type=tactile_type,
        channel_mean=stats_payload["channel_mean"],
        channel_std=stats_payload["channel_std"],
        tau=float(
            training_cfg.get(
                "tactile_weight_tau",
                stats_payload["tau_value"],
            )
        ),
        action_horizon=action_horizon,
        action_dim=action_dim,
        alpha=float(
            training_cfg.get("tactile_weight_alpha", 2.0)
        ),
        slope=float(
            training_cfg.get("tactile_weight_slope", 5.0)
        ),
        eps=float(
            training_cfg.get("tactile_weight_eps", 1e-6)
        ),
        tactile_input_already_normalized=bool(
            training_cfg.get(
                "tactile_input_already_normalized",
                False,
            )
        ),
        use_weighted_loss=use_weighted,
    )
    criterion_metadata = {
        "tactile_stats_path": str(stats_path),
        "tactile_stats_path_source": stats_path_source,
        "stats_payload": stats_payload,
    }
    return criterion, criterion_metadata


def validate_force_channel_order(
    dataloader_config,
    model_config,
    stats_payload,
):
    data_order = dataloader_config["dataset"]["keys"].get(
        "tactile_force_channel_order"
    )
    model_order = model_config["tactile_encoder"]["force"].get(
        "channel_order"
    )
    stats_order = stats_payload.get("channel_names")
    orders = {
        "dataloader tactile_force_channel_order": data_order,
        "model tactile_encoder.force.channel_order": model_order,
        "stats channel_names": stats_order,
    }
    for source, order in orders.items():
        if not isinstance(order, list) or not order or any(
            not isinstance(name, str) or not name for name in order
        ):
            raise ValueError(
                f"{source} must be a non-empty list of channel names."
            )
        if len(order) != len(set(order)):
            raise ValueError(f"{source} contains duplicate channel names.")

    if data_order != model_order or data_order != stats_order:
        raise ValueError(
            "Force channel order mismatch. "
            f"dataloader={data_order}, model={model_order}, stats={stats_order}."
        )
    return data_order


def load_and_validate_state_stats(model_config, action_dim):
    state_cfg = model_config["state_encoder"]
    state_dim = int(state_cfg["input_dim"])
    if state_dim != action_dim:
        raise ValueError(
            "state_encoder.input_dim must match the per-step action dimension: "
            f"{state_dim} vs {action_dim}."
        )
    stats_path = state_cfg.get("stats_path")
    if not stats_path:
        raise ValueError("state_encoder.stats_path is required.")
    stats = load_tactile_stats(stats_path)
    expected_order = state_cfg.get("channel_order")
    if stats.get("channel_names") != expected_order:
        raise ValueError(
            "State channel order mismatch between model config and statistics: "
            f"model={expected_order}, stats={stats.get('channel_names')}."
        )
    if len(stats.get("mean", [])) != state_dim or len(
        stats.get("std", [])
    ) != state_dim:
        raise ValueError(
            f"State statistics must contain {state_dim} means and standard deviations."
        )
    return stats


def resolve_current_force_normalization(model_config, criterion_metadata):
    cfg = model_config.get("current_force_encoder", {})
    if not bool(cfg.get("normalize_input", False)):
        return None, None, False
    stats = criterion_metadata["stats_payload"]
    return (
        torch.tensor(stats["channel_mean"], dtype=torch.float32),
        torch.tensor(stats["channel_std"], dtype=torch.float32),
        True,
    )


def summarize_batch_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            # 对于多元素tensor（如vqvae_token_id），保留为tensor
            if value.numel() > 1:
                result[key] = value.detach()
            else:
                result[key] = float(value.detach().to(torch.float32).item())
        else:
            result[key] = value
    return result


def compute_grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(torch.float32)
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5


def add_action_noise(action, std=0.0):
    """Add zero-mean Gaussian noise to an action tensor without mutating it."""
    std = float(std)
    if std <= 0.0:
        return action
    return action + torch.randn_like(action) * std


def add_relative_gaussian_noise(values, relative_std=0.0):
    """Add batch-channel-scaled Gaussian noise without mutating the input."""
    relative_std = float(relative_std)
    if relative_std <= 0.0:
        return values
    scale = values.detach().to(torch.float32).std(dim=0, keepdim=True)
    scale = scale.clamp_min(1e-6).to(dtype=values.dtype, device=values.device)
    return values + torch.randn_like(values) * scale * relative_std


def apply_modality_dropout(action_chunk, act_visual_tokens, drop_mask):
    """Replace selected ACT reference modalities with zero vectors."""
    keep = (~drop_mask).to(dtype=action_chunk.dtype)
    action_shape = (action_chunk.shape[0],) + (1,) * (action_chunk.ndim - 1)
    dropped_action = action_chunk * keep.reshape(action_shape)
    if act_visual_tokens is None:
        return dropped_action, None
    visual_shape = (act_visual_tokens.shape[0],) + (1,) * (
        act_visual_tokens.ndim - 1
    )
    dropped_visual = act_visual_tokens * keep.to(
        dtype=act_visual_tokens.dtype
    ).reshape(visual_shape)
    return dropped_action, dropped_visual


def add_random_gain(values, max_gain_change=0.0):
    """Randomly scale each sample to simulate force-amplitude variation."""
    max_gain_change = max(0.0, float(max_gain_change))
    if max_gain_change <= 0.0:
        return values
    gains = torch.empty(
        values.shape[0], 1,
        device=values.device,
        dtype=values.dtype,
    ).uniform_(1.0 - max_gain_change, 1.0 + max_gain_change)
    return values * gains


class OverfitMonitor:
    """Track epoch losses and detect sustained test degradation."""

    def __init__(self, patience=3, min_relative_increase=0.01):
        self.patience = max(1, int(patience))
        self.min_relative_increase = max(0.0, float(min_relative_increase))
        self.history = []
        self.best_test_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")
        self.rise_streak = 0
        self.stop_reason = None

    def update(self, train_loss, val_loss, test_loss):
        epoch = len(self.history) + 1
        train_loss = float(train_loss)
        val_loss = float(val_loss)
        test_loss = float(test_loss)
        train_or_val_improved = (
            train_loss < self.best_train_loss
            or val_loss < self.best_val_loss
        )

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.rise_streak = 0
        elif (
            test_loss > self.best_test_loss * (1.0 + self.min_relative_increase)
        ):
            self.rise_streak += 1
        else:
            self.rise_streak = 0

        self.best_train_loss = min(self.best_train_loss, train_loss)
        self.best_val_loss = min(self.best_val_loss, val_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "best_test_loss": self.best_test_loss,
            "rise_streak": self.rise_streak,
        }
        self.history.append(record)

        if self.rise_streak >= self.patience and train_or_val_improved:
            self.stop_reason = (
                f"test_loss rose for {self.rise_streak} consecutive epochs; "
                f"best_test_loss={self.best_test_loss:.6f}, "
                f"current_test_loss={test_loss:.6f}"
            )
            return True
        return False


def persist_overfit_monitor(monitor, output_dir):
    """Persist monitor history and a diagnostic loss plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "loss_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "test_loss",
                "best_test_loss",
                "rise_streak",
            ],
        )
        writer.writeheader()
        writer.writerows(monitor.history)

    reason_path = output_dir / "stop_reason.json"
    reason_path.write_text(
        json.dumps(
            {
                "stop_reason": monitor.stop_reason,
                "best_test_loss": monitor.best_test_loss,
                "best_val_loss": monitor.best_val_loss,
                "epochs_recorded": len(monitor.history),
            },
            indent=2,
        )
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in monitor.history]
        plt.figure(figsize=(9, 5), dpi=160)
        for key, label in (
            ("train_loss", "Train"),
            ("val_loss", "Validation"),
            ("test_loss", "Test"),
        ):
            plt.plot(epochs, [row[key] for row in monitor.history], marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Residual ACT loss history")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png")
        plt.close()
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc))


@contextmanager
def temporary_parameter_dropout(model, fraction):
    """Temporarily zero a global fraction of trainable parameter elements."""
    if fraction <= 0:
        yield
        return

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.is_floating_point()
    ]
    total_parameters = sum(parameter.numel() for parameter in parameters)
    num_to_zero = int(total_parameters * fraction)
    saved = []

    if num_to_zero > 0:
        selected = torch.randperm(
            total_parameters,
            device=parameters[0].device,
        )[:num_to_zero]
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                next_offset = offset + parameter.numel()
                local_indices = selected[
                    (selected >= offset) & (selected < next_offset)
                ] - offset
                if local_indices.numel() > 0:
                    flat_parameter = parameter.reshape(-1)
                    saved.append(
                        (parameter, local_indices, flat_parameter[local_indices].clone())
                    )
                    flat_parameter[local_indices] = 0
                offset = next_offset

    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, local_indices, original_values in saved:
                parameter.reshape(-1)[local_indices] = original_values


def safe_masked_mean(values, mask):
    if mask.any():
        return values[mask].mean()
    return torch.zeros(
        (),
        dtype=values.dtype,
        device=values.device,
    )


def compute_batch_diagnostics(
    batch,
    pred_delta,
    target_delta,
    criterion,
    topk=3,
):
    pred_float = pred_delta.detach().to(torch.float32)
    target_float = target_delta.detach().to(torch.float32)
    act_chunk = batch["act_chunk"].detach().to(torch.float32)
    expert_action = batch["expert_action"].detach().to(torch.float32)
    tactile_history = batch.get("tactile_history")

    squared_error = (pred_float - target_float).square()
    loss_per_window = squared_error.mean(dim=(1, 2))
    final_action_mse_per_window = (
        pred_float + act_chunk - expert_action
    ).square().mean(dim=(1, 2))
    act_expert_mse_per_window = (
        act_chunk - expert_action
    ).square().mean(dim=(1, 2))

    if tactile_history is not None:
        tactile_magnitude, window_weights = criterion.compute_window_weights(
            tactile_history
        )
        tactile_magnitude = tactile_magnitude.to(torch.float32)
        window_weights = window_weights.to(torch.float32)
        high_mask = tactile_magnitude >= criterion.tau.to(torch.float32)
        low_mask = ~high_mask

    topk_count = min(int(topk), int(loss_per_window.shape[0]))
    top_values, top_indices = torch.topk(
        loss_per_window,
        k=topk_count,
    )
    episode_index = batch.get("episode_index")
    frame_index = batch.get("frame_index")
    top_examples = []
    for rank in range(topk_count):
        batch_index = int(top_indices[rank].item())
        top_examples.append(
            {
                "rank": rank + 1,
                "batch_index": batch_index,
                "episode_index": (
                    int(episode_index[batch_index].item())
                    if isinstance(episode_index, torch.Tensor)
                    else None
                ),
                "frame_index": (
                    int(frame_index[batch_index].item())
                    if isinstance(frame_index, torch.Tensor)
                    else None
                ),
                "loss_per_window": float(top_values[rank].item()),
                "tactile_magnitude": (
                    float(tactile_magnitude[batch_index].item())
                    if tactile_history is not None
                    else None
                ),
                "window_weight": (
                    float(window_weights[batch_index].item())
                    if tactile_history is not None
                    else None
                ),
            }
        )

    diagnostics = {
        "pred_delta_abs_mean": pred_float.abs().mean(),
        "pred_delta_abs_max": pred_float.abs().max(),
        "pred_delta_std": pred_float.std(unbiased=False),
        "target_delta_abs_mean": target_float.abs().mean(),
        "target_delta_abs_max": target_float.abs().max(),
        "target_delta_std": target_float.std(unbiased=False),
        "loss_per_window_mean": loss_per_window.mean(),
        "loss_per_window_max": loss_per_window.max(),
        "loss_per_window_p90": torch.quantile(loss_per_window, 0.90),
        "loss_per_window_p95": torch.quantile(loss_per_window, 0.95),
        "act_expert_mse": act_expert_mse_per_window.mean(),
        "final_action_mse": final_action_mse_per_window.mean(),
    }
    if tactile_history is not None:
        diagnostics.update(
            {
                "high_magnitude_final_action_mse": safe_masked_mean(
                    final_action_mse_per_window, high_mask
                ),
                "low_magnitude_final_action_mse": safe_masked_mean(
                    final_action_mse_per_window, low_mask
                ),
                "high_magnitude_target_delta_abs_mean": safe_masked_mean(
                    target_float.abs().mean(dim=(1, 2)), high_mask
                ),
                "low_magnitude_target_delta_abs_mean": safe_masked_mean(
                    target_float.abs().mean(dim=(1, 2)), low_mask
                ),
                "tactile_input_abs_mean": tactile_history.detach()
                .to(torch.float32)
                .abs()
                .mean(),
                "tactile_input_abs_max": tactile_history.detach()
                .to(torch.float32)
                .abs()
                .max(),
            }
        )

    for action_dim in range(target_float.shape[-1]):
        dim_target = target_float[:, :, action_dim]
        dim_pred = pred_float[:, :, action_dim]
        diagnostics[f"target_delta_dim{action_dim}_std"] = dim_target.std(unbiased=False)
        diagnostics[f"pred_delta_dim{action_dim}_std"] = dim_pred.std(unbiased=False)
        diagnostics[f"pred_delta_dim{action_dim}_mean"] = dim_pred.mean()
        diagnostics[f"action_dim{action_dim}_mse"] = (
            (dim_pred - dim_target).square().mean()
        )

    diagnostics = {
        key: float(value.detach().to(torch.float32).item())
        for key, value in diagnostics.items()
    }
    diagnostics["top_examples"] = top_examples
    return diagnostics


def compute_losses(
    model,
    criterion,
    batch,
    reference_dropout=0.5,
):
    tactile_history = batch.get("tactile_history")
    current_force = batch["current_force"]
    state = batch["observation.state"]
    act_chunk = batch["act_chunk"]
    expert_action = batch["expert_action"]
    act_visual_tokens = batch.get("act_visual_tokens")

    pred_delta, feature_metrics = model(
        tactile_history,
        current_force,
        state,
        act_chunk,
        act_visual_tokens=act_visual_tokens,
        return_feature_metrics=True,
    )
    target_delta = compute_target_delta(
        expert_action,
        act_chunk,
    )
    if pred_delta.ndim == 2:
        target_delta = target_delta[:, 0, :]

    objective_loss, metrics = criterion(
        pred_delta=pred_delta,
        target_delta=target_delta,
        tactile_history=tactile_history,
        act_chunk=act_chunk,
        expert_action=expert_action,
        feature_metrics=feature_metrics,
    )

    return objective_loss, metrics, pred_delta, target_delta


def temporal_smoothness_loss(pred_action):
    """Second-order smoothness penalty for predicted action chunks."""
    if pred_action.shape[1] < 3:
        return pred_action.new_zeros(())
    second_difference = (
        pred_action[:, 2:]
        - 2.0 * pred_action[:, 1:-1]
        + pred_action[:, :-2]
    )
    return second_difference.square().mean()


def action_chunk_overlap_consistency_loss(
    predicted_action,
    absolute_indices,
    episode_ids,
):
    """Penalize disagreement between overlapping predictions from adjacent windows."""
    if predicted_action.shape[0] < 2 or predicted_action.shape[1] < 2:
        return predicted_action.new_zeros(())

    losses = []
    for left in range(predicted_action.shape[0]):
        for right in range(left + 1, predicted_action.shape[0]):
            same_episode = episode_ids[left] == episode_ids[right]
            adjacent = absolute_indices[right] == absolute_indices[left] + 1
            if not (same_episode and adjacent):
                continue
            overlap = predicted_action[left, 1:] - predicted_action[right, :-1]
            losses.append(overlap.square().mean())

    if not losses:
        return predicted_action.new_zeros(())
    return torch.stack(losses).mean()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    epoch,
    use_weighted_loss,
    diagnostic_topk,
    wandb_log_every,
    overfit_single_batch_steps=0,
    reference_dropout=0.5,
    action_noise_std=0.0,
    current_force_noise_std=0.0,
    current_force_gain_range=0.0,
    state_noise_std=0.0,
    temporal_smoothness_weight=0.0,
    action_chunk_consistency_weight=0.0,
    modality_dropout_prob=0.0,
    ablate_modalities=None,
):
    model.train()
    metric_accumulator = init_metric_accumulator()
    if overfit_single_batch_steps > 0:
        first_batch = next(iter(loader))
        train_source = [first_batch] * overfit_single_batch_steps
        pbar = tqdm(
            train_source,
            desc=f"Epoch {epoch} [Train-Overfit1Batch]",
            leave=False,
        )
    else:
        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

    for batch_idx, raw_batch in enumerate(pbar):
        batch = move_batch_to_device(raw_batch, device)
        apply_fixed_modality_ablation(batch, ablate_modalities)
        if action_noise_std > 0.0:
            batch["act_chunk"] = add_action_noise(
                batch["act_chunk"],
                std=action_noise_std,
            )
        if current_force_noise_std > 0.0:
            batch["current_force"] = add_relative_gaussian_noise(
                batch["current_force"],
                relative_std=current_force_noise_std,
            )
        if current_force_gain_range > 0.0:
            batch["current_force"] = add_random_gain(
                batch["current_force"],
                max_gain_change=current_force_gain_range,
            )
        if state_noise_std > 0.0:
            batch["observation.state"] = add_relative_gaussian_noise(
                batch["observation.state"],
                relative_std=state_noise_std,
            )
        if modality_dropout_prob > 0.0:
            drop_mask = torch.rand(
                batch["act_chunk"].shape[0],
                device=batch["act_chunk"].device,
            ) < modality_dropout_prob
            if batch.get("act_visual_tokens") is not None:
                visual = batch["act_visual_tokens"]
                keep = (~drop_mask).to(dtype=visual.dtype)
                visual_shape = (visual.shape[0],) + (1,) * (visual.ndim - 1)
                batch["act_visual_tokens"] = visual * keep.reshape(
                    visual_shape
                )
        optimizer.zero_grad(set_to_none=True)

        with temporary_parameter_dropout(model, reference_dropout):
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                objective_loss, metrics, pred_delta, target_delta = compute_losses(
                    model=model,
                    criterion=criterion,
                    batch=batch,
                )
                if temporal_smoothness_weight > 0.0:
                    predicted_action = batch["act_chunk"] + pred_delta
                    objective_loss = objective_loss + (
                        temporal_smoothness_weight
                        * temporal_smoothness_loss(predicted_action)
                    )
                if action_chunk_consistency_weight > 0.0:
                    predicted_action = batch["act_chunk"] + pred_delta
                    objective_loss = objective_loss + (
                        action_chunk_consistency_weight
                        * action_chunk_overlap_consistency_loss(
                            predicted_action,
                            batch["absolute_index"],
                            batch["episode_index"],
                        )
                    )

            scaler.scale(objective_loss).backward()
            grad_norm = compute_grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()

        metric_values = summarize_batch_metrics(metrics)
        diagnostic_values = compute_batch_diagnostics(
            batch=batch,
            pred_delta=pred_delta,
            target_delta=target_delta,
            criterion=criterion,
            topk=diagnostic_topk,
        )
        metric_values["grad_norm"] = grad_norm
        metric_values["objective_loss"] = float(objective_loss.detach().to(torch.float32).item())
        metric_values.update(
            {
                key: value
                for key, value in diagnostic_values.items()
                if key != "top_examples"
            }
        )
        batch_size = int(batch["act_chunk"].shape[0])
        update_metric_accumulator(
            metric_accumulator,
            metric_values,
            batch_size,
        )

        pbar.set_postfix(
            {
                "loss": f"{float(objective_loss.detach().to(torch.float32).item()):.6f}",
            }
        )

        global_batch = (epoch - 1) * len(loader) + batch_idx + 1
        if global_batch % wandb_log_every == 0:
            log_compact_wandb_metrics(
                step=global_batch,
                metrics=metric_values,
            )
    epoch_metrics = finalize_metric_accumulator(
        metric_accumulator
    )
    return epoch_metrics


def validate(
    model,
    loader,
    criterion,
    device,
    epoch,
    use_weighted_loss,
    diagnostic_topk,
    max_batches=None,
    desc="Val",
    optimizer=None,
    scaler=None,
    update_weights=False,
    reference_dropout=0.5,
    ablate_modalities=None,
):
    if update_weights and (optimizer is None or scaler is None):
        raise ValueError(
            "optimizer and scaler are required when update_weights=True."
        )

    if update_weights:
        model.train()
    else:
        model.eval()
    metric_accumulator = init_metric_accumulator()
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch} [{desc}]",
        leave=False,
    )

    context = torch.enable_grad() if update_weights else torch.no_grad()
    with context:
        for batch_idx, raw_batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            apply_fixed_modality_ablation(batch, ablate_modalities)
            if update_weights:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                objective_loss, metrics, pred_delta, target_delta = compute_losses(
                    model=model,
                    criterion=criterion,
                    batch=batch,
                )
            if update_weights:
                scaler.scale(objective_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            metric_values = summarize_batch_metrics(metrics)
            diagnostic_values = compute_batch_diagnostics(
                batch=batch,
                pred_delta=pred_delta,
                target_delta=target_delta,
                criterion=criterion,
                topk=diagnostic_topk,
            )
            metric_values["grad_norm"] = (
                compute_grad_norm(model.parameters())
                if update_weights
                else 0.0
            )
            metric_values["objective_loss"] = float(objective_loss.detach().to(torch.float32).item())
            metric_values.update(
                {
                    key: value
                    for key, value in diagnostic_values.items()
                    if key != "top_examples"
                }
            )
            batch_size = int(batch["act_chunk"].shape[0])
            update_metric_accumulator(
                metric_accumulator,
                metric_values,
                batch_size,
            )
            pbar.set_postfix(
                {
                    "loss": f"{float(objective_loss.detach().to(torch.float32).item()):.6f}",
                }
            )
    epoch_metrics = finalize_metric_accumulator(
        metric_accumulator
    )
    return epoch_metrics


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scaler,
    train_metrics,
    val_metrics,
    criterion,
    config_snapshot,
    monitor_state=None,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "criterion_state": criterion.state_dict(),
            "tactile_loss": {
                "use_weighted_loss": criterion.use_weighted_loss,
                "tactile_input_already_normalized": criterion.tactile_input_already_normalized,
                "tau": float(criterion.tau.item()),
                "alpha": float(criterion.alpha.item()),
                "slope": float(criterion.slope.item()),
                "eps": float(criterion.eps.item()),
                "channel_mean": criterion.channel_mean.detach().cpu(),
                "channel_std": criterion.channel_std.detach().cpu(),
            },
            "config": config_snapshot,
            "overfit_monitor": monitor_state,
        },
        path,
    )


def resolve_checkpoint_dir(training_cfg):
    checkpoint_dir = training_cfg.get(
        "checkpoint_dir",
        ".",
    )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return checkpoint_dir


def maybe_resume_training(
    model,
    optimizer,
    scaler,
    criterion,
    training_cfg,
    checkpoint_path,
    device,
):
    if checkpoint_path is None:
        return 0, float("inf")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = int(checkpoint["epoch"])
    best_val_loss = float("inf")
    if checkpoint.get("val_metrics") is not None:
        best_val_loss = float(
            checkpoint["val_metrics"].get(
                "objective_loss",
                checkpoint["val_metrics"].get(
                    "weighted_loss",
                    checkpoint["val_metrics"].get(
                        "unweighted_loss",
                        float("inf"),
                    ),
                ),
            )
        )

    checkpoint_tactile = checkpoint.get("tactile_loss")
    override = bool(
        training_cfg.get(
            "override_tactile_loss_from_checkpoint",
            False,
        )
    )
    if checkpoint_tactile:
        if override:
            warnings.warn(
                "Resuming with current tactile loss config overriding checkpoint tactile config.",
                stacklevel=2,
            )
        else:
            criterion.load_state_dict(
                checkpoint["criterion_state"],
                strict=False,
            )
            warnings.warn(
                "Loaded tactile loss statistics and tau/alpha/slope from checkpoint.",
                stacklevel=2,
            )

    print(
        f"Resume from {checkpoint_path}, start_epoch={start_epoch + 1}"
    )
    return start_epoch, best_val_loss


def main():
    args = parse_args()
    dataloader_config = load_yaml(args.dataloader_config)
    model_config = load_yaml(args.model_config)
    training_cfg = model_config["training"]
    if args.ablate_modalities is not None:
        training_cfg["ablate_modalities"] = list(args.ablate_modalities)
    if args.checkpoint_dir is not None:
        training_cfg["checkpoint_dir"] = str(args.checkpoint_dir)

    set_seed(int(dataloader_config["split"]["seed"]))
    device = resolve_device(training_cfg)
    os.environ.setdefault("WANDB_MODE", "online")
    policy_cfg = dataloader_config.get("policy", {})
    if (
        policy_cfg.get("device") == "cuda"
        and not torch.cuda.is_available()
    ):
        print(
            "CUDA unavailable in current environment. "
            "Override dataloader policy.device from cuda to cpu."
        )
        policy_cfg["device"] = "cpu"

    tactile_type = dataloader_config["dataset"]["keys"]["tactile_type"]
    model_tactile_type = model_config["tactile_encoder"]["type"]
    if model_tactile_type != tactile_type:
        raise ValueError(
            "Tactile type mismatch between config/model_config.yaml and "
            "dataloader/tactile_dataloader.yaml: "
            f"model={model_tactile_type!r}, dataloader={tactile_type!r}."
        )
    decoder_cfg = model_config["decoder"]
    model_action_horizon = int(decoder_cfg["action_horizon"])
    model_action_dim = int(decoder_cfg["action_dim"])
    data_action_horizon = int(
        dataloader_config["sequence"]["action_horizon"]
    )
    if model_action_horizon != data_action_horizon:
        raise ValueError(
            "Action horizon mismatch between "
            "config/model_config.yaml and "
            "dataloader/tactile_dataloader.yaml: "
            f"model={model_action_horizon}, "
            f"dataloader={data_action_horizon}."
        )

    if tactile_type == "image":
        tactile_history = dataloader_config["sequence"]["tactile_history_image"]
        tactile_channels = int(
            model_config["tactile_encoder"]["image"]["in_channels"]
        )
    elif tactile_type == "force":
        tactile_history = dataloader_config["sequence"]["tactile_history_force"]
        tactile_channels = int(
            model_config["tactile_encoder"]["force"]["input_dim"]
        )
    elif tactile_type == "vqvae":
        tactile_history = dataloader_config["sequence"]["tactile_history_force"]
        tactile_channels = int(
            model_config["tactile_encoder"]["vqvae"]["input_dim"]
        )
    else:
        raise ValueError(f"未知的 tactile_type: {tactile_type}")

    criterion, criterion_metadata = build_tactile_criterion(
        training_cfg=training_cfg,
        tactile_type=tactile_type,
        tactile_channels=tactile_channels,
        action_horizon=model_action_horizon,
        action_dim=model_action_dim,
    )
    criterion = criterion.to(device)
    state_stats = load_and_validate_state_stats(
        model_config=model_config,
        action_dim=model_action_dim,
    )
    tactile_channel_names = None
    if tactile_type == "force":
        tactile_channel_names = validate_force_channel_order(
            dataloader_config=dataloader_config,
            model_config=model_config,
            stats_payload=criterion_metadata["stats_payload"],
        )
    elif tactile_type == "vqvae":
        # VQ-VAE使用force的通道顺序配置
        tactile_channel_names = validate_force_channel_order(
            dataloader_config=dataloader_config,
            model_config=model_config,
            stats_payload=criterion_metadata["stats_payload"],
        )
    use_weighted_loss = bool(
        training_cfg.get("use_tactile_weighted_loss", False)
    )

    try:
        wandb.init(
            project="tactile-residual-act",
            name=f"train_{dataloader_config['split']['seed']}_{model_config['tactile_encoder']['type']}",
            config={
                "epochs": int(training_cfg["num_epochs"]),
                "batch_size": dataloader_config["loader"]["batch_size"],
                "learning_rate": float(training_cfg["learning_rate"]),
                "weight_decay": float(training_cfg["weight_decay"]),
                "tactile_type": tactile_type,
                "tactile_history": tactile_history,
                "action_horizon": model_action_horizon,
                "action_dim": model_action_dim,
                "train_ratio": dataloader_config["split"]["train"],
                "val_ratio": dataloader_config["split"]["val"],
                "test_ratio": dataloader_config["split"]["test"],
                "seed": dataloader_config["split"]["seed"],
                "device": str(device),
                "tactile_encoder_type": model_config["tactile_encoder"]["type"],
                "use_tactile_weighted_loss": use_weighted_loss,
                "tactile_weight_tau": float(criterion.tau.item()),
                "tactile_weight_alpha": float(criterion.alpha.item()),
                "tactile_weight_slope": float(criterion.slope.item()),
                "tactile_weight_eps": float(criterion.eps.item()),
                "tactile_input_already_normalized": bool(
                    criterion.tactile_input_already_normalized
                ),
                "tactile_stats_path": criterion_metadata["tactile_stats_path"],
            },
        )
    except Exception as exc:
        print(f"wandb disabled due to init failure: {exc}")

    print(
        "Tactile weighting input scale: "
        + (
            "already normalized"
            if criterion.tactile_input_already_normalized
            else "raw tactile_history, normalized inside TactileMagnitudeWeightedMSE with saved train mean/std"
        )
    )

    use_cache_loader = bool(
        dataloader_config.get("use_cache_loader", False)
    )
    act_cache_path = dataloader_config["policy"].get("act_cache_path")

    if use_cache_loader:
        # 完全缓存模式：从缓存读取所有训练输入，不加载LeRobotDataset/ACT策略
        if not act_cache_path:
            raise ValueError(
                "use_cache_loader=true需要设置policy.act_cache_path。"
            )
        print(f"[CacheLoader] 使用离线缓存: {act_cache_path}")
        from dataloader.cache_loader import build_cached_loaders

        loaders = build_cached_loaders(
            dataloader_config,
            act_cache_path,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders.get("test")
    else:
        dataset = build_base_dataset(dataloader_config)
        normal_loaders, _ = build_normal_dataloaders(
            dataloader_config,
            dataset,
        )

        if act_cache_path:
            # 使用离线预处理缓存：不再加载/运行ACT策略
            print(f"使用ACT离线缓存: {act_cache_path}")
            policy, preprocessor, postprocessor, act_device = (
                None,
                None,
                None,
                device,
            )
        else:
            policy, preprocessor, postprocessor, act_device = (
                load_lerobot_policy(
                    dataloader_config,
                    dataset,
                )
            )

        loaders = build_augmented_loaders(
            dataloader_config,
            normal_loaders,
            policy,
            preprocessor,
            postprocessor,
            act_device,
        )

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders.get("test")

    current_force_mean, current_force_std, normalize_current_force = (
        resolve_current_force_normalization(model_config, criterion_metadata)
    )
    model = TactileResidualACT(
        tactile_encoder_type=model_config["tactile_encoder"]["type"],
        action_horizon=model_action_horizon,
        action_dim=model_action_dim,
        use_tactile_history=bool(
            model_config["tactile_encoder"].get("enabled", True)
        ),
        tactile_encoder_cfg=model_config["tactile_encoder"][tactile_type],
        state_encoder_cfg=model_config.get("state_encoder"),
        action_encoder_cfg=model_config.get("action_encoder"),
        current_force_encoder_cfg=model_config.get("current_force_encoder"),
        force_film_cfg=model_config.get("force_film"),
        fusion_cfg=model_config.get("fusion"),
        decoder_cfg=decoder_cfg,
        tactile_channel_mean=criterion.channel_mean,
        tactile_channel_std=criterion.channel_std,
        tactile_channel_names=tactile_channel_names,
        normalize_tactile_input=not criterion.tactile_input_already_normalized,
        state_mean=state_stats["mean"],
        state_std=state_stats["std"],
        state_channel_names=state_stats["channel_names"],
        normalize_state_input=bool(
            model_config["state_encoder"].get("normalize_input", True)
        ),
        current_force_mean=current_force_mean,
        current_force_std=current_force_std,
        normalize_current_force_input=normalize_current_force,
        use_act_visual=bool(
            model_config.get("act_visual", {}).get("enabled", False)
        ),
        visual_encoder_cfg=model_config.get("act_visual"),
    ).to(device)

    print(
        f"\n使用触觉编码器类型: {model_config['tactile_encoder']['type']}"
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 60)
    print("模型参数统计:")
    print("=" * 60)
    print(f"总参数数量:       {total_params:,}")
    print(f"可训练参数数量:   {trainable_params:,}")
    print(f"冻结参数数量:     {total_params - trainable_params:,}")
    print("=" * 60)

    print("\n模块参数详情:")
    print("-" * 60)
    for name, module in model.named_children():
        module_params = sum(
            p.numel() for p in module.parameters()
        )
        module_trainable = sum(
            p.numel() for p in module.parameters()
            if p.requires_grad
        )
        print(
            f"{name:20s} | 总参数: {module_params:>10,} | 可训练: {module_trainable:>10,}"
        )
    print("-" * 60 + "\n")

    if wandb.run is not None:
        wandb.config.update(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
        )

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    resume_checkpoint = (
        args.resume_checkpoint
        if args.resume_checkpoint is not None
        else (
            Path(training_cfg["resume_checkpoint"])
            if training_cfg.get("resume_checkpoint")
            else None
        )
    )
    start_epoch, best_val_loss = maybe_resume_training(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        criterion=criterion,
        training_cfg=training_cfg,
        checkpoint_path=resume_checkpoint,
        device=device,
    )

    epochs = int(training_cfg["num_epochs"])
    validate_every = int(training_cfg.get("validate_every", 3))
    checkpoint_every = int(training_cfg.get("checkpoint_every", 10))
    wandb_log_every = int(training_cfg.get("wandb_log_every", 100))
    if wandb_log_every < 1:
        raise ValueError("training.wandb_log_every must be at least 1.")
    diagnostic_topk = int(training_cfg.get("diagnostic_topk", 3))
    overfit_single_batch_steps = int(
        training_cfg.get("overfit_single_batch_steps", 0)
    )
    diagnostic_val_batches = int(
        training_cfg.get("diagnostic_val_batches", 1)
    )
    reference_dropout = float(
        training_cfg.get("reference_dropout", 0.5)
    )
    action_noise_std = float(
        training_cfg.get("action_noise_std", 0.0)
    )
    current_force_noise_std = float(
        training_cfg.get("current_force_noise_std", 0.0)
    )
    current_force_gain_range = float(
        training_cfg.get("current_force_gain_range", 0.0)
    )
    state_noise_std = float(training_cfg.get("state_noise_std", 0.0))
    temporal_smoothness_weight = float(
        training_cfg.get("temporal_smoothness_weight", 0.0)
    )
    action_chunk_consistency_weight = float(
        training_cfg.get("action_chunk_consistency_weight", 0.0)
    )
    modality_dropout_prob = float(
        training_cfg.get("modality_dropout_prob", 0.0)
    )
    ablate_modalities = list(training_cfg.get("ablate_modalities", []))
    minimum_reference_dropout = reference_dropout
    adaptive_dropout_on_test_increase = bool(
        training_cfg.get("adaptive_dropout_on_test_increase", True)
    )
    dropout_increase = float(
        training_cfg.get("test_mse_dropout_increase", 0.01)
    )
    max_reference_dropout = float(
        training_cfg.get("max_reference_dropout", 0.5)
    )
    checkpoint_dir = resolve_checkpoint_dir(training_cfg)
    monitor_cfg = training_cfg.get("overfit_monitor", {})
    monitor_enabled = bool(monitor_cfg.get("enabled", True))
    overfit_monitor = OverfitMonitor(
        patience=int(monitor_cfg.get("patience", 3)),
        min_relative_increase=float(
            monitor_cfg.get("min_relative_test_increase", 0.01)
        ),
    )
    monitor_dir = checkpoint_dir / (
        f"overfit_monitor_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    config_snapshot = {
        "dataloader_config": copy.deepcopy(dataloader_config),
        "model_config": copy.deepcopy(model_config),
    }

    epoch_pbar = tqdm(
        range(start_epoch, epochs),
        desc="Training Progress",
    )

    for epoch_index in epoch_pbar:
        epoch = epoch_index + 1
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_weighted_loss=use_weighted_loss,
            diagnostic_topk=diagnostic_topk,
            wandb_log_every=wandb_log_every,
            overfit_single_batch_steps=overfit_single_batch_steps,
            reference_dropout=reference_dropout,
            action_noise_std=action_noise_std,
            current_force_noise_std=current_force_noise_std,
            current_force_gain_range=current_force_gain_range,
            state_noise_std=state_noise_std,
            temporal_smoothness_weight=temporal_smoothness_weight,
            action_chunk_consistency_weight=action_chunk_consistency_weight,
            modality_dropout_prob=modality_dropout_prob,
            ablate_modalities=ablate_modalities,
        )

        if epoch % validate_every == 0:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                use_weighted_loss=use_weighted_loss,
                diagnostic_topk=diagnostic_topk,
                max_batches=(
                    diagnostic_val_batches
                    if args.diagnostic_only
                    else None
                ),
                update_weights=False,
                reference_dropout=reference_dropout,
                ablate_modalities=ablate_modalities,
            )
            val_loss = float(val_metrics.get("objective_loss", 0.0))
            safe_wandb_log(
                {
                    "epoch": epoch,
                    "val/loss": val_loss,
                }
            )

            test_metrics = None
            if test_loader is not None:
                test_metrics = validate(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    use_weighted_loss=use_weighted_loss,
                    diagnostic_topk=diagnostic_topk,
                    max_batches=(
                        diagnostic_val_batches
                        if args.diagnostic_only
                        else None
                    ),
                    desc="Test",
                    ablate_modalities=ablate_modalities,
                )
                test_loss = float(test_metrics.get("objective_loss", 0.0))
                safe_wandb_log(
                    {
                        "epoch": epoch,
                        "test/loss": test_loss,
                    }
                )

                overfit_triggered = False
                if monitor_enabled:
                    overfit_triggered = overfit_monitor.update(
                        train_loss=float(train_metrics.get("objective_loss", 0.0)),
                        val_loss=val_loss,
                        test_loss=test_loss,
                    )
                    persist_overfit_monitor(overfit_monitor, monitor_dir)
                safe_wandb_log(
                    {
                        "epoch": epoch,
                        "monitor/test_best_loss": overfit_monitor.best_test_loss,
                        "monitor/test_rise_streak": overfit_monitor.rise_streak,
                    }
                )

            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{train_metrics.get('objective_loss', 0.0):.6f}",
                    "val_loss": f"{val_loss:.6f}",
                }
            )
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_metrics.get('objective_loss', 0.0):.6f} "
                f"val_loss={val_loss:.6f}"
                + (
                    f" test_loss={test_loss:.6f}"
                    if test_metrics is not None
                    else ""
                )
            )
        else:
            val_metrics = None
            val_loss = None
            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{train_metrics.get('objective_loss', 0.0):.6f}",
                }
            )
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_metrics.get('objective_loss', 0.0):.6f}"
            )

        if args.diagnostic_only:
            print("Diagnostic-only run finished after one epoch.")
            break

        if test_metrics is not None and overfit_triggered:
            stop_checkpoint_path = (
                checkpoint_dir / f"checkpoint_overfit_epoch_{epoch}.pth"
            )
            save_checkpoint(
                path=stop_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                criterion=criterion,
                config_snapshot=config_snapshot,
                monitor_state={
                    "history": overfit_monitor.history,
                    "stop_reason": overfit_monitor.stop_reason,
                },
            )
            safe_wandb_save(stop_checkpoint_path)
            print(
                "自动停止：检测到过拟合。"
                f"原因：{overfit_monitor.stop_reason}"
            )
            break

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = (
                checkpoint_dir / "checkpoint_best.pth"
            )
            save_checkpoint(
                path=best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                criterion=criterion,
                config_snapshot=config_snapshot,
            )
            safe_wandb_save(best_checkpoint_path)
            print(
                "  → 保存最佳模型 "
                f"{best_checkpoint_path} "
                f"(val_loss={val_loss:.6f})"
            )

        latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.pth"
        save_checkpoint(
            path=latest_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            criterion=criterion,
            config_snapshot=config_snapshot,
            monitor_state=(
                {
                    "history": overfit_monitor.history,
                    "stop_reason": overfit_monitor.stop_reason,
                }
                if monitor_enabled and test_metrics is not None
                else None
            ),
        )

        if epoch % checkpoint_every == 0:
            checkpoint_path = (
                checkpoint_dir
                / f"checkpoint_{epoch}.pth"
            )
            save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                criterion=criterion,
                config_snapshot=config_snapshot,
            )
            safe_wandb_save(checkpoint_path)

    safe_wandb_finish()
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
