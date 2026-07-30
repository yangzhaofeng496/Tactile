import argparse
import copy
import json
import os
import warnings
from pathlib import Path

import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
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

    payload = {
        "train/weight_loss": float(metrics["weighted_loss"]),
        "train/unweight_loss": float(metrics["unweighted_loss"]),
        "train/encoder_rms/action": float(metrics["action_encoder_rms"]),
        "train/encoder_rms/tactile": float(metrics["tactile_encoder_rms"]),
        "train/encoder_rms/state": float(metrics["state_encoder_rms"]),
        "train/modality_contribution/action": float(
            metrics["action_contribution_ratio"]
        ),
        "train/modality_contribution/tactile": float(
            metrics["tactile_contribution_ratio"]
        ),
        "train/modality_contribution/state": float(
            metrics["state_contribution_ratio"]
        ),
    }
    for action_dim in range(6):
        payload[f"train/action_dim_mse/axis_{action_dim}"] = float(
            metrics[f"action_dim{action_dim}_mse"]
        )
        payload[f"train/delta_action/axis_{action_dim}"] = float(
            metrics[f"pred_delta_dim{action_dim}_mean"]
        )

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
    return parser.parse_args()


def move_batch_to_device(batch, device):
    keys = [
        "tactile_history",
        "observation.state",
        "act_chunk",
        "expert_action",
        "delta_action_target",
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


def init_metric_accumulator():
    return {
        "count": 0,
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
        accumulator[key] += float(metrics[metric_name]) * batch_size


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


def summarize_batch_metrics(metrics):
    return {
        key: float(value.detach().to(torch.float32).item())
        for key, value in metrics.items()
    }


def compute_grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(torch.float32)
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5


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
    tactile_history = batch["tactile_history"]

    squared_error = (pred_float - target_float).square()
    loss_per_window = squared_error.mean(dim=(1, 2))
    final_action_mse_per_window = (
        pred_float + act_chunk - expert_action
    ).square().mean(dim=(1, 2))
    act_expert_mse_per_window = (
        act_chunk - expert_action
    ).square().mean(dim=(1, 2))

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
                "tactile_magnitude": float(tactile_magnitude[batch_index].item()),
                "window_weight": float(window_weights[batch_index].item()),
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
        "high_magnitude_final_action_mse": safe_masked_mean(
            final_action_mse_per_window,
            high_mask,
        ),
        "low_magnitude_final_action_mse": safe_masked_mean(
            final_action_mse_per_window,
            low_mask,
        ),
        "high_magnitude_target_delta_abs_mean": safe_masked_mean(
            target_float.abs().mean(dim=(1, 2)),
            high_mask,
        ),
        "low_magnitude_target_delta_abs_mean": safe_masked_mean(
            target_float.abs().mean(dim=(1, 2)),
            low_mask,
        ),
        "tactile_input_abs_mean": tactile_history.detach().to(torch.float32).abs().mean(),
        "tactile_input_abs_max": tactile_history.detach().to(torch.float32).abs().max(),
    }

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
):
    tactile_history = batch["tactile_history"]
    state = batch["observation.state"]
    act_chunk = batch["act_chunk"]
    expert_action = batch["expert_action"]

    pred_delta, feature_metrics = model(
        tactile_history,
        state,
        act_chunk,
        return_feature_metrics=True,
    )
    target_delta = compute_target_delta(
        expert_action,
        act_chunk,
    )

    objective_loss, metrics = criterion(
        pred_delta=pred_delta,
        target_delta=target_delta,
        tactile_history=tactile_history,
        act_chunk=act_chunk,
        expert_action=expert_action,
    )

    metrics = {
        **metrics,
        **{
            name: value.detach()
            for name, value in feature_metrics.items()
        },
    }

    return objective_loss, metrics, pred_delta, target_delta


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
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            objective_loss, metrics, pred_delta, target_delta = compute_losses(
                model=model,
                criterion=criterion,
                batch=batch,
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
        metric_values.update(
            {
                key: value
                for key, value in diagnostic_values.items()
                if key != "top_examples"
            }
        )
        batch_size = int(batch["tactile_history"].shape[0])
        update_metric_accumulator(
            metric_accumulator,
            metric_values,
            batch_size,
        )

        pbar.set_postfix(
            {
                "loss": f"{float(objective_loss.detach().to(torch.float32).item()):.6f}",
                "weighted": f"{metric_values['weighted_loss']:.6f}",
                "unweighted": f"{metric_values['unweighted_loss']:.6f}",
                "w_mean": f"{metric_values['weight_mean']:.3f}",
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
):
    model.eval()
    metric_accumulator = init_metric_accumulator()
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch} [Val]",
        leave=False,
    )

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with autocast(enabled=device.type == "cuda"):
                _, metrics, pred_delta, target_delta = compute_losses(
                    model=model,
                    criterion=criterion,
                    batch=batch,
                )
            metric_values = summarize_batch_metrics(metrics)
            diagnostic_values = compute_batch_diagnostics(
                batch=batch,
                pred_delta=pred_delta,
                target_delta=target_delta,
                criterion=criterion,
                topk=diagnostic_topk,
            )
            metric_values.update(
                {
                    key: value
                    for key, value in diagnostic_values.items()
                    if key != "top_examples"
                }
            )
            batch_size = int(batch["tactile_history"].shape[0])
            update_metric_accumulator(
                metric_accumulator,
                metric_values,
                batch_size,
            )
            pbar.set_postfix(
                {
                    "weighted": f"{metric_values['weighted_loss']:.6f}",
                    "unweighted": f"{metric_values['unweighted_loss']:.6f}",
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
                "weighted_loss",
                checkpoint["val_metrics"].get(
                    "unweighted_loss",
                    float("inf"),
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

    set_seed(int(dataloader_config["split"]["seed"]))
    device = resolve_device(training_cfg)
    os.environ.setdefault("WANDB_MODE", "disable")
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

    dataset = build_base_dataset(dataloader_config)
    normal_loaders, _ = build_normal_dataloaders(
        dataloader_config,
        dataset,
    )

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

    model = TactileResidualACT(
        tactile_encoder_type=model_config["tactile_encoder"]["type"],
        action_horizon=model_action_horizon,
        action_dim=model_action_dim,
        tactile_encoder_cfg=model_config["tactile_encoder"][tactile_type],
        state_encoder_cfg=model_config.get("state_encoder"),
        action_encoder_cfg=model_config.get("action_encoder"),
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
    scaler = GradScaler(enabled=device.type == "cuda")

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
    checkpoint_dir = resolve_checkpoint_dir(training_cfg)

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
            )
            val_loss = float(val_metrics["weighted_loss"])
            epoch_pbar.set_postfix(
                {
                    "train_w": f"{train_metrics['weighted_loss']:.6f}",
                    "val_w": f"{val_loss:.6f}",
                }
            )
            print(
                f"epoch {epoch}/{epochs} "
                f"train_weighted={train_metrics['weighted_loss']:.6f} "
                f"train_unweighted={train_metrics['unweighted_loss']:.6f} "
                f"val_weighted={val_metrics['weighted_loss']:.6f} "
                f"val_unweighted={val_metrics['unweighted_loss']:.6f}"
            )
        else:
            val_metrics = None
            val_loss = None
            epoch_pbar.set_postfix(
                {
                    "train_w": f"{train_metrics['weighted_loss']:.6f}",
                }
            )
            print(
                f"epoch {epoch}/{epochs} "
                f"train_weighted={train_metrics['weighted_loss']:.6f} "
                f"train_unweighted={train_metrics['unweighted_loss']:.6f}"
            )

        if args.diagnostic_only:
            print("Diagnostic-only run finished after one epoch.")
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
                f"(val_weighted_loss={val_loss:.6f})"
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
