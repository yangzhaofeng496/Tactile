import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.cache_loader import build_cached_loaders
from dataloader.dataloader import load_yaml, set_seed
from model import TactileResidualACT, compute_target_delta
from train import (
    apply_fixed_modality_ablation,
    build_tactile_criterion,
    load_and_validate_state_stats,
    move_batch_to_device,
    resolve_current_force_normalization,
    resolve_device,
    validate_force_channel_order,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint loss grouped by episode."
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
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ablate-modalities",
        nargs="*",
        default=None,
        choices=["current_force", "state", "visual"],
    )
    parser.add_argument(
        "--recent-episodes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--overfit-threshold",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--relative-increase",
        type=float,
        default=0.01,
        help="Episode loss must rise by this relative amount to count as overfit.",
    )
    return parser.parse_args()


def infer_tactile_shape(dataloader_config, model_config):
    tactile_type = dataloader_config["dataset"]["keys"]["tactile_type"]
    if model_config["tactile_encoder"]["type"] != tactile_type:
        raise ValueError(
            "Tactile type mismatch: "
            f"model={model_config['tactile_encoder']['type']!r}, "
            f"dataloader={tactile_type!r}."
        )

    if tactile_type == "image":
        tactile_channels = int(
            model_config["tactile_encoder"]["image"]["in_channels"]
        )
    elif tactile_type == "force":
        tactile_channels = int(
            model_config["tactile_encoder"]["force"]["input_dim"]
        )
    elif tactile_type == "vqvae":
        tactile_channels = int(
            model_config["tactile_encoder"]["vqvae"]["input_dim"]
        )
    else:
        raise ValueError(f"Unknown tactile_type: {tactile_type}")
    return tactile_type, tactile_channels


def build_model_and_criterion(dataloader_config, model_config, device):
    training_cfg = model_config["training"]
    tactile_type, tactile_channels = infer_tactile_shape(
        dataloader_config,
        model_config,
    )
    decoder_cfg = model_config["decoder"]
    action_horizon = int(decoder_cfg["action_horizon"])
    action_dim = int(decoder_cfg["action_dim"])

    criterion, criterion_metadata = build_tactile_criterion(
        training_cfg=training_cfg,
        tactile_type=tactile_type,
        tactile_channels=tactile_channels,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    criterion = criterion.to(device)

    state_stats = load_and_validate_state_stats(
        model_config=model_config,
        action_dim=action_dim,
    )
    tactile_channel_names = None
    if tactile_type in ("force", "vqvae"):
        tactile_channel_names = validate_force_channel_order(
            dataloader_config=dataloader_config,
            model_config=model_config,
            stats_payload=criterion_metadata["stats_payload"],
        )

    current_force_mean, current_force_std, normalize_current_force = (
        resolve_current_force_normalization(model_config, criterion_metadata)
    )

    model = TactileResidualACT(
        tactile_encoder_type=tactile_type,
        action_horizon=action_horizon,
        action_dim=action_dim,
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
    return model, criterion


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def evaluate_by_episode(model, loader, device, ablate_modalities):
    model.eval()
    sums = defaultdict(float)
    counts = defaultdict(int)

    with torch.no_grad():
        for raw_batch in tqdm(loader, desc="Episode eval", leave=False):
            batch = move_batch_to_device(raw_batch, device)
            apply_fixed_modality_ablation(batch, ablate_modalities)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred_delta, _ = model(
                    batch.get("tactile_history"),
                    batch["current_force"],
                    batch["observation.state"],
                    batch["act_chunk"],
                    act_visual_tokens=batch.get("act_visual_tokens"),
                    return_feature_metrics=True,
                )
                target_delta = compute_target_delta(
                    batch["expert_action"],
                    batch["act_chunk"],
                )
                if pred_delta.ndim == 2:
                    target_delta = target_delta[:, 0, :]
                per_window_loss = (pred_delta - target_delta).square().mean(
                    dim=tuple(range(1, pred_delta.ndim))
                )

            episode_ids = batch["episode_index"].detach().cpu().tolist()
            losses = per_window_loss.detach().to(torch.float32).cpu().tolist()
            for episode_id, loss in zip(episode_ids, losses):
                sums[int(episode_id)] += float(loss)
                counts[int(episode_id)] += 1

    rows = []
    for episode_id in sorted(sums):
        rows.append(
            {
                "episode_index": episode_id,
                "loss": sums[episode_id] / max(counts[episode_id], 1),
                "num_windows": counts[episode_id],
            }
        )
    return rows


def load_previous_rows(output_dir, split):
    candidates = sorted(output_dir.glob(f"{split}_episode_loss_*.csv"))
    if not candidates:
        return None, None
    previous = candidates[-1]
    rows = {}
    with previous.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[int(row["episode_index"])] = float(row["loss"])
    return previous, rows


def summarize_overfit(rows, previous_rows, recent_episodes, relative_increase):
    if previous_rows is None:
        return {
            "previous_file": None,
            "recent_episode_count": 0,
            "overfit_episode_count": 0,
            "overfit_fraction": 0.0,
            "triggered": False,
            "overfit_episode_ids": [],
        }

    recent = rows[-recent_episodes:]
    overfit_episode_ids = []
    for row in recent:
        previous_loss = previous_rows.get(row["episode_index"])
        if previous_loss is None:
            continue
        if row["loss"] > previous_loss * (1.0 + relative_increase):
            overfit_episode_ids.append(row["episode_index"])

    comparable = [
        row for row in recent
        if row["episode_index"] in previous_rows
    ]
    denominator = max(len(comparable), 1)
    overfit_fraction = len(overfit_episode_ids) / denominator
    return {
        "recent_episode_count": len(comparable),
        "overfit_episode_count": len(overfit_episode_ids),
        "overfit_fraction": overfit_fraction,
        "triggered": False,
        "overfit_episode_ids": overfit_episode_ids,
    }


def write_outputs(args, rows, summary, checkpoint):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(checkpoint.get("epoch", -1))
    tag = f"epoch_{epoch:03d}" if epoch >= 0 else "epoch_unknown"
    csv_path = args.output_dir / f"{args.split}_episode_loss_{tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode_index", "loss", "num_windows"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_epoch"] = epoch
    summary["split"] = args.split
    summary["csv_path"] = str(csv_path)
    summary["recent_episodes"] = args.recent_episodes
    summary["overfit_threshold"] = args.overfit_threshold
    summary["triggered"] = (
        summary["overfit_fraction"] >= args.overfit_threshold
        and summary["recent_episode_count"] > 0
    )
    summary_path = args.output_dir / f"{args.split}_episode_overfit_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return csv_path, summary_path


def main():
    args = parse_args()
    dataloader_config = load_yaml(args.dataloader_config)
    model_config = load_yaml(args.model_config)
    training_cfg = model_config["training"]
    if args.ablate_modalities is not None:
        training_cfg["ablate_modalities"] = list(args.ablate_modalities)

    set_seed(int(dataloader_config["split"]["seed"]))
    device = resolve_device(training_cfg)
    loaders = build_cached_loaders(
        dataloader_config,
        dataloader_config["policy"]["act_cache_path"],
    )
    model, _criterion = build_model_and_criterion(
        dataloader_config,
        model_config,
        device,
    )
    checkpoint = load_model_weights(model, args.checkpoint, device)
    previous_file, previous_rows = load_previous_rows(args.output_dir, args.split)
    rows = evaluate_by_episode(
        model=model,
        loader=loaders[args.split],
        device=device,
        ablate_modalities=training_cfg.get("ablate_modalities", []),
    )
    summary = summarize_overfit(
        rows=rows,
        previous_rows=previous_rows,
        recent_episodes=args.recent_episodes,
        relative_increase=args.relative_increase,
    )
    summary["previous_file"] = str(previous_file) if previous_file else None
    csv_path, summary_path = write_outputs(args, rows, summary, checkpoint)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(
        "recent_overfit="
        f"{summary['overfit_episode_count']}/{summary['recent_episode_count']} "
        f"({summary['overfit_fraction']:.1%}), "
        f"triggered={summary['triggered']}"
    )


if __name__ == "__main__":
    main()
