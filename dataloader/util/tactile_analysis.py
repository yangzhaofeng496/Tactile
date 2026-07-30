#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std/p50/p95/p99 for one training feature and plot them."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../tactile_dataloader.yaml"),
        help="Path to the existing dataloader YAML config.",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path(__file__).with_name("tactile_analysis.yaml"),
        help="YAML file that selects the feature and channel to analyze.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="tactile_history",
        help=(
            "Feature name to analyze. "
            "Use `tactile_history` for the train split window output, "
            "or a raw dataset key such as `observation.tactile.left_force`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tactile_analysis"),
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/tactile_analysis_hf"),
        help="Writable Hugging Face cache directory.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_analysis_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Analysis config must be a YAML mapping: {config_path}")

    feature = config.get("feature")
    channels = config.get("channels", config.get("channel"))
    channel_mapping = config.get("channel_mapping")
    if not isinstance(feature, str) or not feature.strip():
        raise ValueError(f"`feature` must be a non-empty string in {config_path}")
    if isinstance(channels, str):
        channels = [channels]
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(channel, str) or not channel.strip() for channel in channels)
    ):
        raise ValueError(
            f"`channels` must be a non-empty list of channel names in {config_path}"
        )
    normalized_channels = [channel.strip() for channel in channels]
    if len({channel.casefold() for channel in normalized_channels}) != len(
        normalized_channels
    ):
        raise ValueError("Selected channel names must not contain duplicates.")
    if not isinstance(channel_mapping, dict):
        raise ValueError(f"`channel_mapping` must be a YAML mapping in {config_path}")

    normalized_mapping: dict[int, str] = {}
    for raw_index, raw_label in channel_mapping.items():
        if isinstance(raw_index, bool):
            raise ValueError("Channel mapping indices must be integers, not booleans.")
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid channel mapping index `{raw_index}`; expected an integer."
            ) from error
        if index in normalized_mapping:
            raise ValueError(f"Duplicate channel mapping index: {index}")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(f"Channel {index} must have a non-empty string label.")
        normalized_mapping[index] = raw_label.strip()

    labels_casefolded = [label.casefold() for label in normalized_mapping.values()]
    if len(labels_casefolded) != len(set(labels_casefolded)):
        raise ValueError("Channel mapping labels must be unique.")

    return {
        "feature": feature.strip(),
        "channels": normalized_channels,
        "channel_mapping": normalized_mapping,
    }


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        batch = np.asarray(values, dtype=np.float64)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + (delta ** 2) * self.count * batch_count / total
        )
        self.count = total

    @property
    def std(self) -> np.ndarray:
        if self.count <= 1:
            return np.zeros_like(self.mean)
        return np.sqrt(np.maximum(self.m2 / (self.count - 1), 0.0))


def save_curve_figure(
    stats: dict[str, np.ndarray],
    channel_labels: list[str],
    output_base: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(channel_labels))
    for name in ["mean", "std", "p50", "p95", "p99"]:
        ax.plot(x, stats[name], marker="o", linewidth=1.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_labels, rotation=45, ha="right")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Statistic value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def zscore_statistics(
    stats: dict[str, np.ndarray],
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if np.any(std < eps):
        invalid = np.flatnonzero(std < eps).tolist()
        raise ValueError(
            f"Cannot apply Z-score to near-constant channels at positions {invalid}."
        )

    normalized = {
        name: (np.asarray(values, dtype=np.float64) - mean) / (std + eps)
        for name, values in stats.items()
        if name not in {"mean", "std"}
    }
    normalized["mean"] = np.zeros_like(mean)
    normalized["std"] = std / (std + eps)
    return normalized


def select_statistics_by_group(
    stats: dict[str, np.ndarray],
    channel_labels: list[str],
    group_prefix: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    indices = [
        index
        for index, label in enumerate(channel_labels)
        if label.rsplit("_", 1)[-1].casefold().startswith(group_prefix.casefold())
    ]
    return (
        {
            name: np.asarray(values)[indices]
            for name, values in stats.items()
        },
        [channel_labels[index] for index in indices],
    )


def save_zscore_comparison_figure(
    raw_stats: dict[str, np.ndarray],
    normalized_stats: dict[str, np.ndarray],
    channel_labels: list[str],
    output_base: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    x = np.arange(len(channel_labels))
    statistic_names = ["mean", "std", "p50", "p95", "p99"]

    for name in statistic_names:
        axes[0].plot(
            x,
            raw_stats[name],
            marker="o",
            linestyle="--",
            linewidth=1.8,
            label=name,
        )
        axes[1].plot(
            x,
            normalized_stats[name],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            label=name,
        )

    axes[0].set_title("Before Z-score (dashed)")
    axes[0].set_ylabel("Raw statistic value")
    axes[1].set_title("After Z-score (solid)")
    axes[1].set_ylabel("Normalized statistic value")
    axes[1].set_xlabel("Channel")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channel_labels)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_csv(
    stats: dict[str, np.ndarray],
    channel_labels: list[str],
    output_path: Path,
) -> None:
    build_stats_dataframe(stats, channel_labels).to_csv(
        output_path,
        index=False,
    )


def build_stats_dataframe(
    stats: dict[str, np.ndarray],
    channel_labels: list[str],
) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(channel_labels):
        rows.append(
            {
                "channel_index": index,
                "channel_name": name,
                "mean": float(stats["mean"][index]),
                "std": float(stats["std"][index]),
                "p50": float(stats["p50"][index]),
                "p95": float(stats["p95"][index]),
                "p99": float(stats["p99"][index]),
            }
        )
    return pd.DataFrame(rows)


def save_interactive_html_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
  <style>
    body {{
      font-family: sans-serif;
      margin: 24px;
    }}
    h1 {{
      margin-bottom: 16px;
      font-size: 20px;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {dataframe.to_html(index=False, table_id="channel-stats", classes="display compact", border=0)}
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script>
    new DataTable('#channel-stats', {{
      paging: false,
      searching: true,
      info: false,
      order: [[0, 'asc']]
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def reshape_feature_to_channels(
    value: Any,
    feature_name: str,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim == 0:
        raise ValueError(
            f"Feature `{feature_name}` is scalar; expected at least 1-D."
        )
    if array.ndim == 1:
        return array.reshape(1, -1).astype(np.float32)
    return array.reshape(-1, array.shape[-1]).astype(np.float32)


def build_train_context(
    config_path: Path,
    cache_dir: Path,
):
    project_root = config_path.resolve().parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    os.environ["HF_HOME"] = str(cache_dir / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "hf_datasets")

    from dataloader import (  # type: ignore
        build_base_dataset,
        build_normal_dataloaders,
        get_episode_bounds,
        load_yaml,
        split_episode_ids,
    )

    config = load_yaml(config_path)
    base_dataset = build_base_dataset(config)
    _, datasets = build_normal_dataloaders(config, base_dataset)
    episode_bounds = get_episode_bounds(base_dataset)
    split_cfg = config["split"]
    train_episode_ids = split_episode_ids(
        episode_ids=sorted(episode_bounds.keys()),
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )["train"]
    return config, base_dataset, datasets["train"], episode_bounds, train_episode_ids


def iter_feature_rows(
    base_dataset,
    train_dataset,
    episode_bounds,
    train_episode_ids,
    feature_name: str,
):
    if feature_name == "tactile_history":
        for index in tqdm(range(len(train_dataset)), desc="Reading train windows", unit="window"):
            sample = train_dataset[index]
            yield sample["tactile_history"], sample
        return

    available = set(base_dataset.features.keys())
    if feature_name not in available:
        raise KeyError(
            f"Feature `{feature_name}` not found. Available keys: {sorted(available)}"
        )

    for episode_id in tqdm(train_episode_ids, desc="Reading train episodes", unit="episode"):
        start, end = episode_bounds[episode_id]
        for absolute_index in range(start, end):
            row = base_dataset.get_raw_item(absolute_index)
            yield row[feature_name], row


def infer_channel_labels(
    feature_name: str,
    channel_dim: int,
) -> list[str]:
    if feature_name == "tactile_history" and channel_dim == 12:
        return [
            "left_Fx",
            "left_Fy",
            "left_Fz",
            "left_Mx",
            "left_My",
            "left_Mz",
            "right_Fx",
            "right_Fy",
            "right_Fz",
            "right_Mx",
            "right_My",
            "right_Mz",
        ]
    if channel_dim == 6:
        return ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    return [f"ch{i}" for i in range(channel_dim)]


def select_channel_statistics(
    stats: dict[str, np.ndarray],
    channel_labels: list[str],
    selected_channels: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    label_to_index = {
        label.casefold(): index for index, label in enumerate(channel_labels)
    }
    missing = [
        channel
        for channel in selected_channels
        if channel.casefold() not in label_to_index
    ]
    if missing:
        raise ValueError(
            f"Selected channels do not match the mapping: {', '.join(missing)}. "
            f"Available channels: {', '.join(channel_labels)}"
        )

    channel_indices = [
        label_to_index[channel.casefold()] for channel in selected_channels
    ]
    selected_stats = {
        name: np.asarray(values)[channel_indices]
        for name, values in stats.items()
    }
    return selected_stats, [channel_labels[index] for index in channel_indices]


def labels_from_channel_mapping(
    channel_mapping: dict[int, str],
    channel_dim: int,
) -> list[str]:
    expected_indices = set(range(channel_dim))
    actual_indices = set(channel_mapping)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(
            "Channel mapping does not match the data dimensions. "
            f"Expected indices 0-{channel_dim - 1}; "
            f"missing={missing}, extra={extra}."
        )
    return [channel_mapping[index] for index in range(channel_dim)]


def compute_feature_statistics(
    base_dataset,
    train_dataset,
    episode_bounds,
    train_episode_ids,
    feature_name: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    running: RunningStats | None = None
    chunks: list[np.ndarray] = []

    for value, _ in iter_feature_rows(
        base_dataset=base_dataset,
        train_dataset=train_dataset,
        episode_bounds=episode_bounds,
        train_episode_ids=train_episode_ids,
        feature_name=feature_name,
    ):
        rows = reshape_feature_to_channels(value, feature_name)
        if running is None:
            running = RunningStats(rows.shape[1])
        if rows.shape[1] != running.mean.shape[0]:
            raise ValueError(
                f"Inconsistent channel dimension for `{feature_name}`: "
                f"{rows.shape[1]} vs {running.mean.shape[0]}"
            )
        running.update(rows)
        chunks.append(rows)

    if running is None:
        raise RuntimeError(f"No data found for feature `{feature_name}`.")

    all_values = np.concatenate(chunks, axis=0).astype(np.float64)
    stats = {
        "mean": running.mean.astype(np.float64),
        "std": running.std.astype(np.float64),
        "p50": np.percentile(all_values, 50, axis=0).astype(np.float64),
        "p95": np.percentile(all_values, 95, axis=0).astype(np.float64),
        "p99": np.percentile(all_values, 99, axis=0).astype(np.float64),
    }
    channel_labels = infer_channel_labels(
        feature_name=feature_name,
        channel_dim=all_values.shape[1],
    )
    return stats, channel_labels


def main() -> None:
    args = parse_args()
    analysis_config = load_analysis_config(args.analysis_config)
    args.feature = analysis_config["feature"]
    selected_channels = analysis_config["channels"]
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config, base_dataset, train_dataset, episode_bounds, train_episode_ids = build_train_context(
        config_path=args.config,
        cache_dir=args.cache_dir,
    )

    stats, channel_labels = compute_feature_statistics(
        base_dataset=base_dataset,
        train_dataset=train_dataset,
        episode_bounds=episode_bounds,
        train_episode_ids=train_episode_ids,
        feature_name=args.feature,
    )
    channel_labels = labels_from_channel_mapping(
        channel_mapping=analysis_config["channel_mapping"],
        channel_dim=len(channel_labels),
    )
    stats, channel_labels = select_channel_statistics(
        stats=stats,
        channel_labels=channel_labels,
        selected_channels=selected_channels,
    )
    normalized_stats = zscore_statistics(stats)

    csv_path = args.output_dir / f"{args.feature.replace('.', '_')}_channel_stats.csv"
    normalized_csv_path = (
        args.output_dir
        / f"{args.feature.replace('.', '_')}_channel_stats_zscore.csv"
    )
    fig_base = args.output_dir / f"{args.feature.replace('.', '_')}_channel_stats"
    dataframe = build_stats_dataframe(
        stats=stats,
        channel_labels=channel_labels,
    )
    save_csv(
        stats=stats,
        channel_labels=channel_labels,
        output_path=csv_path,
    )
    save_csv(
        stats=normalized_stats,
        channel_labels=channel_labels,
        output_path=normalized_csv_path,
    )
    save_curve_figure(
        stats=stats,
        channel_labels=channel_labels,
        output_base=fig_base,
        title=f"{args.feature} channel statistics (train split)",
    )
    comparison_paths: list[Path] = []
    for group_prefix, group_name in (("F", "force"), ("M", "moment")):
        group_raw_stats, group_labels = select_statistics_by_group(
            stats=stats,
            channel_labels=channel_labels,
            group_prefix=group_prefix,
        )
        if not group_labels:
            print(f"skip_{group_name}_figure: no selected {group_prefix} channels")
            continue
        group_normalized_stats, _ = select_statistics_by_group(
            stats=normalized_stats,
            channel_labels=channel_labels,
            group_prefix=group_prefix,
        )
        comparison_base = (
            args.output_dir
            / f"{args.feature.replace('.', '_')}_{group_name}_zscore_comparison"
        )
        save_zscore_comparison_figure(
            raw_stats=group_raw_stats,
            normalized_stats=group_normalized_stats,
            channel_labels=group_labels,
            output_base=comparison_base,
            title=f"{group_name.capitalize()} channels: before and after Z-score",
        )
        comparison_paths.append(comparison_base.with_suffix(".png"))
    html_path = args.output_dir / f"{args.feature.replace('.', '_')}_channel_stats.html"
    save_interactive_html_table(
        dataframe=dataframe,
        output_path=html_path,
        title=f"{args.feature} channel statistics (train split)",
    )

    print(f"feature: {args.feature}")
    print(f"channels: {', '.join(selected_channels)}")
    print(f"train_windows: {len(train_dataset)}")
    print(f"train_episodes: {len(train_episode_ids)}")
    print(f"dataset_repo: {config['dataset']['repo_id']}")
    print(f"csv: {csv_path}")
    print(f"zscore_csv: {normalized_csv_path}")
    print(f"figure_png: {fig_base.with_suffix('.png')}")
    for comparison_path in comparison_paths:
        print(f"zscore_comparison_png: {comparison_path}")
    print(f"table_html: {html_path}")


if __name__ == "__main__":
    main()
