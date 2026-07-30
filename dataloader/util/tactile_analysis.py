#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm import tqdm

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

try:
    from sklearn.mixture import GaussianMixture
except ModuleNotFoundError:
    GaussianMixture = None


EPS = 1e-6


@dataclass
class EpisodeData:
    episode_id: int
    frame_indices: NDArray[np.int64]
    left: NDArray[np.float32]
    right: NDArray[np.float32]
    contact: NDArray[np.float32] | None = None
    source: str = "dataset"

    @property
    def tactile(self) -> NDArray[np.float32]:
        return np.concatenate([self.left, self.right], axis=1)


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, values: NDArray[np.float32]) -> None:
        if values.size == 0:
            return
        arr = np.asarray(values, dtype=np.float64)
        batch_count = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_m2 = ((arr - batch_mean) ** 2).sum(axis=0)
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
            + (delta**2) * self.count * batch_count / total
        )
        self.count = total

    @property
    def std(self) -> NDArray[np.float64]:
        if self.count <= 1:
            return np.zeros_like(self.mean)
        return np.sqrt(np.maximum(self.m2 / (self.count - 1), 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tactile force/torque data and estimate soft thresholds tau."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../tactile_dataloader.yaml"),
        help="Path to the existing dataloader YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tactile_analysis"),
        help="Output directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/tactile_analysis_hf"),
        help="Writable cache directory for Hugging Face datasets.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Sliding window length. Defaults to the training tactile history length.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-episode-plots",
        type=int,
        default=6,
        help="How many representative episodes to visualize.",
    )
    parser.add_argument(
        "--max-window-examples",
        type=int,
        default=10,
        help="How many windows per class to visualize.",
    )
    parser.add_argument(
        "--gmm-max-samples",
        type=int,
        default=50000,
        help="Maximum number of samples used to fit the 2-component GMM.",
    )
    parser.add_argument("--tau-value", type=float, default=None)
    parser.add_argument("--tau-change", type=float, default=None)
    parser.add_argument("--tau-near-change", type=float, default=None)
    parser.add_argument("--tau-low-change", type=float, default=None)
    parser.add_argument("--tau-high-change", type=float, default=None)
    parser.add_argument("--tau-spike", type=float, default=None)
    parser.add_argument("--tau-transition-delta", type=float, default=None)
    parser.add_argument("--alpha-value", type=float, default=1.0)
    parser.add_argument("--alpha-change", type=float, default=2.0)
    parser.add_argument("--slope-value", type=float, default=5.0)
    parser.add_argument("--slope-change", type=float, default=5.0)
    parser.add_argument(
        "--synthetic-test",
        action="store_true",
        help="Run the analysis on a synthetic dataset instead of the project dataset.",
    )
    parser.add_argument(
        "--synthetic-episodes",
        type=int,
        default=12,
        help="Synthetic test episode count.",
    )
    parser.add_argument(
        "--synthetic-episode-length",
        type=int,
        default=96,
        help="Synthetic test episode length.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    paths = {
        "root": output_dir,
        "figures": figures,
        "raw_signals": figures / "raw_signals",
        "distributions": figures / "distributions",
        "window_examples": figures / "window_examples",
        "thresholds": figures / "thresholds",
        "weights": figures / "weights",
        "tables": tables,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_figure(fig: Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def percentiles(arr: NDArray[np.float64], q: Iterable[float]) -> list[float]:
    if arr.size == 0:
        return [float("nan") for _ in q]
    return [float(np.percentile(arr, item)) for item in q]


def rms(arr: NDArray[np.float32], axis: tuple[int, ...] | int | None = None) -> NDArray[np.float32]:
    return np.sqrt(np.mean(np.square(arr), axis=axis))


def mad(arr: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    med = np.median(arr, axis=axis)
    return np.median(np.abs(arr - med), axis=axis)


def flatten_for_dist(values: NDArray[np.float32], channel_names: list[str]) -> pd.DataFrame:
    data = []
    for idx, name in enumerate(channel_names):
        data.append(
            pd.DataFrame(
                {
                    "channel": name,
                    "value": values[:, idx].astype(np.float64),
                }
            )
        )
    return pd.concat(data, ignore_index=True)


def line_hist(
    ax: plt.Axes,
    values: NDArray[np.float64],
    bins: int = 80,
    label: str | None = None,
    density: bool = False,
    log_y: bool = False,
) -> None:
    counts, edges = np.histogram(values, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, linewidth=1.5, label=label)
    if log_y:
        ax.set_yscale("log")


def detect_contact_label_keys(feature_names: list[str]) -> list[str]:
    keywords = ("contact", "touch_label", "is_contact", "grasp_label")
    return [name for name in feature_names if any(word in name.lower() for word in keywords)]


def quadratic_real_roots(a: float, b: float, c: float) -> list[float]:
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [float(-c / b)]
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    sqrt_disc = math.sqrt(disc)
    return [
        float((-b - sqrt_disc) / (2.0 * a)),
        float((-b + sqrt_disc) / (2.0 * a)),
    ]


def gaussian_intersection(
    mean1: float,
    std1: float,
    weight1: float,
    mean2: float,
    std2: float,
    weight2: float,
) -> float | None:
    std1 = max(std1, 1e-8)
    std2 = max(std2, 1e-8)
    a = 1.0 / (2.0 * std2**2) - 1.0 / (2.0 * std1**2)
    b = mean1 / (std1**2) - mean2 / (std2**2)
    c = (
        mean2**2 / (2.0 * std2**2)
        - mean1**2 / (2.0 * std1**2)
        + math.log((weight2 / std2) / max(weight1 / std1, 1e-12))
    )
    roots = quadratic_real_roots(a, b, c)
    low, high = sorted((mean1, mean2))
    valid = [root for root in roots if low <= root <= high]
    if valid:
        return valid[0]
    if roots:
        roots.sort(key=lambda item: min(abs(item - low), abs(item - high)))
        return roots[0]
    return None


def format_optional_float(value: Any, precision: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{precision}f}"


def load_project_train_episodes(
    config_path: Path,
    cache_dir: Path,
    seed: int,
) -> tuple[list[EpisodeData], dict[str, Any]]:
    import os
    import sys

    project_root = config_path.resolve().parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    os.environ["HF_HOME"] = str(cache_dir / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "hf_datasets")

    from dataloader import (  # type: ignore
        DatasetKeys,
        build_base_dataset,
        get_episode_bounds,
        load_yaml,
        split_episode_ids,
    )

    config = load_yaml(config_path)
    keys = DatasetKeys(**config["dataset"]["keys"])
    dataset = build_base_dataset(config)
    feature_names = sorted(dataset.features.keys())
    episode_bounds = get_episode_bounds(dataset)
    split_cfg = config["split"]
    train_ids = split_episode_ids(
        episode_ids=sorted(episode_bounds.keys()),
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )["train"]
    tactile_keys = keys.tactile_force if isinstance(keys.tactile_force, list) else [keys.tactile_force]
    if len(tactile_keys) != 2:
        raise RuntimeError(
            f"Expected exactly two tactile force keys, got {tactile_keys}."
        )
    contact_candidates = detect_contact_label_keys(feature_names)
    contact_key = contact_candidates[0] if contact_candidates else None

    episodes: list[EpisodeData] = []
    for episode_id in tqdm(train_ids, desc="Loading train episodes", unit="episode"):
        start, end = episode_bounds[episode_id]
        left_frames: list[NDArray[np.float32]] = []
        right_frames: list[NDArray[np.float32]] = []
        frame_indices: list[int] = []
        contact_values: list[float] = []
        for absolute_index in range(start, end):
            row = dataset.get_raw_item(absolute_index)
            left = np.asarray(row[tactile_keys[0]], dtype=np.float32)
            right = np.asarray(row[tactile_keys[1]], dtype=np.float32)
            left_frames.append(left)
            right_frames.append(right)
            frame_indices.append(int(row["frame_index"]))
            if contact_key is not None:
                value = row[contact_key]
                array = np.asarray(value, dtype=np.float32).reshape(-1)
                contact_values.append(float(array[0]))
        if not left_frames:
            episodes.append(
                EpisodeData(
                    episode_id=int(episode_id),
                    frame_indices=np.zeros(0, dtype=np.int64),
                    left=np.zeros((0, 6), dtype=np.float32),
                    right=np.zeros((0, 6), dtype=np.float32),
                    contact=None,
                )
            )
            continue
        episodes.append(
            EpisodeData(
                episode_id=int(episode_id),
                frame_indices=np.asarray(frame_indices, dtype=np.int64),
                left=np.stack(left_frames).astype(np.float32),
                right=np.stack(right_frames).astype(np.float32),
                contact=(
                    np.asarray(contact_values, dtype=np.float32)
                    if contact_values
                    else None
                ),
            )
        )

    dataset_root = Path(dataset.root)
    file_count = len(list((dataset_root / "data").rglob("*.parquet")))
    context = {
        "config": config,
        "keys": keys,
        "dataset_root": str(dataset_root),
        "file_count": file_count,
        "feature_names": feature_names,
        "contact_key": contact_key,
        "fps": int(dataset.fps),
    }
    return episodes, context


def make_synthetic_episodes(
    num_episodes: int,
    episode_length: int,
    seed: int,
) -> tuple[list[EpisodeData], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    episodes: list[EpisodeData] = []
    for episode_id in range(num_episodes):
        left = np.zeros((episode_length, 6), dtype=np.float32)
        right = np.zeros((episode_length, 6), dtype=np.float32)
        contact = np.zeros(episode_length, dtype=np.float32)
        state = 0
        for frame in range(episode_length):
            if frame % 24 == 0:
                state = (state + 1) % 4
            noise = rng.normal(0.0, 0.015, size=(2, 6)).astype(np.float32)
            if state == 0:
                contact[frame] = 0.0
                base_l = np.array([0.02, -0.01, -0.03, 0.01, 0.0, -0.01], dtype=np.float32)
                base_r = np.array([0.01, 0.00, -0.02, 0.0, -0.01, 0.01], dtype=np.float32)
            elif state == 1:
                contact[frame] = 1.0
                ramp = (frame % 24) / 24.0
                base_l = np.array([0.25, -0.15, -0.55, 0.12, 0.07, -0.06], dtype=np.float32) * ramp
                base_r = np.array([0.18, -0.08, -0.42, 0.08, -0.05, 0.03], dtype=np.float32) * ramp
            elif state == 2:
                contact[frame] = 1.0
                base_l = np.array([0.55, -0.30, -1.05, 0.20, 0.18, -0.13], dtype=np.float32)
                base_r = np.array([0.40, -0.20, -0.82, 0.15, -0.11, 0.08], dtype=np.float32)
            else:
                contact[frame] = 1.0
                base_l = np.array([0.55, -0.30, -1.05, 0.20, 0.18, -0.13], dtype=np.float32)
                base_r = np.array([0.40, -0.20, -0.82, 0.15, -0.11, 0.08], dtype=np.float32)
                if frame % 11 == 0:
                    base_l += np.array([0.35, 0.0, -0.45, 0.18, 0.0, 0.0], dtype=np.float32)
            left[frame] = base_l + noise[0]
            right[frame] = base_r + noise[1]
        if episode_id % 4 == 0:
            left[5:10] = left[4]
            right[5:10] = right[4]
        if episode_id % 5 == 0:
            left[30] = 0.0
            right[30] = 0.0
        episodes.append(
            EpisodeData(
                episode_id=episode_id,
                frame_indices=np.arange(episode_length, dtype=np.int64),
                left=left,
                right=right,
                contact=contact,
                source="synthetic",
            )
        )
    context = {
        "config": {"dataset": {"repo_id": "synthetic"}},
        "keys": None,
        "dataset_root": "synthetic",
        "file_count": num_episodes,
        "feature_names": ["observation.tactile.left_force", "observation.tactile.right_force", "contact"],
        "contact_key": "contact",
        "fps": 30,
    }
    return episodes, context


def raw_statistics(
    episodes: list[EpisodeData],
    channel_names: list[str],
    warnings_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    running = RunningStats(dim=len(channel_names))
    channel_chunks: list[NDArray[np.float32]] = []
    episode_rows: list[dict[str, Any]] = []
    total_frames = 0
    total_zero_frames = 0
    total_repeat_transitions = 0
    total_transitions = 0
    total_nan = 0
    total_inf = 0
    left_shapes: set[tuple[int, ...]] = set()
    right_shapes: set[tuple[int, ...]] = set()

    for episode in tqdm(episodes, desc="Pass 1/2 raw stats", unit="episode"):
        left_shapes.add(tuple(episode.left.shape[1:]))
        right_shapes.add(tuple(episode.right.shape[1:]))
        if episode.left.shape[0] != episode.right.shape[0]:
            warnings_list.append(
                f"Episode {episode.episode_id}: left/right length mismatch "
                f"{episode.left.shape[0]} vs {episode.right.shape[0]}."
            )
        tactile = episode.tactile.astype(np.float32)
        if tactile.size == 0:
            warnings_list.append(f"Episode {episode.episode_id}: empty episode.")
            episode_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "num_frames": 0,
                    "nan_count": 0,
                    "inf_count": 0,
                    "zero_frame_ratio": float("nan"),
                    "repeated_transition_ratio": float("nan"),
                    "value_rms": float("nan"),
                    "change_rms": float("nan"),
                }
            )
            continue
        nan_mask = np.isnan(tactile)
        inf_mask = np.isinf(tactile)
        nan_count = int(nan_mask.sum())
        inf_count = int(inf_mask.sum())
        total_nan += nan_count
        total_inf += inf_count
        if nan_count > 0:
            warnings_list.append(f"Episode {episode.episode_id}: found {nan_count} NaN values.")
        if inf_count > 0:
            warnings_list.append(f"Episode {episode.episode_id}: found {inf_count} Inf values.")
        finite = np.where(np.isfinite(tactile), tactile, 0.0)
        channel_chunks.append(finite)
        running.update(finite)
        zero_frames = np.all(np.abs(finite) <= EPS, axis=1)
        diff1 = finite[1:] - finite[:-1]
        repeat_transitions = np.all(np.abs(diff1) <= EPS, axis=1) if len(finite) > 1 else np.zeros(0, dtype=bool)
        total_frames += finite.shape[0]
        total_zero_frames += int(zero_frames.sum())
        total_repeat_transitions += int(repeat_transitions.sum())
        total_transitions += int(max(finite.shape[0] - 1, 0))
        episode_rows.append(
            {
                "episode_id": episode.episode_id,
                "num_frames": int(finite.shape[0]),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "zero_frame_ratio": float(zero_frames.mean()),
                "repeated_transition_ratio": (
                    float(repeat_transitions.mean()) if repeat_transitions.size > 0 else float("nan")
                ),
                "value_rms": float(rms(finite)),
                "change_rms": float(rms(diff1)) if diff1.size > 0 else 0.0,
            }
        )

    all_values = np.concatenate(channel_chunks, axis=0) if channel_chunks else np.zeros((0, len(channel_names)), dtype=np.float32)
    channel_rows = []
    quantile_levels = [1, 5, 25, 75, 95, 99]
    medians = np.median(all_values, axis=0) if all_values.size else np.zeros(len(channel_names))
    mads = mad(all_values.astype(np.float64), axis=0) if all_values.size else np.zeros(len(channel_names))
    for idx, name in enumerate(channel_names):
        column = all_values[:, idx].astype(np.float64)
        q = percentiles(column, quantile_levels)
        channel_rows.append(
            {
                "channel": name,
                "min": float(np.min(column)) if column.size else float("nan"),
                "max": float(np.max(column)) if column.size else float("nan"),
                "mean": float(running.mean[idx]),
                "std": float(running.std[idx]),
                "median": float(medians[idx]),
                "mad": float(mads[idx]),
                "p1": q[0],
                "p5": q[1],
                "p25": q[2],
                "p75": q[3],
                "p95": q[4],
                "p99": q[5],
                "nan_count": int(np.isnan(column).sum()),
                "inf_count": int(np.isinf(column).sum()),
            }
        )

    if left_shapes != {(6,)}:
        warnings_list.append(f"Unexpected left tactile shapes: {sorted(left_shapes)}")
    if right_shapes != {(6,)}:
        warnings_list.append(f"Unexpected right tactile shapes: {sorted(right_shapes)}")

    summary = {
        "total_frames": total_frames,
        "num_episodes": len(episodes),
        "nan_count": total_nan,
        "inf_count": total_inf,
        "zero_frame_ratio": float(total_zero_frames / max(total_frames, 1)),
        "repeated_transition_ratio": float(total_repeat_transitions / max(total_transitions, 1)),
        "channel_mean": running.mean.astype(float).tolist(),
        "channel_std": running.std.astype(float).tolist(),
        "channel_median": medians.astype(float).tolist(),
        "channel_mad": mads.astype(float).tolist(),
        "all_values": all_values,
    }
    return pd.DataFrame(channel_rows), pd.DataFrame(episode_rows), summary


def construct_windows(
    episodes: list[EpisodeData],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    window_length: int,
    stride: int,
) -> dict[str, Any]:
    windows_raw: list[NDArray[np.float32]] = []
    windows_norm: list[NDArray[np.float32]] = []
    metadata_rows: list[dict[str, Any]] = []
    contacts: list[float] = []
    left_value = []
    right_value = []
    force_value = []
    torque_value = []
    channel_diff_mag = []
    value_mag = []
    change_mag = []
    second_mag = []
    start_value_mag = []
    end_value_mag = []
    endpoint_delta = []

    for episode in tqdm(episodes, desc="Pass 2/2 windows", unit="episode"):
        tactile = episode.tactile.astype(np.float32)
        if tactile.shape[0] < window_length:
            continue
        norm = (tactile - mean.astype(np.float32)) / (std.astype(np.float32) + EPS)
        for start in range(0, tactile.shape[0] - window_length + 1, stride):
            end = start + window_length
            raw_window = tactile[start:end]
            norm_window = norm[start:end]
            diff1 = norm_window[1:] - norm_window[:-1]
            diff2 = norm_window[2:] - 2.0 * norm_window[1:-1] + norm_window[:-2]
            windows_raw.append(raw_window.astype(np.float32))
            windows_norm.append(norm_window.astype(np.float32))
            value = float(rms(norm_window))
            change = float(rms(diff1)) if diff1.size else 0.0
            second = float(rms(diff2)) if diff2.size else 0.0
            left_norm = norm_window[:, :6]
            right_norm = norm_window[:, 6:]
            force_norm = norm_window[:, [0, 1, 2, 6, 7, 8]]
            torque_norm = norm_window[:, [3, 4, 5, 9, 10, 11]]
            ch_diff = rms(diff1, axis=0) if diff1.size else np.zeros(12, dtype=np.float32)
            start_value = float(rms(norm_window[: max(1, window_length // 4)]))
            end_value = float(rms(norm_window[-max(1, window_length // 4):]))
            delta_value = float(rms(norm_window[-1] - norm_window[0]))
            metadata_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "start_frame_index": int(episode.frame_indices[start]),
                    "end_frame_index": int(episode.frame_indices[end - 1]),
                    "window_index_in_episode": int(start // stride),
                }
            )
            value_mag.append(value)
            change_mag.append(change)
            second_mag.append(second)
            left_value.append(float(rms(left_norm)))
            right_value.append(float(rms(right_norm)))
            force_value.append(float(rms(force_norm)))
            torque_value.append(float(rms(torque_norm)))
            channel_diff_mag.append(ch_diff.astype(np.float32))
            start_value_mag.append(start_value)
            end_value_mag.append(end_value)
            endpoint_delta.append(delta_value)
            if episode.contact is not None:
                contacts.append(float(np.mean(episode.contact[start:end])))
            else:
                contacts.append(float("nan"))

    windows_raw_arr = np.stack(windows_raw).astype(np.float32) if windows_raw else np.zeros((0, window_length, 12), dtype=np.float32)
    windows_norm_arr = np.stack(windows_norm).astype(np.float32) if windows_norm else np.zeros((0, window_length, 12), dtype=np.float32)
    return {
        "windows_raw": windows_raw_arr,
        "windows_norm": windows_norm_arr,
        "metadata": pd.DataFrame(metadata_rows),
        "value_magnitude": np.asarray(value_mag, dtype=np.float64),
        "change_magnitude": np.asarray(change_mag, dtype=np.float64),
        "second_diff_magnitude": np.asarray(second_mag, dtype=np.float64),
        "left_value_magnitude": np.asarray(left_value, dtype=np.float64),
        "right_value_magnitude": np.asarray(right_value, dtype=np.float64),
        "force_value_magnitude": np.asarray(force_value, dtype=np.float64),
        "torque_value_magnitude": np.asarray(torque_value, dtype=np.float64),
        "channel_diff_magnitude": np.stack(channel_diff_mag).astype(np.float64) if channel_diff_mag else np.zeros((0, 12), dtype=np.float64),
        "contact_ratio": np.asarray(contacts, dtype=np.float64),
        "start_value_magnitude": np.asarray(start_value_mag, dtype=np.float64),
        "end_value_magnitude": np.asarray(end_value_mag, dtype=np.float64),
        "endpoint_delta_magnitude": np.asarray(endpoint_delta, dtype=np.float64),
    }


def estimate_tau_value(
    value_magnitude: NDArray[np.float64],
    contact_ratio: NDArray[np.float64],
    rng: np.random.Generator,
    gmm_max_samples: int,
) -> dict[str, Any]:
    has_label = np.isfinite(contact_ratio).any()
    result: dict[str, Any] = {"has_contact_labels": False}
    if has_label:
        no_contact = value_magnitude[contact_ratio <= 0.05]
        result["has_contact_labels"] = True
        result["no_contact_percentiles"] = {
            "p90": float(np.percentile(no_contact, 90)),
            "p95": float(np.percentile(no_contact, 95)),
            "p97_5": float(np.percentile(no_contact, 97.5)),
            "p99": float(np.percentile(no_contact, 99)),
        }
        result["tau_value"] = result["no_contact_percentiles"]["p95"]
        return result

    candidate = value_magnitude
    if candidate.size > gmm_max_samples:
        indices = rng.choice(candidate.size, size=gmm_max_samples, replace=False)
        fit_values = candidate[indices]
    else:
        fit_values = candidate
    if GaussianMixture is not None:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gmm.fit(fit_values.reshape(-1, 1))
        means = gmm.means_.reshape(-1)
        vars_ = gmm.covariances_.reshape(-1)
        weights = gmm.weights_.reshape(-1)
        order = np.argsort(means)
        means = means[order]
        vars_ = vars_[order]
        weights = weights[order]
        intersection = gaussian_intersection(
            float(means[0]),
            float(np.sqrt(vars_[0])),
            float(weights[0]),
            float(means[1]),
            float(np.sqrt(vars_[1])),
            float(weights[1]),
        )
        result["gmm"] = {
            "low_component_mean": float(means[0]),
            "low_component_var": float(vars_[0]),
            "low_component_weight": float(weights[0]),
            "high_component_mean": float(means[1]),
            "high_component_var": float(vars_[1]),
            "high_component_weight": float(weights[1]),
            "intersection": float(intersection) if intersection is not None else None,
        }
    else:
        result["gmm"] = {
            "low_component_mean": None,
            "low_component_var": None,
            "low_component_weight": None,
            "high_component_mean": None,
            "high_component_var": None,
            "high_component_weight": None,
            "intersection": None,
        }
        intersection = None
    result["candidate_percentiles"] = {
        "p50": float(np.percentile(candidate, 50)),
        "p70": float(np.percentile(candidate, 70)),
        "p80": float(np.percentile(candidate, 80)),
        "p90": float(np.percentile(candidate, 90)),
        "p95": float(np.percentile(candidate, 95)),
        "p97_5": float(np.percentile(candidate, 97.5)),
        "p99": float(np.percentile(candidate, 99)),
    }
    result["tau_value"] = float(intersection) if intersection is not None else result["candidate_percentiles"]["p80"]
    return result


def classify_windows(
    windows_raw: NDArray[np.float32],
    windows_norm: NDArray[np.float32],
    metrics: dict[str, NDArray[np.float64]],
    thresholds: dict[str, float],
    contact_ratio: NDArray[np.float64],
) -> pd.DataFrame:
    diff1 = windows_raw[:, 1:] - windows_raw[:, :-1] if len(windows_raw) else np.zeros((0, 0, 12), dtype=np.float32)
    exact_no_change = np.all(np.abs(diff1) <= EPS, axis=(1, 2)) if diff1.size else np.zeros(len(windows_raw), dtype=bool)
    value = metrics["value_magnitude"]
    change = metrics["change_magnitude"]
    second = metrics["second_diff_magnitude"]
    start_value = metrics["start_value_magnitude"]
    end_value = metrics["end_value_magnitude"]
    endpoint_delta = metrics["endpoint_delta_magnitude"]
    approx_no_change = change <= thresholds["tau_near_change"]
    low_change = change <= thresholds["tau_low_change"]
    high_change = change >= thresholds["tau_high_change"]
    maybe_no_contact = value <= thresholds["tau_value"]
    stable_contact = (value > thresholds["tau_value"]) & (change <= thresholds["tau_change"])
    contact_start = (
        (start_value <= thresholds["tau_value"])
        & (end_value > thresholds["tau_value"])
        & (endpoint_delta >= thresholds["tau_transition_delta"])
    )
    contact_end = (
        (start_value > thresholds["tau_value"])
        & (end_value <= thresholds["tau_value"])
        & (endpoint_delta >= thresholds["tau_transition_delta"])
    )
    spike = second >= thresholds["tau_spike"]
    labeled_no_contact = np.isfinite(contact_ratio) & (contact_ratio <= 0.05)
    labeled_contact = np.isfinite(contact_ratio) & (contact_ratio >= 0.95)
    return pd.DataFrame(
        {
            "exact_no_change": exact_no_change,
            "approx_no_change": approx_no_change,
            "low_change": low_change,
            "high_change": high_change,
            "maybe_no_contact": maybe_no_contact,
            "stable_contact": stable_contact,
            "contact_start": contact_start,
            "contact_end": contact_end,
            "spike_or_outlier": spike,
            "labeled_no_contact": labeled_no_contact,
            "labeled_contact": labeled_contact,
        }
    )


def make_window_statistics_table(
    metadata: pd.DataFrame,
    metrics: dict[str, NDArray[np.float64]],
    classes: pd.DataFrame,
    channel_names: list[str],
) -> pd.DataFrame:
    table = metadata.copy()
    table["value_magnitude"] = metrics["value_magnitude"]
    table["change_magnitude"] = metrics["change_magnitude"]
    table["second_diff_magnitude"] = metrics["second_diff_magnitude"]
    table["left_value_magnitude"] = metrics["left_value_magnitude"]
    table["right_value_magnitude"] = metrics["right_value_magnitude"]
    table["force_value_magnitude"] = metrics["force_value_magnitude"]
    table["torque_value_magnitude"] = metrics["torque_value_magnitude"]
    table["contact_ratio"] = metrics["contact_ratio"]
    for idx, name in enumerate(channel_names):
        table[f"{name}_diff_mag"] = metrics["channel_diff_magnitude"][:, idx]
    for col in classes.columns:
        table[col] = classes[col].astype(bool)
    return table


def channel_order_assessment(context: dict[str, Any]) -> list[str]:
    notes = []
    notes.append("Observed both left and right tactile tensors as float32 with shape [num_frames, 6].")
    notes.append("The analysis script uses the assumed order [Fx, Fy, Fz, Mx, My, Mz] from the user specification.")
    notes.append("No explicit unit or channel-order metadata was found in the dataset feature schema, so unit consistency cannot be programmatically proven.")
    return notes


def plot_episode_family(
    episode: EpisodeData,
    out_dir: Path,
    prefix: str,
) -> None:
    groups = [
        ("left_force", episode.left[:, :3], ["Fx", "Fy", "Fz"]),
        ("left_torque", episode.left[:, 3:], ["Mx", "My", "Mz"]),
        ("right_force", episode.right[:, :3], ["Fx", "Fy", "Fz"]),
        ("right_torque", episode.right[:, 3:], ["Mx", "My", "Mz"]),
    ]
    time = np.arange(episode.left.shape[0])
    for group_name, values, names in groups:
        for variant, data in [
            ("raw", values),
            ("diff1", np.diff(values, axis=0)),
            ("diff2", np.diff(values, n=2, axis=0)),
        ]:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            if data.shape[0] == 0:
                plt.close(fig)
                continue
            xs = time[: data.shape[0]]
            for idx in range(3):
                axes[idx].plot(xs, data[:, idx], linewidth=1.2)
                axes[idx].set_ylabel(names[idx])
            axes[-1].set_xlabel("Frame")
            fig.suptitle(f"Episode {episode.episode_id} {group_name} {variant}")
            save_figure(fig, out_dir / f"{prefix}_{group_name}_{variant}")


def plot_global_distributions(
    all_values: NDArray[np.float32],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    median: NDArray[np.float64],
    mad_values: NDArray[np.float64],
    channel_names: list[str],
    out_dir: Path,
) -> None:
    left_channels = channel_names[:6]
    right_channels = channel_names[6:]
    groups = [
        ("left_force", all_values[:, :3], left_channels[:3]),
        ("left_torque", all_values[:, 3:6], left_channels[3:]),
        ("right_force", all_values[:, 6:9], right_channels[:3]),
        ("right_torque", all_values[:, 9:12], right_channels[3:]),
    ]
    for group_name, values, names in groups:
        df = flatten_for_dist(values, names)
        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.histplot(data=df, x="value", hue="channel", element="step", stat="density", common_norm=False, ax=ax)
        else:
            for name in names:
                line_hist(ax, df.loc[df["channel"] == name, "value"].to_numpy(), density=True, label=name)
            ax.legend()
        ax.set_title(f"{group_name} histogram")
        save_figure(fig, out_dir / f"{group_name}_hist")

        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.histplot(data=df, x="value", hue="channel", element="step", stat="count", common_norm=False, log_scale=(False, True), ax=ax)
        else:
            for name in names:
                line_hist(ax, df.loc[df["channel"] == name, "value"].to_numpy(), density=False, log_y=True, label=name)
            ax.legend()
        ax.set_title(f"{group_name} histogram log-y")
        save_figure(fig, out_dir / f"{group_name}_hist_logy")

        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.boxplot(data=df, x="channel", y="value", ax=ax)
        else:
            grouped = [df.loc[df["channel"] == name, "value"].to_numpy() for name in names]
            ax.boxplot(grouped)
            ax.set_xticks(np.arange(1, len(names) + 1))
            ax.set_xticklabels(names)
        ax.set_title(f"{group_name} boxplot")
        save_figure(fig, out_dir / f"{group_name}_box")

        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.violinplot(data=df, x="channel", y="value", inner="quartile", cut=0, ax=ax)
        else:
            grouped = [df.loc[df["channel"] == name, "value"].to_numpy() for name in names]
            parts = ax.violinplot(grouped, showmeans=False, showmedians=True)
            for body in parts["bodies"]:
                body.set_alpha(0.5)
            ax.set_xticks(np.arange(1, len(names) + 1))
            ax.set_xticklabels(names)
        ax.set_title(f"{group_name} violin")
        save_figure(fig, out_dir / f"{group_name}_violin")

    normalized = (all_values - mean.astype(np.float32)) / (std.astype(np.float32) + EPS)
    robust = (all_values - median.astype(np.float32)) / (mad_values.astype(np.float32) + EPS)
    for title, values in [("raw", all_values), ("mean_std_normalized", normalized), ("median_mad_normalized", robust)]:
        df = flatten_for_dist(values, channel_names)
        fig, ax = plt.subplots(figsize=(12, 6))
        if sns is not None:
            sns.histplot(
                data=df,
                x="value",
                hue="channel",
                element="step",
                stat="density",
                common_norm=False,
                ax=ax,
                legend=False,
            )
        else:
            for name in channel_names:
                line_hist(ax, df.loc[df["channel"] == name, "value"].to_numpy(), density=True)
        ax.set_title(f"All channels {title}")
        save_figure(fig, out_dir / f"all_channels_{title}")


def plot_thresholds(
    value_magnitude: NDArray[np.float64],
    change_magnitude: NDArray[np.float64],
    second_diff_magnitude: NDArray[np.float64],
    tau_info: dict[str, Any],
    thresholds: dict[str, float],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if sns is not None:
        sns.histplot(value_magnitude, bins=80, stat="density", kde=True, ax=ax)
    else:
        line_hist(ax, value_magnitude, bins=80, density=True)
    ax.axvline(thresholds["tau_value"], color="red", linestyle="--", label=f"tau_value={thresholds['tau_value']:.4f}")
    if tau_info.get("gmm", {}).get("intersection") is not None:
        ax.axvline(float(tau_info["gmm"]["intersection"]), color="black", linestyle=":", label="GMM intersection")
    ax.set_title("Value magnitude distribution")
    ax.legend()
    save_figure(fig, out_dir / "value_magnitude_hist")

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns is not None:
        sns.histplot(change_magnitude, bins=80, stat="density", kde=True, ax=ax)
    else:
        line_hist(ax, change_magnitude, bins=80, density=True)
    ax.axvline(thresholds["tau_change"], color="red", linestyle="--", label=f"tau_change={thresholds['tau_change']:.4f}")
    ax.axvline(thresholds["tau_low_change"], color="green", linestyle=":", label="tau_low_change")
    ax.axvline(thresholds["tau_high_change"], color="purple", linestyle=":", label="tau_high_change")
    ax.set_title("Change magnitude distribution")
    ax.legend()
    save_figure(fig, out_dir / "change_magnitude_hist")

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns is not None:
        sns.histplot(second_diff_magnitude, bins=80, stat="density", kde=True, ax=ax)
    else:
        line_hist(ax, second_diff_magnitude, bins=80, density=True)
    ax.axvline(thresholds["tau_spike"], color="red", linestyle="--", label=f"tau_spike={thresholds['tau_spike']:.4f}")
    ax.set_title("Second-difference magnitude distribution")
    ax.legend()
    save_figure(fig, out_dir / "second_diff_magnitude_hist")


def plot_window_examples(
    windows_raw: NDArray[np.float32],
    window_table: pd.DataFrame,
    class_name: str,
    out_dir: Path,
    max_examples: int,
    rng: np.random.Generator,
) -> None:
    mask = window_table[class_name].to_numpy(dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return
    chosen = indices if indices.size <= max_examples else rng.choice(indices, size=max_examples, replace=False)
    for example_idx, window_idx in enumerate(sorted(chosen.tolist())):
        window = windows_raw[window_idx]
        row = window_table.iloc[window_idx]
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(window[:, :3])
        axes[0].set_ylabel("Left F")
        axes[1].plot(window[:, 3:6])
        axes[1].set_ylabel("Left M")
        axes[2].plot(window[:, 6:9])
        axes[2].set_ylabel("Right F")
        axes[3].plot(window[:, 9:12])
        axes[3].set_ylabel("Right M")
        axes[3].set_xlabel("Frame")
        fig.suptitle(
            f"{class_name} example {example_idx} | "
            f"episode={int(row['episode_id'])} start={int(row['start_frame_index'])} "
            f"value={row['value_magnitude']:.4f} change={row['change_magnitude']:.4f}"
        )
        save_figure(fig, out_dir / f"{class_name}_{example_idx:02d}")


def compute_weights(
    value_magnitude: NDArray[np.float64],
    change_magnitude: NDArray[np.float64],
    tau_value: float,
    tau_change: float,
    alpha_value: float,
    alpha_change: float,
    slope_value: float,
    slope_change: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    w_value = 1.0 + alpha_value * sigmoid(
        slope_value * (value_magnitude - tau_value) / (tau_value + EPS)
    )
    w_combined = (
        1.0
        + alpha_value * sigmoid(
            slope_value * (value_magnitude - tau_value) / (tau_value + EPS)
        )
        + alpha_change * sigmoid(
            slope_change * (change_magnitude - tau_change) / (tau_change + EPS)
        )
    )
    return w_value, w_combined


def plot_weights(
    value_magnitude: NDArray[np.float64],
    change_magnitude: NDArray[np.float64],
    weights: NDArray[np.float64],
    metadata: pd.DataFrame,
    thresholds: dict[str, float],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, float]:
    xs = np.linspace(0.0, max(np.max(value_magnitude), thresholds["tau_value"] * 2.0, 1e-3), 400)
    ys = 1.0 + args.alpha_value * sigmoid(
        args.slope_value * (xs - thresholds["tau_value"]) / (thresholds["tau_value"] + EPS)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys)
    ax.axvline(thresholds["tau_value"], color="red", linestyle="--", label="tau_value")
    ax.set_title("Value magnitude to weight")
    ax.set_xlabel("value_magnitude")
    ax.set_ylabel("w_value")
    ax.legend()
    save_figure(fig, out_dir / "value_weight_curve")

    fig, ax = plt.subplots(figsize=(10, 6))
    for tau_multiplier in [0.9, 1.0, 1.1]:
        tau = thresholds["tau_value"] * tau_multiplier
        y = 1.0 + args.alpha_value * sigmoid(args.slope_value * (xs - tau) / (tau + EPS))
        ax.plot(xs, y, label=f"tau={tau:.4f}")
    ax.set_title("Tau comparison")
    ax.legend()
    save_figure(fig, out_dir / "tau_comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    for slope in [2.0, 5.0, 10.0]:
        y = 1.0 + args.alpha_value * sigmoid(slope * (xs - thresholds["tau_value"]) / (thresholds["tau_value"] + EPS))
        ax.plot(xs, y, label=f"slope={slope:g}")
    ax.set_title("Slope comparison")
    ax.legend()
    save_figure(fig, out_dir / "slope_comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns is not None:
        sns.histplot(weights, bins=80, ax=ax)
    else:
        line_hist(ax, weights, bins=80)
    ax.set_title("Combined weight histogram")
    save_figure(fig, out_dir / "weight_hist")

    if len(value_magnitude) > 0:
        sample = np.arange(len(value_magnitude))
        if len(sample) > 15000:
            rng = np.random.default_rng(0)
            sample = rng.choice(sample, size=15000, replace=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(value_magnitude[sample], weights[sample], s=8, alpha=0.35)
        ax.set_xlabel("value_magnitude")
        ax.set_ylabel("weight")
        ax.set_title("Weight vs value magnitude")
        save_figure(fig, out_dir / "weight_vs_value")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(change_magnitude[sample], weights[sample], s=8, alpha=0.35)
        ax.set_xlabel("change_magnitude")
        ax.set_ylabel("weight")
        ax.set_title("Weight vs change magnitude")
        save_figure(fig, out_dir / "weight_vs_change")

    for episode_id in metadata["episode_id"].drop_duplicates().head(4):
        mask = metadata["episode_id"] == episode_id
        fig, ax = plt.subplots(figsize=(12, 4))
        xs_ep = metadata.loc[mask, "start_frame_index"].to_numpy()
        ax.plot(xs_ep, weights[mask.to_numpy()], linewidth=1.2)
        ax.set_title(f"Weight over episode time: episode {int(episode_id)}")
        ax.set_xlabel("start_frame_index")
        ax.set_ylabel("weight")
        save_figure(fig, out_dir / f"weight_time_episode_{int(episode_id)}")

    return {
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "p50": float(np.percentile(weights, 50)),
        "p90": float(np.percentile(weights, 90)),
        "p95": float(np.percentile(weights, 95)),
        "p99": float(np.percentile(weights, 99)),
    }


def compare_threshold_settings(
    value_magnitude: NDArray[np.float64],
    change_magnitude: NDArray[np.float64],
    classes: pd.DataFrame,
    tau_candidates: list[float],
    tau_change: float,
    alpha_value: float,
    alpha_change: float,
    slopes: list[float],
) -> pd.DataFrame:
    rows = []
    no_contact_mask = classes["maybe_no_contact"].to_numpy(dtype=bool)
    stable_mask = classes["stable_contact"].to_numpy(dtype=bool)
    tactile_change_mask = (
        classes["high_change"].to_numpy(dtype=bool)
        | classes["contact_start"].to_numpy(dtype=bool)
        | classes["contact_end"].to_numpy(dtype=bool)
        | classes["spike_or_outlier"].to_numpy(dtype=bool)
    )
    for tau in tau_candidates:
        for slope in slopes:
            _, weights = compute_weights(
                value_magnitude=value_magnitude,
                change_magnitude=change_magnitude,
                tau_value=tau,
                tau_change=tau_change,
                alpha_value=alpha_value,
                alpha_change=alpha_change,
                slope_value=slope,
                slope_change=slope,
            )
            top10 = np.percentile(weights, 90)
            top_mask = weights >= top10
            rows.append(
                {
                    "tau_value": float(tau),
                    "slope": float(slope),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                    "weight_mean": float(np.mean(weights)),
                    "weight_std": float(np.std(weights)),
                    "effective_samples": float((weights.sum() ** 2) / np.square(weights).sum()),
                    "top10pct_weight_share": float(weights[top_mask].sum() / max(weights.sum(), EPS)),
                    "mean_weight_no_contact": float(np.mean(weights[no_contact_mask])) if no_contact_mask.any() else float("nan"),
                    "mean_weight_stable_contact": float(np.mean(weights[stable_mask])) if stable_mask.any() else float("nan"),
                    "mean_weight_tactile_change": float(np.mean(weights[tactile_change_mask])) if tactile_change_mask.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    context: dict[str, Any],
    raw_summary: dict[str, Any],
    channel_stats: pd.DataFrame,
    thresholds: dict[str, float],
    tau_info: dict[str, Any],
    class_summary: pd.DataFrame,
    weight_summary: dict[str, float],
    threshold_table: pd.DataFrame,
    warnings_list: list[str],
) -> None:
    low_value_ratio = float(class_summary.loc[class_summary["class_name"] == "maybe_no_contact", "ratio"].iloc[0])
    low_change_ratio = float(class_summary.loc[class_summary["class_name"] == "low_change", "ratio"].iloc[0])
    recommended_row = threshold_table.sort_values(
        ["top10pct_weight_share", "weight_mean"], ascending=[True, True]
    ).iloc[0]
    lines = [
        "# Tactile Analysis Report",
        "",
        "## Data quality",
        f"- Dataset root: `{context['dataset_root']}`",
        f"- Training files scanned: {context['file_count']}",
        f"- Training episodes: {raw_summary['num_episodes']}",
        f"- Training frames: {raw_summary['total_frames']}",
        f"- NaN values: {raw_summary['nan_count']}",
        f"- Inf values: {raw_summary['inf_count']}",
        f"- All-zero frame ratio: {raw_summary['zero_frame_ratio']:.6f}",
        f"- Repeated-transition ratio: {raw_summary['repeated_transition_ratio']:.6f}",
        "",
        "## Outliers and anomalies",
        f"- Largest per-channel standard deviation: {channel_stats['std'].max():.6f}",
        f"- Largest absolute channel value: {max(abs(channel_stats['min']).max(), abs(channel_stats['max']).max()):.6f}",
        f"- Spike threshold (tau_spike): {thresholds['tau_spike']:.6f}",
        "",
        "## Window composition",
        f"- Low-value / maybe-no-contact window ratio: {low_value_ratio:.6f}",
        f"- Low-change window ratio: {low_change_ratio:.6f}",
        f"- Recommended tau_value: {thresholds['tau_value']:.6f}",
        f"- Recommended tau_change: {thresholds['tau_change']:.6f}",
        f"- Recommended weight parameters: alpha_value={thresholds['alpha_value']:.3f}, alpha_change={thresholds['alpha_change']:.3f}, slope_value={thresholds['slope_value']:.3f}, slope_change={thresholds['slope_change']:.3f}",
        "",
        "## Rationale",
    ]
    if tau_info.get("has_contact_labels"):
        p = tau_info["no_contact_percentiles"]
        lines.extend(
            [
                f"- Contact labels were detected. `tau_value` uses P95 of no-contact windows: {p['p95']:.6f}.",
                f"- `tau_change` uses P95 of low-change windows inside the training split: {thresholds['tau_change']:.6f}.",
            ]
        )
    else:
        gmm = tau_info.get("gmm", {})
        lines.extend(
            [
                "- No reliable contact labels were detected, so the analysis used unsupervised low-value/high-value components.",
                f"- GMM low component mean/var/weight: {format_optional_float(gmm.get('low_component_mean'))} / {format_optional_float(gmm.get('low_component_var'))} / {format_optional_float(gmm.get('low_component_weight'))}.",
                f"- GMM high component mean/var/weight: {format_optional_float(gmm.get('high_component_mean'))} / {format_optional_float(gmm.get('high_component_var'))} / {format_optional_float(gmm.get('high_component_weight'))}.",
                f"- The low/high posterior intersection was {format_optional_float(gmm.get('intersection'))}. This is treated only as a component boundary, not a true contact label.",
            ]
        )
    lines.extend(
        [
            f"- Selected comparison row: tau_value={recommended_row['tau_value']:.6f}, slope={recommended_row['slope']:.2f}, top10pct weight share={recommended_row['top10pct_weight_share']:.6f}.",
            "",
            "## Weight summary",
            f"- Weight min/max/mean/std: {weight_summary['min']:.6f} / {weight_summary['max']:.6f} / {weight_summary['mean']:.6f} / {weight_summary['std']:.6f}",
            f"- Weight percentiles P50/P90/P95/P99: {weight_summary['p50']:.6f} / {weight_summary['p90']:.6f} / {weight_summary['p95']:.6f} / {weight_summary['p99']:.6f}",
            "",
            "## Limitations",
            "- Channel order [Fx, Fy, Fz, Mx, My, Mz] is assumed from task documentation because explicit unit metadata was not present in the dataset schema.",
            "- Unsupervised low-value/high-value components are not ground-truth contact labels.",
            "- Quantiles are computed on the training split only, and val/test are intentionally excluded.",
            "",
            "## Warnings",
        ]
    )
    if warnings_list:
        lines.extend([f"- {item}" for item in warnings_list[:50]])
    else:
        lines.append("- No additional warnings.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.simplefilter("ignore", category=FutureWarning)
    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_paths = ensure_dirs(args.output)
    warnings_list: list[str] = []

    channel_order = [
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

    if args.synthetic_test:
        episodes, context = make_synthetic_episodes(
            num_episodes=args.synthetic_episodes,
            episode_length=args.synthetic_episode_length,
            seed=args.seed,
        )
        default_window_length = 16
    else:
        episodes, context = load_project_train_episodes(
            config_path=args.config,
            cache_dir=args.cache_dir,
            seed=args.seed,
        )
        sequence_cfg = context["config"]["sequence"]
        keys = context["config"]["dataset"]["keys"]
        default_window_length = int(
            sequence_cfg["tactile_history_force"]
            if keys["tactile_type"] == "force"
            else sequence_cfg["tactile_history_image"]
        )

    window_length = args.window_length or default_window_length
    if window_length < 3:
        raise ValueError("window_length must be at least 3.")
    if args.window_stride < 1:
        raise ValueError("window_stride must be at least 1.")

    channel_stats, episode_stats, raw_summary = raw_statistics(
        episodes=episodes,
        channel_names=channel_order,
        warnings_list=warnings_list,
    )
    all_values = raw_summary.pop("all_values")
    mean = np.asarray(raw_summary["channel_mean"], dtype=np.float64)
    std = np.asarray(raw_summary["channel_std"], dtype=np.float64)
    median_values = np.asarray(raw_summary["channel_median"], dtype=np.float64)
    mad_values = np.asarray(raw_summary["channel_mad"], dtype=np.float64)

    window_metrics = construct_windows(
        episodes=episodes,
        mean=mean,
        std=std,
        window_length=window_length,
        stride=args.window_stride,
    )
    tau_info = estimate_tau_value(
        value_magnitude=window_metrics["value_magnitude"],
        contact_ratio=window_metrics["contact_ratio"],
        rng=rng,
        gmm_max_samples=args.gmm_max_samples,
    )
    change_magnitude = window_metrics["change_magnitude"]
    second_diff_magnitude = window_metrics["second_diff_magnitude"]
    threshold_candidates = {
        "tau_value": args.tau_value if args.tau_value is not None else float(tau_info["tau_value"]),
        "tau_near_change": args.tau_near_change if args.tau_near_change is not None else float(np.percentile(change_magnitude, 5)),
        "tau_low_change": args.tau_low_change if args.tau_low_change is not None else float(np.percentile(change_magnitude, 25)),
        "tau_high_change": args.tau_high_change if args.tau_high_change is not None else float(np.percentile(change_magnitude, 95)),
        "tau_spike": args.tau_spike if args.tau_spike is not None else float(np.percentile(second_diff_magnitude, 99)),
        "tau_transition_delta": args.tau_transition_delta if args.tau_transition_delta is not None else float(np.percentile(window_metrics["endpoint_delta_magnitude"], 75)),
    }
    stable_reference = change_magnitude <= threshold_candidates["tau_low_change"]
    tau_change_default = float(np.percentile(change_magnitude[stable_reference], 95)) if stable_reference.any() else float(np.percentile(change_magnitude, 50))
    threshold_candidates["tau_change"] = args.tau_change if args.tau_change is not None else tau_change_default
    threshold_candidates["alpha_value"] = args.alpha_value
    threshold_candidates["alpha_change"] = args.alpha_change
    threshold_candidates["slope_value"] = args.slope_value
    threshold_candidates["slope_change"] = args.slope_change

    classes = classify_windows(
        windows_raw=window_metrics["windows_raw"],
        windows_norm=window_metrics["windows_norm"],
        metrics=window_metrics,
        thresholds=threshold_candidates,
        contact_ratio=window_metrics["contact_ratio"],
    )
    window_table = make_window_statistics_table(
        metadata=window_metrics["metadata"],
        metrics=window_metrics,
        classes=classes,
        channel_names=channel_order,
    )
    class_rows = []
    for col in [
        "exact_no_change",
        "approx_no_change",
        "low_change",
        "high_change",
        "maybe_no_contact",
        "stable_contact",
        "contact_start",
        "contact_end",
        "spike_or_outlier",
    ]:
        mask = classes[col].to_numpy(dtype=bool)
        class_rows.append(
            {
                "class_name": col,
                "count": int(mask.sum()),
                "ratio": float(mask.mean()) if len(mask) else float("nan"),
            }
        )
    class_summary = pd.DataFrame(class_rows)

    w_value, w_combined = compute_weights(
        value_magnitude=window_metrics["value_magnitude"],
        change_magnitude=window_metrics["change_magnitude"],
        tau_value=threshold_candidates["tau_value"],
        tau_change=threshold_candidates["tau_change"],
        alpha_value=args.alpha_value,
        alpha_change=args.alpha_change,
        slope_value=args.slope_value,
        slope_change=args.slope_change,
    )
    window_table["w_value"] = w_value
    window_table["w_combined"] = w_combined

    plot_global_distributions(
        all_values=all_values,
        mean=mean,
        std=std,
        median=median_values,
        mad_values=mad_values,
        channel_names=channel_order,
        out_dir=output_paths["distributions"],
    )

    episode_stats_sorted = episode_stats.sort_values("change_rms")
    selected_ids = set(episode_stats_sorted.head(1)["episode_id"].tolist())
    selected_ids.update(episode_stats_sorted.tail(1)["episode_id"].tolist())
    episode_ids = [episode.episode_id for episode in episodes if episode.left.shape[0] > 0]
    if episode_ids:
        random_ids = rng.choice(
            np.asarray(episode_ids),
            size=min(args.max_episode_plots, len(episode_ids)),
            replace=False,
        )
        selected_ids.update(int(item) for item in random_ids.tolist())
    episode_map = {episode.episode_id: episode for episode in episodes}
    for episode_id in sorted(selected_ids):
        plot_episode_family(
            episode=episode_map[episode_id],
            out_dir=output_paths["raw_signals"],
            prefix=f"episode_{episode_id}",
        )

    plot_thresholds(
        value_magnitude=window_metrics["value_magnitude"],
        change_magnitude=window_metrics["change_magnitude"],
        second_diff_magnitude=window_metrics["second_diff_magnitude"],
        tau_info=tau_info,
        thresholds=threshold_candidates,
        out_dir=output_paths["thresholds"],
    )

    for class_name in [
        "exact_no_change",
        "approx_no_change",
        "low_change",
        "high_change",
        "maybe_no_contact",
        "stable_contact",
        "contact_start",
        "contact_end",
        "spike_or_outlier",
    ]:
        plot_window_examples(
            windows_raw=window_metrics["windows_raw"],
            window_table=window_table,
            class_name=class_name,
            out_dir=output_paths["window_examples"],
            max_examples=args.max_window_examples,
            rng=rng,
        )

    weight_summary = plot_weights(
        value_magnitude=window_metrics["value_magnitude"],
        change_magnitude=window_metrics["change_magnitude"],
        weights=w_combined,
        metadata=window_metrics["metadata"],
        thresholds=threshold_candidates,
        args=args,
        out_dir=output_paths["weights"],
    )

    tau_candidates = [
        float(np.percentile(window_metrics["value_magnitude"], q))
        for q in [90, 95, 97.5, 99]
    ]
    threshold_comparison = compare_threshold_settings(
        value_magnitude=window_metrics["value_magnitude"],
        change_magnitude=window_metrics["change_magnitude"],
        classes=classes,
        tau_candidates=tau_candidates,
        tau_change=threshold_candidates["tau_change"],
        alpha_value=args.alpha_value,
        alpha_change=args.alpha_change,
        slopes=[2.0, 5.0, 10.0],
    )

    channel_stats.to_csv(output_paths["tables"] / "channel_statistics.csv", index=False)
    episode_stats.to_csv(output_paths["tables"] / "episode_statistics.csv", index=False)
    window_table.to_csv(output_paths["tables"] / "window_statistics.csv", index=False)
    threshold_comparison.to_csv(output_paths["tables"] / "threshold_comparison.csv", index=False)

    json_payload = {
        "channel_mean": [float(x) for x in mean.tolist()],
        "channel_std": [float(x) for x in std.tolist()],
        "channel_median": [float(x) for x in median_values.tolist()],
        "channel_mad": [float(x) for x in mad_values.tolist()],
        "tau_value": float(threshold_candidates["tau_value"]),
        "tau_change": float(threshold_candidates["tau_change"]),
        "alpha_value": float(args.alpha_value),
        "alpha_change": float(args.alpha_change),
        "slope_value": float(args.slope_value),
        "slope_change": float(args.slope_change),
        "window_length": int(window_length),
        "channel_order": channel_order,
        "window_stride": int(args.window_stride),
        "file_count": int(context["file_count"]),
        "num_episodes": int(raw_summary["num_episodes"]),
        "total_frames": int(raw_summary["total_frames"]),
        "contact_label_key": context.get("contact_key"),
        "channel_order_notes": channel_order_assessment(context),
        "thresholds": {k: float(v) for k, v in threshold_candidates.items() if isinstance(v, (int, float))},
        "tau_estimation": tau_info,
        "weight_summary": weight_summary,
        "warnings": warnings_list,
    }
    (output_paths["root"] / "tactile_stats.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_report(
        output_path=output_paths["root"] / "analysis_report.md",
        context=context,
        raw_summary=raw_summary,
        channel_stats=channel_stats,
        thresholds=threshold_candidates,
        tau_info=tau_info,
        class_summary=class_summary,
        weight_summary=weight_summary,
        threshold_table=threshold_comparison,
        warnings_list=warnings_list,
    )

    example = textwrap.dedent(
        f"""
        Run example:
          /home/yang/miniconda3/envs/lerobot_new/bin/python3 {Path(__file__).name} \\
            --config {args.config} \\
            --output {args.output} \\
            --window-stride {args.window_stride}
        """
    ).strip()
    print(example)
    print(f"Analysis complete. Outputs written to: {output_paths['root']}")


if __name__ == "__main__":
    main()
