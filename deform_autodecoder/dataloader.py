from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets import LeRobotDataset


@dataclass(frozen=True)
class DeformDatasetKeys:
    tactile_image: str | list[str]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML 顶层必须是字典。")

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_history_timestamps(
    history_length: int,
    fps: int,
) -> list[float]:
    if history_length < 1:
        raise ValueError("history_length 必须大于等于 1。")

    return [
        offset / fps
        for offset in range(-(history_length - 1), 1)
    ]


def get_table_columns(table: Any) -> set[str]:
    if hasattr(table, "column_names"):
        return set(table.column_names)

    if hasattr(table, "keys"):
        return set(table.keys())

    return set()


def to_int(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.item())

    if isinstance(value, np.generic):
        return int(value.item())

    return int(value)


def get_episode_bounds(
    dataset: LeRobotDataset,
) -> dict[int, tuple[int, int]]:
    episodes = dataset.meta.episodes
    columns = get_table_columns(episodes)
    required = {"dataset_from_index", "dataset_to_index"}
    missing = required - columns

    if missing:
        raise KeyError(
            "episode 元数据缺少字段："
            f"{sorted(missing)}\n"
            f"当前字段：{sorted(columns)}"
        )

    starts = episodes["dataset_from_index"]
    ends = episodes["dataset_to_index"]
    episode_ids = (
        episodes["episode_index"]
        if "episode_index" in columns
        else range(len(starts))
    )

    bounds: dict[int, tuple[int, int]] = {}
    for episode_id, start, end in zip(
        episode_ids,
        starts,
        ends,
        strict=True,
    ):
        bounds[to_int(episode_id)] = (
            to_int(start),
            to_int(end),
        )

    return bounds


def split_episode_ids(
    episode_ids: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "train/val/test 比例之和必须为 1，"
            f"当前为 {ratio_sum:.6f}"
        )

    episode_ids = list(episode_ids)
    if len(episode_ids) < 3:
        raise ValueError("至少需要 3 个 episode 才能划分 train/val/test。")

    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    total = len(episode_ids)
    num_val = max(1, round(total * val_ratio))
    num_test = max(1, round(total * test_ratio))
    num_train = total - num_val - num_test

    if num_train < 1:
        raise ValueError("episode 数量不足，无法按照当前比例划分。")

    return {
        "train": episode_ids[:num_train],
        "val": episode_ids[num_train:num_train + num_val],
        "test": episode_ids[num_train + num_val:],
    }


def check_dataset_features(
    dataset: LeRobotDataset,
    keys: DeformDatasetKeys,
) -> None:
    available = set(dataset.features.keys())
    tactile_keys = (
        {keys.tactile_image}
        if isinstance(keys.tactile_image, str)
        else set(keys.tactile_image)
    )
    missing = tactile_keys - available

    if missing:
        raise KeyError(
            f"数据集缺少字段：{sorted(missing)}\n"
            f"当前数据集字段：{sorted(available)}"
        )


def _concat_tactile_history(
    sample: dict[str, Any],
    tactile_keys: str | list[str],
) -> Tensor:
    if isinstance(tactile_keys, str):
        tactile_history = sample[tactile_keys].float()
    else:
        tactile_tensors = [
            sample[key].float()
            for key in tactile_keys
        ]
        tactile_history = torch.cat(tactile_tensors, dim=1)

    if tactile_history.ndim != 4:
        raise ValueError(
            "Expected tactile_history with shape [T, C, H, W], "
            f"got {tuple(tactile_history.shape)}."
        )

    return tactile_history


def extract_deformation_image(
    tactile_history: Tensor,
    preprocess_cfg: dict[str, Any],
) -> Tensor:
    history_index = int(preprocess_cfg["history_index"])
    channel_start = int(preprocess_cfg["channel_start"])
    num_channels = int(preprocess_cfg["num_channels"])

    image = tactile_history[
        history_index,
        channel_start:channel_start + num_channels,
    ]

    if image.shape[0] != num_channels:
        raise ValueError(
            "Failed to extract the requested channels from tactile_history, "
            f"got {tuple(image.shape)}, expected first dim {num_channels}."
        )

    image = image.float()

    if bool(preprocess_cfg.get("resize_to_model_input", True)):
        target_size = tuple(
            int(v) for v in preprocess_cfg["target_size"]
        )
        if image.shape[-2:] != target_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    return image


class DeformAutoencoderDataset(Dataset):
    def __init__(
        self,
        base_dataset: LeRobotDataset,
        episode_bounds: dict[int, tuple[int, int]],
        episode_ids: Sequence[int],
        keys: DeformDatasetKeys,
        history_length: int,
        preprocess_cfg: dict[str, Any],
        randomize_indices: bool = False,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.episode_bounds = episode_bounds
        self.episode_ids = list(episode_ids)
        self.keys = keys
        self.history_length = history_length
        self.preprocess_cfg = preprocess_cfg
        self.randomize_indices = randomize_indices
        self.random_seed = random_seed
        self.valid_indices = self._build_valid_indices()

        if not self.valid_indices:
            raise RuntimeError(
                "当前 split 没有合法窗口。请检查 episode 长度和 tactile_history。"
            )

    def _build_valid_indices(self) -> list[int]:
        valid_indices: list[int] = []

        for episode_id in self.episode_ids:
            start, end = self.episode_bounds[episode_id]
            first_center = start + self.history_length - 1
            last_center = end - 1
            if first_center <= last_center:
                valid_indices.extend(
                    range(first_center, last_center + 1)
                )

        if self.randomize_indices:
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(valid_indices)

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        absolute_index = self.valid_indices[index]
        sample = self.base_dataset[absolute_index]
        tactile_history = _concat_tactile_history(
            sample,
            self.keys.tactile_image,
        )

        if tactile_history.shape[0] != self.history_length:
            raise RuntimeError(
                "触觉历史长度错误："
                f"得到 {tuple(tactile_history.shape)}，"
                f"期望第一维为 {self.history_length}"
            )

        image = extract_deformation_image(
            tactile_history,
            self.preprocess_cfg,
        )

        output: dict[str, Tensor] = {
            "image": image,
            "tactile_history": tactile_history,
            "episode_index": torch.as_tensor(
                sample["episode_index"],
                dtype=torch.long,
            ),
            "frame_index": torch.as_tensor(
                sample["frame_index"],
                dtype=torch.long,
            ),
            "absolute_index": torch.tensor(
                absolute_index,
                dtype=torch.long,
            ),
        }
        return output


def build_base_dataset(
    config: dict[str, Any],
) -> LeRobotDataset:
    data_cfg = config["data"]["dataloader"]
    keys = DeformDatasetKeys(
        tactile_image=data_cfg["tactile_image"]
    )

    metadata_dataset = LeRobotDataset(
        repo_id=data_cfg["repo_id"],
        root=data_cfg.get("root"),
        revision=data_cfg.get("revision"),
        video_backend=data_cfg.get("video_backend", "pyav"),
    )

    fps = int(metadata_dataset.fps)
    history_timestamps = make_history_timestamps(
        history_length=int(data_cfg["tactile_history"]),
        fps=fps,
    )

    delta_timestamps: dict[str, list[float]] = {}
    if isinstance(keys.tactile_image, str):
        delta_timestamps[keys.tactile_image] = history_timestamps
    else:
        for tactile_key in keys.tactile_image:
            delta_timestamps[tactile_key] = history_timestamps

    return LeRobotDataset(
        repo_id=data_cfg["repo_id"],
        root=data_cfg.get("root"),
        revision=data_cfg.get("revision"),
        delta_timestamps=delta_timestamps,
        video_backend=data_cfg.get("video_backend", "pyav"),
    )


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[
    dict[str, DataLoader],
    dict[str, DeformAutoencoderDataset],
]:
    data_cfg = config["data"]["dataloader"]
    training_cfg = config["training"]
    preprocess_cfg = config["preprocess"]
    keys = DeformDatasetKeys(
        tactile_image=data_cfg["tactile_image"]
    )

    dataset = build_base_dataset(config)
    check_dataset_features(dataset, keys)
    episode_bounds = get_episode_bounds(dataset)

    episode_splits = split_episode_ids(
        episode_ids=sorted(episode_bounds.keys()),
        train_ratio=float(data_cfg["split"]["train"]),
        val_ratio=float(data_cfg["split"]["val"]),
        test_ratio=float(data_cfg["split"]["test"]),
        seed=int(data_cfg["split"]["seed"]),
    )

    datasets = {
        split_name: DeformAutoencoderDataset(
            base_dataset=dataset,
            episode_bounds=episode_bounds,
            episode_ids=episode_ids,
            keys=keys,
            history_length=int(data_cfg["tactile_history"]),
            preprocess_cfg=preprocess_cfg,
            randomize_indices=bool(
                data_cfg["loader"].get("randomize_indices", True)
            ),
            random_seed=int(data_cfg["split"]["seed"]) + {
                "train": 0,
                "val": 1,
                "test": 2,
            }[split_name],
        )
        for split_name, episode_ids
        in episode_splits.items()
    }

    num_workers = int(training_cfg["num_workers"])
    common_loader_kwargs = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg["loader"].get("pin_memory", True)),
        "persistent_workers": bool(
            data_cfg["loader"].get("persistent_workers", True)
            and num_workers > 0
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(int(data_cfg["split"]["seed"]))

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=bool(data_cfg["loader"].get("shuffle_train", True)),
            drop_last=bool(data_cfg["loader"].get("drop_last_train", False)),
            generator=generator,
            **common_loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=bool(data_cfg["loader"].get("shuffle_val", False)),
            drop_last=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=bool(data_cfg["loader"].get("shuffle_test", False)),
            drop_last=False,
            **common_loader_kwargs,
        ),
    }

    return dataloaders, datasets
