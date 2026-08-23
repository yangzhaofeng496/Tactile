"""
独立缓存DataLoader：完全脱离LeRobotDataset，直接从离线预处理缓存读取所有训练输入。

缓存由 dataloader/preprocess_act_cache.py --full-cache 生成，每个absolute_index条目包含:
    - act_chunk: [K, action_dim]
    - act_visual: [D]（可选）
    - tactile_history: [T, D]
    - current_force: [D]
    - state: [D]
    - expert_action: [K, D]

训练时不再解码任何视频，CPU负担极低。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from dataloader.dataloader import (
    DatasetKeys,
    get_episode_bounds,
    load_yaml,
    set_seed,
    split_episode_ids,
)


class CachedResidualDataset(Dataset):
    """从缓存读取窗口数据，与TactileACTDataset输出字段保持一致。"""

    def __init__(
        self,
        cache: dict[int, dict],
        episode_bounds: dict[int, tuple[int, int]],
        episode_ids: Sequence[int],
        keys: DatasetKeys,
        tactile_history: int,
        action_horizon: int | None = None,
        use_act_visual: bool = False,
        window_stride: int = 1,
    ) -> None:
        super().__init__()

        self.cache = cache
        self.episode_bounds = episode_bounds
        self.episode_ids = list(episode_ids)
        self.keys = keys
        self.tactile_history = tactile_history
        self.action_horizon = action_horizon if action_horizon is not None else 0
        self.use_act_visual = use_act_visual
        self.window_stride = max(1, int(window_stride))

        self.valid_episode_ids: list[int] = []
        self.valid_indices = self._build_valid_indices()
        self.episode_positions = self._build_episode_positions()

        if not self.valid_indices:
            raise RuntimeError(
                "当前split没有合法窗口。请检查episode长度、"
                "tactile_history和action_horizon。"
            )

    def _build_valid_indices(self) -> list[int]:
        valid_indices: list[int] = []

        for episode_id in self.episode_ids:
            start, end = self.episode_bounds[episode_id]
            first_center = start + self.tactile_history - 1
            if self.action_horizon > 0:
                last_center = end - self.action_horizon
            else:
                last_center = end - 1
            if first_center <= last_center:
                indices = list(range(first_center, last_center + 1, self.window_stride))
                valid_indices.extend(indices)
                self.valid_episode_ids.extend([episode_id] * len(indices))

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _build_episode_positions(self) -> list[list[int]]:
        """Return dataset positions grouped by episode for stratified sampling."""
        groups: list[list[int]] = []
        offset = 0
        for episode_id in self.episode_ids:
            start, end = self.episode_bounds[episode_id]
            first_center = start + self.tactile_history - 1
            last_center = end - self.action_horizon if self.action_horizon > 0 else end - 1
            count = 0
            if first_center <= last_center:
                count = len(range(first_center, last_center + 1, self.window_stride))
            if count:
                groups.append(list(range(offset, offset + count)))
                offset += count
        return groups

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        absolute_index = self.valid_indices[index]
        entry = self.cache[absolute_index]

        output: dict[str, torch.Tensor] = {
            "absolute_index": torch.tensor(
                absolute_index,
                dtype=torch.long,
            ),
            "episode_index": torch.tensor(
                self.valid_episode_ids[index],
                dtype=torch.long,
            ),
        }

        # 触觉历史分支关闭时，缓存不需要保存该大字段。
        if "tactile_history" in entry:
            output["tactile_history"] = entry["tactile_history"].float()

        if self.use_act_visual:
            output["act_visual_tokens"] = entry["act_visual"].float()

        output["act_chunk"] = entry["act_chunk"].float()
        output["current_force"] = entry["current_force"].float()
        output[self.keys.state] = entry["state"].float()
        output["expert_action"] = entry["expert_action"].float()

        return output


class EpisodeRandomWindowSampler(Sampler[int]):
    """Sample a fresh, episode-balanced subset of train windows each epoch."""

    def __init__(
        self,
        dataset: CachedResidualDataset,
        fraction: float,
        seed: int,
        balance_current_force: bool = False,
        hard_replay_fraction: float = 0.0,
        reference_dataset: CachedResidualDataset | None = None,
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("train_window_fraction must be in (0, 1].")
        self.groups = dataset.episode_positions
        self.fraction = float(fraction)
        self.balance_current_force = bool(balance_current_force)
        self.reference_dataset = reference_dataset
        self.match_reference_distribution = reference_dataset is not None
        self.sample_weights: dict[int, float] = {}
        if reference_dataset is not None:
            self._build_reference_weights(dataset, reference_dataset)
        if not 0.0 <= float(hard_replay_fraction) <= 1.0:
            raise ValueError("hard_replay_fraction must be in [0, 1].")
        self.hard_replay_fraction = float(hard_replay_fraction)
        self.force_norms = {}
        self.action_norms = {}
        self.hard_positions: list[int] = []
        if self.balance_current_force:
            self.force_norms = {
                position: float(
                    dataset.cache[dataset.valid_indices[position]][
                        "current_force"
                    ].float().norm().item()
                )
                for group in self.groups
                for position in group
            }
        if self.hard_replay_fraction > 0.0:
            force_norms = []
            action_norms = []
            for group in self.groups:
                for position in group:
                    entry = dataset.cache[dataset.valid_indices[position]]
                    force_norms.append(
                        float(entry["current_force"].float().norm().item())
                    )
                    action_norms.append(
                        float(entry["expert_action"].float().norm().item())
                    )
            force_q80 = float(torch.tensor(force_norms).quantile(0.80))
            action_q50 = float(torch.tensor(action_norms).quantile(0.50))
            self.hard_positions = [
                position
                for group in self.groups
                for position in group
                if force_norms[position] >= force_q80
                and action_norms[position] <= action_q50
            ]
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self.base_sample_count = sum(
            max(1, int(len(group) * self.fraction)) for group in self.groups
        )
        self.sample_count = self.base_sample_count + int(
            self.base_sample_count * self.hard_replay_fraction
        )

    def _build_reference_weights(self, dataset, reference_dataset) -> None:
        """Match coarse force/action norm proportions to validation only."""
        def features(ds):
            rows = []
            for absolute_index in ds.valid_indices:
                entry = ds.cache[absolute_index]
                rows.append([
                    float(entry["current_force"].float().norm().item()),
                    float(entry["expert_action"].float().norm().item()),
                ])
            return torch.tensor(rows, dtype=torch.float32)

        train_features = features(dataset)
        reference_features = features(reference_dataset)
        combined = torch.cat([train_features, reference_features], dim=0)
        edges = [
            torch.quantile(combined[:, column], torch.tensor([0.25, 0.5, 0.75]))
            for column in range(2)
        ]

        def bin_ids(values):
            force_bin = torch.bucketize(values[:, 0], edges[0])
            action_bin = torch.bucketize(values[:, 1], edges[1])
            return force_bin * 4 + action_bin

        train_bins = bin_ids(train_features)
        reference_bins = bin_ids(reference_features)
        train_counts = torch.bincount(train_bins, minlength=16).float()
        reference_counts = torch.bincount(reference_bins, minlength=16).float()
        train_prob = train_counts / train_counts.sum().clamp_min(1.0)
        reference_prob = reference_counts / reference_counts.sum().clamp_min(1.0)
        bin_weights = reference_prob / train_prob.clamp_min(1e-6)
        bin_weights = bin_weights / bin_weights[train_bins].mean().clamp_min(1e-6)
        self.sample_weights = {
            position: float(bin_weights[train_bins[position]].item())
            for position in range(len(dataset.valid_indices))
        }

    def __iter__(self):
        selected: list[int] = []
        for group in self.groups:
            count = max(1, int(len(group) * self.fraction))
            if self.balance_current_force and len(group) > 1:
                ranked = sorted(group, key=lambda p: self.force_norms[p])
                half = count // 2
                low_pool = ranked[: max(1, len(ranked) // 2)]
                high_pool = ranked[len(ranked) // 2 :]
                low_order = torch.randperm(
                    len(low_pool), generator=self.generator
                ).tolist()
                high_order = torch.randperm(
                    len(high_pool), generator=self.generator
                ).tolist()
                chosen = [low_pool[i] for i in low_order[:half]]
                chosen.extend(
                    high_pool[i]
                    for i in high_order[: count - half]
                )
                selected.extend(chosen)
            elif self.match_reference_distribution:
                weights = torch.tensor(
                    [self.sample_weights[position] for position in group],
                    dtype=torch.float32,
                )
                chosen = torch.multinomial(
                    weights,
                    num_samples=count,
                    replacement=False,
                    generator=self.generator,
                ).tolist()
                selected.extend(group[index] for index in chosen)
            else:
                order = torch.randperm(len(group), generator=self.generator).tolist()
                selected.extend(group[index] for index in order[:count])
        if selected:
            order = torch.randperm(len(selected), generator=self.generator).tolist()
            selected = [selected[index] for index in order]
        hard_count = min(
            len(self.hard_positions),
            int(self.base_sample_count * self.hard_replay_fraction),
        )
        if hard_count:
            order = torch.randperm(
                len(self.hard_positions), generator=self.generator
            ).tolist()
            selected.extend(self.hard_positions[index] for index in order[:hard_count])
        return iter(selected)

    def __len__(self) -> int:
        return self.sample_count


class AdjacentPairBatchSampler(Sampler[list[int]]):
    """Yield batches whose entries are paired adjacent windows from one episode."""

    def __init__(
        self,
        dataset: CachedResidualDataset,
        fraction: float,
        seed: int,
        batch_size: int,
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("train_window_fraction must be in (0, 1].")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be an even number >= 2.")
        self.groups = dataset.episode_positions
        self.fraction = float(fraction)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self.pair_count = sum(
            min(
                max(0, len(group) - 1),
                max(1, int(len(group) * self.fraction / 2)),
            )
            for group in self.groups
            if len(group) >= 2
        )

    def __iter__(self) -> Iterator[list[int]]:
        pairs: list[tuple[int, int]] = []
        for group in self.groups:
            if len(group) < 2:
                continue
            eligible = group[:-1]
            count = min(
                len(eligible),
                max(1, int(len(group) * self.fraction / 2)),
            )
            order = torch.randperm(
                len(eligible),
                generator=self.generator,
            ).tolist()
            pairs.extend(
                (eligible[index], eligible[index] + 1)
                for index in order[:count]
            )

        if pairs:
            order = torch.randperm(
                len(pairs),
                generator=self.generator,
            ).tolist()
            pairs = [pairs[index] for index in order]

        flattened = [position for pair in pairs for position in pair]
        for start in range(0, len(flattened), self.batch_size):
            batch = flattened[start:start + self.batch_size]
            if batch:
                yield batch

    def __len__(self) -> int:
        if self.pair_count == 0:
            return 0
        samples = self.pair_count * 2
        return (samples + self.batch_size - 1) // self.batch_size


def build_cached_loaders(
    config: dict[str, Any],
    cache_path: str | Path,
) -> dict[str, DataLoader]:
    """构建完全基于缓存的DataLoader，不加载LeRobotDataset。"""
    dataset_cfg = config["dataset"]
    sequence_cfg = config["sequence"]
    split_cfg = config["split"]
    loader_cfg = config["loader"]

    keys = DatasetKeys(**dataset_cfg["keys"])

    use_act_visual = bool(
        config["policy"].get("use_act_visual", False)
    )

    # 读取episode边界（只读meta parquet，不解码视频）
    from dataloader.dataloader import (
        resolve_dataset_paths,
    )

    repo_id = dataset_cfg["repo_id"]
    root = dataset_cfg.get("root")
    resolved_repo_id, resolved_root = resolve_dataset_paths(
        repo_id,
        root,
    )

    # 复用LeRobotDataset仅读取元数据（不触达视频帧）
    from lerobot.datasets import LeRobotDataset

    video_backend = dataset_cfg.get("video_backend", "pyav")
    meta_dataset = LeRobotDataset(
        repo_id=resolved_repo_id,
        root=resolved_root,
        revision=dataset_cfg.get("revision"),
        video_backend=video_backend,
    )

    episode_bounds = get_episode_bounds(meta_dataset)

    episode_splits = split_episode_ids(
        episode_ids=sorted(episode_bounds.keys()),
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )

    if keys.tactile_type == "image":
        tactile_history = int(sequence_cfg["tactile_history_image"])
    else:
        tactile_history = int(sequence_cfg["tactile_history_force"])

    action_horizon = int(sequence_cfg["action_horizon"]) if "action_horizon" in sequence_cfg else None
    window_stride = max(1, int(loader_cfg.get("window_stride", 1)))

    # 加载缓存并在构建 DataLoader 前检查是否为 full-cache。
    print(f"加载ACT离线缓存: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")
    required_fields = {
        "act_chunk",
        "tactile_history",
        "current_force",
        "state",
        "expert_action",
    }
    if use_act_visual:
        required_fields.add("act_visual")

    if not isinstance(cache, dict) or not cache:
        raise ValueError(
            f"ACT缓存必须是非空字典：{cache_path}"
        )

    sample_key = next(iter(cache))
    sample_entry = cache[sample_key]
    if not isinstance(sample_entry, dict):
        raise ValueError(
            f"ACT缓存条目格式错误：index={sample_key!r}，"
            "期望为字典。"
        )

    missing_fields = required_fields - set(sample_entry)
    if missing_fields:
        raise ValueError(
            "当前缓存不是训练所需的 full-cache，缺少字段："
            f"{sorted(missing_fields)}。\n"
            f"缓存文件：{cache_path}\n"
            "请重新生成，并添加 --full-cache：\n"
            "python -m dataloader.preprocess_act_cache "
            "--config dataloader/tactile_dataloader.yaml "
            f"--output {cache_path} --full-cache"
        )

    datasets = {
        split_name: CachedResidualDataset(
            cache=cache,
            episode_bounds=episode_bounds,
            episode_ids=episode_ids,
            keys=keys,
            tactile_history=tactile_history,
            action_horizon=action_horizon,
            use_act_visual=use_act_visual,
            window_stride=window_stride,
        )
        for split_name, episode_ids
        in episode_splits.items()
        if len(episode_ids) > 0
    }

    num_workers = int(loader_cfg["num_workers"])

    common_loader_kwargs = {
        "batch_size": int(loader_cfg["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", True)),
        "persistent_workers": bool(
            loader_cfg.get("persistent_workers", True)
            and num_workers > 0
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(int(split_cfg["seed"]))

    dataloaders = {}

    if "train" in datasets:
        train_fraction = float(loader_cfg.get("train_window_fraction", 1.0))
        if bool(loader_cfg.get("pair_adjacent_windows", False)):
            pair_sampler = AdjacentPairBatchSampler(
                datasets["train"],
                fraction=train_fraction,
                seed=int(split_cfg["seed"]),
                batch_size=int(loader_cfg["batch_size"]),
            )
            train_loader_kwargs = {
                key: value
                for key, value in common_loader_kwargs.items()
                if key != "batch_size"
            }
            dataloaders["train"] = DataLoader(
                datasets["train"],
                batch_sampler=pair_sampler,
                **train_loader_kwargs,
            )
        else:
            train_sampler = EpisodeRandomWindowSampler(
                datasets["train"],
                fraction=train_fraction,
                seed=int(split_cfg["seed"]),
                balance_current_force=bool(
                    loader_cfg.get("balance_current_force", False)
                ),
                hard_replay_fraction=float(
                    loader_cfg.get("hard_replay_fraction", 0.0)
                ),
                reference_dataset=(
                    datasets.get("val")
                    if bool(loader_cfg.get("match_val_distribution", False))
                    else None
                ),
            )
            dataloaders["train"] = DataLoader(
                datasets["train"],
                sampler=train_sampler,
                drop_last=bool(loader_cfg.get("drop_last_train", False)),
                **common_loader_kwargs,
            )

    if "val" in datasets:
        dataloaders["val"] = DataLoader(
            datasets["val"],
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        )

    if "test" in datasets:
        dataloaders["test"] = DataLoader(
            datasets["test"],
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        )

    return dataloaders
