from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from safetensors import safe_open
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import (
    make_policy,
    make_pre_post_processors,
)


@dataclass(frozen=True)
class DatasetKeys:
    tactile_type: str  # "image" or "force"
    tactile_force: str | list[str]
    current_force: str | list[str]  # 新增：当前力数据键
    state: str
    expert_action: str
    tactile_image: str | list[str] | None = None  # 可选：仅当tactile_type="image"时需要
    tactile_force_channel_order: list[str] | None = None

    @property
    def tactile(self) -> str | list[str]:
        """根据 tactile_type 返回对应的触觉键"""
        if self.tactile_type == "image":
            if self.tactile_image is None:
                raise ValueError("tactile_type='image' 时必须提供 tactile_image 键")
            return self.tactile_image
        elif self.tactile_type == "force":
            return self.tactile_force
        else:
            raise ValueError(f"未知的 tactile_type: {self.tactile_type}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML顶层必须是字典。")

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "YAML配置为cuda，但当前环境无法使用CUDA。"
        )

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "YAML配置为mps，但当前环境无法使用MPS。"
        )

    return device


def resolve_pretrained_policy_path(
    pretrained_path: str | Path,
) -> str:
    candidate = Path(pretrained_path).expanduser()

    if not candidate.exists():
        return str(pretrained_path)

    if candidate.is_file():
        raise ValueError(
            "policy.pretrained_path 必须指向目录，"
            f"当前为文件：{candidate}"
        )

    if (candidate / "config.json").is_file():
        return str(candidate)

    nested_candidate = candidate / "pretrained_model"
    if (nested_candidate / "config.json").is_file():
        return str(nested_candidate)

    raise FileNotFoundError(
        "在策略目录中找不到 config.json。"
        f"已检查：{candidate / 'config.json'} 和 "
        f"{nested_candidate / 'config.json'}"
    )


def check_act_checkpoint_compatibility(
    pretrained_path: str | Path,
) -> None:
    model_path = Path(pretrained_path) / "model.safetensors"

    if not model_path.is_file():
        return

    with safe_open(
        str(model_path),
        framework="pt",
        device="cpu",
    ) as handle:
        keys = list(handle.keys())

    has_multi_camera_backbones = any(
        key.startswith("model.backbones.")
        for key in keys
    )
    has_single_backbone = any(
        key.startswith("model.backbone.")
        for key in keys
    )

    if (
        has_multi_camera_backbones
        and not has_single_backbone
    ):
        raise RuntimeError(
            "ACT checkpoint 与当前安装的 lerobot==0.6.0 "
            "实现不兼容。当前 checkpoint 使用多相机独立骨干"
            "(`model.backbones.*`)，而本地 ACT 实现期望共享骨干"
            "(`model.backbone.*`)。\n"
            f"checkpoint: {model_path}\n"
            "这不是日志卡住，而是模型结构版本不一致。"
            "需要使用训练该 checkpoint 时对应的 lerobot/ACT 代码版本，"
            "或重新导出与当前 lerobot 兼容的 checkpoint。"
        )


def to_int(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.item())

    if isinstance(value, np.generic):
        return int(value.item())

    return int(value)


def make_history_timestamps(
    history_length: int,
    fps: int,
) -> list[float]:
    """
    history_length=4时：

        [-3/fps, -2/fps, -1/fps, 0]
    """
    if history_length < 1:
        raise ValueError("history_length必须大于等于1。")

    return [
        offset / fps
        for offset in range(-(history_length - 1), 1)
    ]


def make_future_timestamps(
    horizon: int,
    fps: int,
) -> list[float]:
    """
    horizon=4时：

        [0, 1/fps, 2/fps, 3/fps]
    """
    if horizon < 1:
        raise ValueError("horizon必须大于等于1。")

    return [
        offset / fps
        for offset in range(horizon)
    ]


def get_table_columns(table: Any) -> set[str]:
    if hasattr(table, "column_names"):
        return set(table.column_names)

    if hasattr(table, "keys"):
        return set(table.keys())

    return set()


def get_episode_bounds(
    dataset: LeRobotDataset,
) -> dict[int, tuple[int, int]]:
    """
    返回：

        episode_index -> [start_index, end_index)
    """
    episodes = dataset.meta.episodes
    columns = get_table_columns(episodes)

    required = {
        "dataset_from_index",
        "dataset_to_index",
    }

    missing = required - columns

    if missing:
        raise KeyError(
            "episode元数据缺少字段："
            f"{sorted(missing)}\n"
            f"当前字段：{sorted(columns)}"
        )

    starts = episodes["dataset_from_index"]
    ends = episodes["dataset_to_index"]

    if "episode_index" in columns:
        episode_ids = episodes["episode_index"]
    else:
        episode_ids = range(len(starts))

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
            "train、val、test比例之和必须为1，"
            f"当前为{ratio_sum:.6f}"
        )

    episode_ids = list(episode_ids)

    if len(episode_ids) < 3:
        raise ValueError(
            "至少需要3个episode才能划分train/val/test。"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    total = len(episode_ids)
    num_val = max(1, round(total * val_ratio))
    num_test = max(1, round(total * test_ratio))
    num_train = total - num_val - num_test

    if num_train < 1:
        raise ValueError(
            "episode数量不足，无法按照当前比例划分。"
        )

    return {
        "train": episode_ids[:num_train],
        "val": episode_ids[
            num_train:num_train + num_val
        ],
        "test": episode_ids[
            num_train + num_val:
        ],
    }


class TactileACTDataset(Dataset):
    """
    普通LeRobot Dataset包装器。

    对中心时刻t返回：

        tactile_history:
            F[t-L+1 : t]

        state:
            q[t]

        expert_action:
            A_expert[t : t+K-1]

        ACT使用的当前图像和状态观测。

    ACT动作不在这里预测，而是在ACTAugmentedLoader中预测。
    """

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        episode_bounds: dict[int, tuple[int, int]],
        episode_ids: Sequence[int],
        keys: DatasetKeys,
        act_observation_keys: Sequence[str],
        tactile_history: int,
        action_horizon: int,
        tactile_type: str,  # 新增：触觉类型
    ) -> None:
        super().__init__()

        self.base_dataset = base_dataset
        self.episode_bounds = episode_bounds
        self.episode_ids = list(episode_ids)
        self.keys = keys
        self.act_observation_keys = list(act_observation_keys)
        self.tactile_history = tactile_history
        self.action_horizon = action_horizon
        self.tactile_type = tactile_type

        self.valid_indices = self._build_valid_indices()

        if not self.valid_indices:
            raise RuntimeError(
                "当前split没有合法窗口。请检查episode长度、"
                "tactile_history和action_horizon。"
            )

    def _build_valid_indices(self) -> list[int]:
        valid_indices: list[int] = []

        for episode_id in self.episode_ids:
            start, end = self.episode_bounds[episode_id]

            # t-L+1 >= start
            first_center = start + self.tactile_history - 1

            # t+K-1 <= end-1
            last_center = end - self.action_horizon

            if first_center <= last_center:
                valid_indices.extend(
                    range(first_center, last_center + 1)
                )

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        absolute_index = self.valid_indices[index]
        sample = self.base_dataset[absolute_index]

        # 支持单个或多个触觉输入
        if isinstance(self.keys.tactile, str):
            tactile_history = sample[self.keys.tactile].float()
        else:
            # 多个触觉输入在通道维度拼接
            tactile_tensors = [
                sample[key].float()
                for key in self.keys.tactile
            ]

            # 根据触觉类型决定拼接方式
            if self.tactile_type == "image":
                # 图像: [T, C, H, W]，在C维度拼接
                tactile_history = torch.cat(tactile_tensors, dim=1)
            elif self.tactile_type == "force":
                # 合力: [T, D]，在D维度拼接
                tactile_history = torch.cat(tactile_tensors, dim=1)
            else:
                raise ValueError(f"未知的 tactile_type: {self.tactile_type}")

        # 处理当前力数据（新增）
        if isinstance(self.keys.current_force, str):
            current_force = sample[self.keys.current_force].float()
        else:
            # 多个当前力输入在维度拼接
            current_force_tensors = [
                sample[key].float()
                for key in self.keys.current_force
            ]
            current_force = torch.cat(current_force_tensors, dim=-1)

        # 如果current_force有时间维度，取最后一帧
        if current_force.ndim > 1 and current_force.shape[0] > 1:
            current_force = current_force[-1]

        # 如果是单帧但有时间维度，squeeze掉
        if current_force.ndim > 1 and current_force.shape[0] == 1:
            current_force = current_force.squeeze(0)

        expert_action = sample[self.keys.expert_action].float()

        if tactile_history.shape[0] != self.tactile_history:
            raise RuntimeError(
                "触觉历史时间长度错误："
                f"得到{tuple(tactile_history.shape)}，"
                f"期望第一维为{self.tactile_history}"
            )

        if expert_action.shape[0] != self.action_horizon:
            raise RuntimeError(
                "专家动作时间长度错误："
                f"得到{tuple(expert_action.shape)}，"
                f"期望第一维为{self.action_horizon}"
            )

        output: dict[str, Tensor] = {
            "tactile_history": tactile_history,
            "current_force": current_force,  # 新增
            "expert_action": expert_action,
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

        # 原样返回ACT需要的当前观测。
        # 注意：触觉键如果在这里，需要特殊处理（取最后一帧）
        tactile_keys_set = {self.keys.tactile} if isinstance(self.keys.tactile, str) else set(self.keys.tactile)

        for key in self.act_observation_keys:
            value = sample[key]

            if isinstance(value, Tensor):
                value = value.float()

                # 如果这是触觉键，它有时间维度，需要取最后一帧
                if key in tactile_keys_set:
                    if value.ndim >= 1 and value.shape[0] == self.tactile_history:
                        value = value[-1]

            output[key] = value

        return output


def move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if isinstance(value, Tensor):
        return value.to(
            device=device,
            non_blocking=True,
        )

    if isinstance(value, dict):
        return {
            key: move_to_device(item, device)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            move_to_device(item, device)
            for item in value
        ]

    return value


def extract_action_tensor(output: Any) -> Tensor:
    """
    兼容postprocessor返回Tensor或包含action的字典。
    """
    if isinstance(output, Tensor):
        return output

    if isinstance(output, dict):
        if "action" in output:
            action = output["action"]

            if not isinstance(action, Tensor):
                raise TypeError(
                    "postprocessor输出中的action不是Tensor。"
                )

            return action

    raise TypeError(
        "无法从postprocessor输出中提取动作。"
        f"输出类型为：{type(output).__name__}"
    )


class ACTAugmentedLoader:
    """
    在普通DataLoader之后在线调用冻结ACT。

    每个输出batch会新增：

        act_chunk:
            冻结ACT预测的动作块，[B, K, action_dim]

        delta_action_target:
            expert_action - act_chunk

    ACT推理发生在主进程，而不是DataLoader worker。
    """

    def __init__(
        self,
        dataloader: DataLoader,
        policy: torch.nn.Module,
        preprocessor: Any,
        postprocessor: Any,
        act_observation_keys: Sequence[str],
        device: torch.device,
        action_horizon: int,
        use_postprocessor: bool = True,
    ) -> None:
        self.dataloader = dataloader
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.act_observation_keys = list(act_observation_keys)
        self.device = device
        self.action_horizon = action_horizon
        self.use_postprocessor = use_postprocessor

        if not hasattr(policy, "predict_action_chunk"):
            raise TypeError(
                f"{type(policy).__name__}没有predict_action_chunk()，"
                "不能作为动作块策略使用。"
            )

        self.policy.eval()

        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self.policy.eval()

        for cpu_batch in self.dataloader:
            batch = move_to_device(
                cpu_batch,
                self.device,
            )

            observation = {
                key: batch[key]
                for key in self.act_observation_keys
            }

            # 使用checkpoint自带processor进行归一化、
            # 图像处理、设备处理等。
            processed_observation = self.preprocessor(
                observation
            )

            with torch.inference_mode():
                predicted_chunk = (
                    self.policy.predict_action_chunk(
                        processed_observation
                    )
                )

                if self.use_postprocessor:
                    predicted_chunk = self.postprocessor(
                        predicted_chunk
                    )

                act_chunk = extract_action_tensor(
                    predicted_chunk
                )

            if act_chunk.ndim != 3:
                raise RuntimeError(
                    "ACT输出必须为[B, chunk_size, action_dim]，"
                    f"当前为{tuple(act_chunk.shape)}"
                )

            if act_chunk.shape[1] < self.action_horizon:
                raise RuntimeError(
                    "ACT输出chunk长度小于YAML配置的"
                    f"action_horizon：ACT={act_chunk.shape[1]}，"
                    f"配置={self.action_horizon}"
                )

            act_chunk = act_chunk[
                :, :self.action_horizon
            ]

            expert_action = batch["expert_action"][
                :, :self.action_horizon
            ]

            act_chunk = act_chunk.to(
                device=expert_action.device,
                dtype=expert_action.dtype,
                non_blocking=True,
            )

            if act_chunk.shape != expert_action.shape:
                raise RuntimeError(
                    "ACT动作与专家动作形状不同：\n"
                    f"ACT:    {tuple(act_chunk.shape)}\n"
                    f"expert: {tuple(expert_action.shape)}"
                )

            batch["act_chunk"] = act_chunk
            batch["delta_action_target"] = (
                expert_action - act_chunk
            )

            yield batch


def check_dataset_features(
    dataset: LeRobotDataset,
    keys: DatasetKeys,
    act_observation_keys: Sequence[str],
) -> None:
    available = set(dataset.features.keys())

    # 处理单个或多个触觉键
    if isinstance(keys.tactile, str):
        tactile_keys = {keys.tactile}
    else:
        tactile_keys = set(keys.tactile)

    # 处理单个或多个当前力键
    if isinstance(keys.current_force, str):
        current_force_keys = {keys.current_force}
    else:
        current_force_keys = set(keys.current_force)

    required = {
        keys.state,
        keys.expert_action,
        *act_observation_keys,
        *tactile_keys,
        *current_force_keys,
    }

    missing = required - available

    if missing:
        raise KeyError(
            f"数据集缺少字段：{sorted(missing)}\n"
            f"当前数据集字段：{sorted(available)}"
        )


def build_base_dataset(
    config: dict[str, Any],
) -> LeRobotDataset:
    dataset_cfg = config["dataset"]
    sequence_cfg = config["sequence"]
    video_backend = dataset_cfg.get(
        "video_backend",
        "pyav",
    )

    keys = DatasetKeys(**dataset_cfg["keys"])

    # 第一次加载只用于获取fps。
    metadata_dataset = LeRobotDataset(
        repo_id=dataset_cfg["repo_id"],
        root=dataset_cfg.get("root"),
        revision=dataset_cfg.get("revision"),
        video_backend=video_backend,
    )

    fps = int(metadata_dataset.fps)

    # 根据触觉类型选择对应的历史长度
    if keys.tactile_type == "image":
        tactile_history_length = int(sequence_cfg["tactile_history_image"])
    elif keys.tactile_type == "force":
        tactile_history_length = int(sequence_cfg["tactile_history_force"])
    else:
        raise ValueError(f"未知的 tactile_type: {keys.tactile_type}")

    tactile_history_timestamps = make_history_timestamps(
        history_length=tactile_history_length,
        fps=fps,
    )

    delta_timestamps = {}

    # 为每个触觉键分配相同的历史时间戳
    if isinstance(keys.tactile, str):
        delta_timestamps[keys.tactile] = tactile_history_timestamps
    else:
        for tactile_key in keys.tactile:
            delta_timestamps[tactile_key] = tactile_history_timestamps

    delta_timestamps[keys.expert_action] = make_future_timestamps(
        horizon=int(
            sequence_cfg["action_horizon"]
        ),
        fps=fps,
    )

    # ACT状态和图像不放入delta_timestamps，
    # 因而返回当前时刻观测，而不是额外的时间维。
    return LeRobotDataset(
        repo_id=dataset_cfg["repo_id"],
        root=dataset_cfg.get("root"),
        revision=dataset_cfg.get("revision"),
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )


def build_normal_dataloaders(
    config: dict[str, Any],
    dataset: LeRobotDataset,
) -> tuple[
    dict[str, DataLoader],
    dict[str, TactileACTDataset],
]:
    dataset_cfg = config["dataset"]
    sequence_cfg = config["sequence"]
    split_cfg = config["split"]
    loader_cfg = config["loader"]

    keys = DatasetKeys(**dataset_cfg["keys"])

    act_observation_keys = dataset_cfg[
        "act_observation_keys"
    ]

    check_dataset_features(
        dataset=dataset,
        keys=keys,
        act_observation_keys=act_observation_keys,
    )

    episode_bounds = get_episode_bounds(dataset)

    episode_splits = split_episode_ids(
        episode_ids=sorted(episode_bounds.keys()),
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )

    datasets = {
        split_name: TactileACTDataset(
            base_dataset=dataset,
            episode_bounds=episode_bounds,
            episode_ids=episode_ids,
            keys=keys,
            act_observation_keys=act_observation_keys,
            tactile_history=int(
                sequence_cfg["tactile_history_image"] if keys.tactile_type == "image"
                else sequence_cfg["tactile_history_force"]
            ),
            action_horizon=int(
                sequence_cfg["action_horizon"]
            ),
            tactile_type=keys.tactile_type,
        )
        for split_name, episode_ids
        in episode_splits.items()
    }

    num_workers = int(loader_cfg["num_workers"])

    common_loader_kwargs = {
        "batch_size": int(loader_cfg["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(
            loader_cfg.get("pin_memory", True)
        ),
        "persistent_workers": bool(
            loader_cfg.get("persistent_workers", True)
            and num_workers > 0
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(int(split_cfg["seed"]))

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=bool(
                loader_cfg.get("shuffle_train", True)
            ),
            drop_last=bool(
                loader_cfg.get("drop_last_train", False)
            ),
            generator=generator,
            **common_loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs,
        ),
    }

    return dataloaders, datasets


def load_lerobot_policy(
    config: dict[str, Any],
    dataset: LeRobotDataset,
) -> tuple[
    torch.nn.Module,
    Any,
    Any,
    torch.device,
]:
    policy_cfg_yaml = config["policy"]

    pretrained_path = str(
        policy_cfg_yaml["pretrained_path"]
    )
    resolved_pretrained_path = (
        resolve_pretrained_policy_path(
            pretrained_path
        )
    )
    revision = policy_cfg_yaml.get("revision")
    device = resolve_device(
        policy_cfg_yaml.get("device", "cuda")
    )

    # 从checkpoint读取真正的策略配置，
    # 包括chunk_size、输入特征和归一化方式。
    policy_config = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=resolved_pretrained_path,
        revision=revision,
    )

    expected_type = policy_cfg_yaml.get("type")

    if (
        expected_type is not None
        and policy_config.type != expected_type
    ):
        raise ValueError(
            "YAML配置的policy.type与checkpoint不一致："
            f"YAML={expected_type}，"
            f"checkpoint={policy_config.type}"
        )

    policy_config.device = str(device)
    policy_config.pretrained_path = (
        resolved_pretrained_path
    )
    check_act_checkpoint_compatibility(
        resolved_pretrained_path
    )

    policy = make_policy(
        cfg=policy_config,
        ds_meta=dataset.meta,
    )

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=resolved_pretrained_path,
            pretrained_revision=revision,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
            device=str(device),
        )
    )

    policy.eval()

    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    return (
        policy,
        preprocessor,
        postprocessor,
        device,
    )


def build_augmented_loaders(
    config: dict[str, Any],
    normal_loaders: dict[str, DataLoader],
    policy: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    device: torch.device,
) -> dict[str, ACTAugmentedLoader]:
    action_horizon = int(
        config["sequence"]["action_horizon"]
    )

    act_observation_keys = config["dataset"][
        "act_observation_keys"
    ]

    use_postprocessor = bool(
        config["policy"].get(
            "use_postprocessor",
            True,
        )
    )

    return {
        split_name: ACTAugmentedLoader(
            dataloader=loader,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            act_observation_keys=act_observation_keys,
            device=device,
            action_horizon=action_horizon,
            use_postprocessor=use_postprocessor,
        )
        for split_name, loader
        in normal_loaders.items()
    }


def print_tensor(
    name: str,
    tensor: Tensor,
) -> None:
    finite = True

    if tensor.is_floating_point():
        finite = bool(
            torch.isfinite(tensor).all().item()
        )

    print(
        f"  {name:<24}"
        f"shape={str(tuple(tensor.shape)):<25}"
        f"dtype={str(tensor.dtype):<15}"
        f"device={str(tensor.device):<10}"
        f"finite={finite}"
    )


def test_augmented_loader(
    loader: ACTAugmentedLoader,
    num_batches: int,
) -> None:
    print("\n开始测试带ACT预测的DataLoader：")

    for batch_index, batch in enumerate(loader):
        print(f"\nBatch {batch_index}")

        for key, value in batch.items():
            if isinstance(value, Tensor):
                print_tensor(key, value)

        reconstructed = (
            batch["act_chunk"]
            + batch["delta_action_target"]
        )

        max_error = (
            reconstructed
            - batch["expert_action"]
        ).abs().max().item()

        print(
            "  residual_check          "
            f"max_error={max_error:.8f}"
        )

        if max_error > 1e-5:
            raise RuntimeError(
                "残差检查失败："
                "act_chunk + delta_action_target "
                "不等于expert_action。"
            )

        if batch_index + 1 >= num_batches:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LeRobot触觉数据加载并在线运行ACT预测"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="tactile_dataloader.yaml",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    set_seed(int(config["split"]["seed"]))

    print("加载LeRobot数据集……")
    dataset = build_base_dataset(config)

    print("构建普通DataLoader……")
    normal_loaders, datasets = (
        build_normal_dataloaders(
            config=config,
            dataset=dataset,
        )
    )

    print("加载冻结的LeRobot策略……")
    (
        policy,
        preprocessor,
        postprocessor,
        device,
    ) = load_lerobot_policy(
        config=config,
        dataset=dataset,
    )

    print("构建带ACT预测的DataLoader……")
    augmented_loaders = build_augmented_loaders(
        config=config,
        normal_loaders=normal_loaders,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
    )

    print("\n基础信息：")
    print(f"  dataset repo:  {dataset.repo_id}")
    print(f"  fps:           {dataset.fps}")
    print(f"  episodes:      {dataset.num_episodes}")
    print(f"  policy class:  {type(policy).__name__}")
    print(f"  policy type:   {policy.config.type}")
    print(f"  chunk size:    {policy.config.chunk_size}")
    print(f"  device:        {device}")

    print("\n数据划分：")

    for split_name in ("train", "val", "test"):
        split_dataset = datasets[split_name]

        print(
            f"  {split_name:<5}"
            f" episodes={len(split_dataset.episode_ids):<5}"
            f" windows={len(split_dataset)}"
        )

    test_augmented_loader(
        loader=augmented_loaders["train"],
        num_batches=int(
            config.get("test", {}).get(
                "num_batches",
                1,
            )
        ),
    )

    print("\n带ACT预测的DataLoader测试通过。")


if __name__ == "__main__":
    main()
