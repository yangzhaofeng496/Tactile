"""
离线预处理ACT推理结果，供残差网络训练直接读取。

把每个合法窗口的:
    - act_chunk: 冻结ACT预测的动作块 [K, action_dim]（截断到action_horizon）
    - act_visual: mean-pooled视觉特征 [D]（来自ACT transformer encoder的视觉token）
按 absolute_index 缓存到磁盘。之后训练时不再需要实时跑ACT。

用法:
    python dataloader/preprocess_act_cache.py --config dataloader/tactile_dataloader.yaml \
        --output outputs/act_cache/act_cache.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloader.dataloader import (
    build_base_dataset,
    extract_action_tensor,
    get_episode_bounds,
    load_lerobot_policy,
    load_yaml,
    move_to_device,
    set_seed,
)


class IndexedWindowDataset(Dataset):
    """遍历所有合法窗口，返回 (absolute_index, 原始sample)。"""

    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        absolute_index = self.indices[index]
        sample = self.base_dataset[absolute_index]
        return absolute_index, sample


def compute_valid_indices(
    config,
    dataset,
):
    """与TactileACTDataset._build_valid_indices一致，但覆盖全部episode。"""
    keys = config["dataset"]["keys"]
    sequence_cfg = config["sequence"]
    tactile_type = keys["tactile_type"]

    if tactile_type == "image":
        tactile_history = int(sequence_cfg["tactile_history_image"])
    else:
        tactile_history = int(sequence_cfg["tactile_history_force"])

    action_horizon = int(sequence_cfg.get("action_horizon", 0))

    episode_bounds = get_episode_bounds(dataset)

    valid_indices = []
    for episode_id in sorted(episode_bounds.keys()):
        start, end = episode_bounds[episode_id]
        first_center = start + tactile_history - 1
        if action_horizon > 0:
            last_center = end - action_horizon
        else:
            last_center = end - 1
        if first_center <= last_center:
            valid_indices.extend(range(first_center, last_center + 1))

    return valid_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="离线预处理ACT推理结果，供残差网络训练读取。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dataloader/tactile_dataloader.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/act_cache/act_cache.pt"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前N个窗口（0=全部）。用于快速测试。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)

    set_seed(int(config["split"]["seed"]))

    act_observation_keys = config["dataset"].get("act_observation_keys", [])
    if not act_observation_keys:
        raise ValueError("act_observation_keys不能为空，否则无法运行ACT。")

    use_postprocessor = bool(
        config["policy"].get("use_postprocessor", True)
    )
    use_act_visual = bool(
        config["policy"].get("use_act_visual", False)
    )
    action_horizon = int(config["sequence"]["action_horizon"])

    print("加载LeRobot数据集……")
    dataset = build_base_dataset(config)

    print("计算合法窗口……")
    valid_indices = compute_valid_indices(config, dataset)
    print(f"合法窗口数: {len(valid_indices)}")
    if args.limit and args.limit > 0:
        valid_indices = valid_indices[:args.limit]
        print(f"（限制模式）只处理前{args.limit}个窗口")

    print("加载冻结ACT策略……")
    policy, preprocessor, postprocessor, device = load_lerobot_policy(
        config,
        dataset,
    )

    # 视觉token hook
    _act_encoder_out = {"value": None}
    if use_act_visual:
        encoder = getattr(policy, "model", None).encoder
        if encoder is None:
            raise TypeError("use_act_visual=True但ACT policy没有transformer encoder。")
        n_pre_tokens = 1
        if getattr(policy.config, "robot_state_feature", None):
            n_pre_tokens += 1
        if getattr(policy.config, "env_state_feature", None):
            n_pre_tokens += 1

        def _hook(module, args, output):
            _act_encoder_out["value"] = output

        handle = encoder.register_forward_hook(_hook)

    window_dataset = IndexedWindowDataset(dataset, valid_indices)

    def collate_fn(batch):
        indices = [item[0] for item in batch]
        samples = [item[1] for item in batch]
        # 只堆叠ACT观测所需字段
        collated = {}
        for key in act_observation_keys:
            tensors = [s[key] for s in samples]
            collated[key] = torch.stack(tensors, dim=0)
        return indices, collated

    loader = DataLoader(
        window_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    cache = {}

    try:
        for indices, cpu_batch in tqdm(loader, desc="ACT预处理"):
            batch = move_to_device(cpu_batch, device)

            observation = {
                key: batch[key]
                for key in act_observation_keys
            }
            processed_observation = preprocessor(observation)

            with torch.inference_mode():
                predicted_chunk = policy.predict_action_chunk(
                    processed_observation
                )
                if use_postprocessor:
                    predicted_chunk = postprocessor(predicted_chunk)
                act_chunk = extract_action_tensor(predicted_chunk)

            if act_chunk.ndim != 3:
                raise RuntimeError(
                    "ACT输出必须为[B, chunk_size, action_dim]，"
                    f"当前为{tuple(act_chunk.shape)}"
                )
            if act_chunk.shape[1] < action_horizon:
                raise RuntimeError(
                    "ACT输出chunk长度小于action_horizon："
                    f"ACT={act_chunk.shape[1]}，配置={action_horizon}"
                )
            act_chunk = act_chunk[:, :action_horizon].float().cpu()

            visual = None
            if use_act_visual:
                if _act_encoder_out["value"] is None:
                    raise RuntimeError("没有捕获到ACT encoder输出。")
                encoder_out = _act_encoder_out["value"]
                _act_encoder_out["value"] = None
                visual_tokens = encoder_out[n_pre_tokens:, :, :].transpose(0, 1)
                if visual_tokens.shape[1] == 0:
                    raise RuntimeError("ACT encoder输出中没有视觉token。")
                visual = visual_tokens.mean(dim=1).float().cpu()  # [B, D]

            for i, idx in enumerate(indices):
                entry = {"act_chunk": act_chunk[i]}
                if visual is not None:
                    entry["act_visual"] = visual[i]
                cache[int(idx)] = entry
    finally:
        if use_act_visual:
            handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(args.output))
    print(f"缓存已保存: {args.output}  (共{len(cache)}个窗口)")

    if use_act_visual:
        print(f"  含视觉特征: 每个条目含 act_visual [{visual.shape[-1]}维]")
    else:
        print("  不含视觉特征（use_act_visual=false）")


if __name__ == "__main__":
    main()
