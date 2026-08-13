#!/usr/bin/env python
"""
合并多个 LeRobot 数据集为一个新数据集。

要求所有源数据集具有相同的 fps、robot_type 与 features
（例如都包含 observation.tactile.left_force / right_force）。

用法:
    python dataloader/merge_datasets.py \
        --datasets "/path/to/ds_a" "/path/to/ds_b" \
        --output_repo_id my_merged \
        --output_dir /home/yang/TactileEncoder/dataset/so101/my_merged
"""

import argparse
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset, merge_datasets, remove_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并多个 LeRobot 数据集为一个新数据集",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="源数据集路径列表（至少两个）。",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        required=True,
        help="合并后数据集的 repo_id。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="合并后数据集保存目录（含 meta/ data/ videos/）。",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec"],
        help="加载视频使用的后端。",
    )
    parser.add_argument(
        "--no-concatenate-videos",
        action="store_true",
        help="不拼接视频，保留每个源文件一个 mp4。",
    )
    parser.add_argument(
        "--no-concatenate-data",
        action="store_true",
        help="不拼接 parquet，保留每个源文件一个 parquet。",
    )
    parser.add_argument(
        "--allow-feature-union",
        action="store_true",
        help="允许不同 features 的数据集合并，自动丢弃非公共特征。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.datasets) < 2:
        raise ValueError("--datasets 至少需要两个路径。")

    output_dir = Path(args.output_dir)

    if output_dir.exists():
        raise FileExistsError(
            f"输出目录已存在，请更换 --output_dir 或先删除：{output_dir}"
        )

    datasets = [
        LeRobotDataset(
            repo_id=Path(path).name,
            root=path,
            video_backend=args.video_backend,
        )
        for path in args.datasets
    ]

    for path, ds in zip(args.datasets, datasets, strict=True):
        print(f"源数据集: {path}")
        print(f"  episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.fps}")
        force_keys = [
            k for k in ds.features
            if "tactile" in k and "force" in k
        ]
        print(f"  触觉力特征: {force_keys}")

    datasets = _align_features(datasets, args)

    merged = merge_datasets(
        datasets,
        output_repo_id=args.output_repo_id,
        output_dir=output_dir,
        concatenate_videos=not args.no_concatenate_videos,
        concatenate_data=not args.no_concatenate_data,
    )

    for ds in datasets:
        if ds.root and "_tmp_align" in str(ds.root):
            shutil.rmtree(ds.root, ignore_errors=True)

    print(f"合并完成: {output_dir}")
    print(f"  episodes={merged.num_episodes} frames={merged.num_frames}")


def _align_features(
    datasets: list[LeRobotDataset],
    args: argparse.Namespace,
) -> list[LeRobotDataset]:
    """将各数据集的 features 对齐到公共子集，返回合并可用的数据集列表。

    若所有数据集 features 一致，直接返回原数据集。
    """
    feature_sets = [set(ds.features) for ds in datasets]
    common = set.intersection(*feature_sets)

    if common == feature_sets[0] and all(s == feature_sets[0] for s in feature_sets):
        return datasets

    if not args.allow_feature_union:
        raise ValueError(
            "数据集 features 不一致，无法直接合并。\n"
            f"公共特征: {sorted(common)}\n"
            f"各数据集特征差集: "
            f"{[sorted(s - common) for s in feature_sets]}\n\n"
            "请用 --allow-feature-union 自动丢弃非公共特征后合并。"
        )

    print("\n[特征对齐] 数据集 features 不一致，自动丢弃非公共特征：")
    for ds, fs in zip(datasets, feature_sets, strict=True):
        extra = sorted(fs - common)
        if extra:
            print(f"  {ds.repo_id}: 丢弃 {extra}")

    aligned: list[LeRobotDataset] = []
    tmp_dirs: list[Path] = []

    for ds in datasets:
        extra = sorted(set(ds.features) - common)
        if not extra:
            aligned.append(ds)
            continue

        tmp_dir = Path(args.output_dir).parent / f"_tmp_align_{ds.repo_id}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dirs.append(tmp_dir)

        print(f"[特征对齐] 正在处理 {ds.repo_id} ...")
        aligned_ds = remove_feature(
            ds,
            feature_names=extra,
            output_dir=tmp_dir,
            repo_id=f"{ds.repo_id}_aligned",
        )
        aligned.append(aligned_ds)

    return aligned


if __name__ == "__main__":
    main()
