"""Analyze highest per-timestep Base Policy action MSE samples.

Example:

    python analyze_base_policy_high_loss.py \
        --config dataloader/tactile_dataloader.yaml \
        --output-dir high_loss_analysis \
        --top-k 20 \
        --window 30 \
        --num-steps 1000 \
        --seed 42

Behavior:

1. Build valid samples from the requested split.
2. If --num-steps is specified, RANDOMLY sample N valid observations
   from the entire split.
3. Run Base Policy inference.
4. Compute per-timestep action MSE:

       MSE_t = mean_d((prediction[t, d] - GT[t, d])^2)

5. Find the global Top-K highest-MSE timesteps.
6. For each Top-K sample, save:
       action_dim_0.png
       action_dim_1.png
       action_dim_2.png
       action_dim_3.png
       action_dim_4.png
       action_dim_5.png
       mse.png

The reported `timestep` is relative to the Base Policy action chunk.
timestep=0 corresponds to the action associated with the current observation.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    split_episode_ids,
)


# =============================================================================
# Data structure
# =============================================================================


@dataclass
class HighLossSample:
    """Information for one high-loss action timestep."""

    mse: float
    sample_index: int
    timestep: int

    # [K, D]
    ground_truth: torch.Tensor

    # [K, D]
    prediction: torch.Tensor

    # [K]
    timestep_mse: torch.Tensor


# =============================================================================
# Indexed dataset
# =============================================================================


class IndexedDataset(Dataset):
    """Expose absolute dataset indices while reusing the original dataset."""

    def __init__(
        self,
        dataset: Any,
        indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[int, dict[str, Any]]:
        absolute_index = self.indices[index]

        return (
            absolute_index,
            self.dataset[absolute_index],
        )


def collate_indexed_batch(
    batch: list[tuple[int, dict[str, Any]]],
) -> tuple[list[int], dict[str, Any]]:
    """Collate samples while preserving absolute dataset indices."""

    indices = [
        item[0]
        for item in batch
    ]

    samples = [
        item[1]
        for item in batch
    ]

    keys = samples[0].keys()

    collated = {
        key: torch.utils.data.default_collate(
            [
                sample[key]
                for sample in samples
            ]
        )
        for key in keys
    }

    return indices, collated


# =============================================================================
# Build valid dataset indices
# =============================================================================


def build_valid_indices(
    config: dict[str, Any],
    dataset: Any,
    split_name: str,
) -> list[int]:
    """Build valid current-observation indices for the requested split."""

    episode_bounds = get_episode_bounds(
        dataset
    )

    split_cfg = config["split"]

    episode_splits = split_episode_ids(
        episode_ids=sorted(episode_bounds),
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )

    if split_name == "all":
        episode_ids = sorted(
            episode_bounds
        )
    else:
        episode_ids = episode_splits[
            split_name
        ]

    keys_cfg = config["dataset"]["keys"]
    sequence_cfg = config["sequence"]

    tactile_history = int(
        sequence_cfg[
            "tactile_history_image"
            if keys_cfg["tactile_type"] == "image"
            else "tactile_history_force"
        ]
    )

    action_horizon = int(
        sequence_cfg["action_horizon"]
    )

    indices: list[int] = []

    for episode_id in episode_ids:

        start, end = episode_bounds[
            episode_id
        ]

        # Enough tactile history must exist before current observation.
        first_center = (
            start
            + tactile_history
            - 1
        )

        # Enough future action timesteps must exist.
        last_center = (
            end
            - action_horizon
        )

        if first_center <= last_center:

            indices.extend(
                range(
                    first_center,
                    last_center + 1,
                )
            )

    return indices


# =============================================================================
# Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze highest per-timestep "
            "Base Policy action MSE samples."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "dataloader/tactile_dataloader.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "high_loss_analysis"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help=(
            "Number of highest-MSE "
            "action timesteps to save."
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help=(
            "Number of action timesteps "
            "shown before/after selected timestep."
        ),
    )

    parser.add_argument(
        "--split",
        choices=(
            "all",
            "train",
            "val",
            "test",
        ),
        default="all",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-steps",
        "--max-samples",
        dest="num_steps",
        type=int,
        default=None,
        help=(
            "Randomly sample N valid observations "
            "from the entire requested split. "
            "If omitted, analyze all valid observations."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used when --num-steps "
            "is specified."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Maintain global Top-K
# =============================================================================


def keep_top_k(
    heap: list[
        tuple[
            float,
            int,
            HighLossSample,
        ]
    ],
    sample: HighLossSample,
    top_k: int,
    serial: int,
) -> None:
    """Keep global Top-K samples using a min-heap."""

    item = (
        sample.mse,
        serial,
        sample,
    )

    if len(heap) < top_k:

        heapq.heappush(
            heap,
            item,
        )

    elif sample.mse > heap[0][0]:

        heapq.heapreplace(
            heap,
            item,
        )


# =============================================================================
# Plot one high-loss sample
# =============================================================================


def plot_sample(
    sample: HighLossSample,
    rank: int,
    window: int,
    output_dir: Path,
) -> int:
    """Plot every action dimension separately and one total-MSE figure."""

    ground_truth = (
        sample.ground_truth.numpy()
    )

    prediction = (
        sample.prediction.numpy()
    )

    timestep_mse = (
        sample.timestep_mse.numpy()
    )

    if ground_truth.ndim != 2:

        raise RuntimeError(
            "Expected ground_truth [K, D], "
            f"got {ground_truth.shape}"
        )

    if prediction.ndim != 2:

        raise RuntimeError(
            "Expected prediction [K, D], "
            f"got {prediction.shape}"
        )

    if ground_truth.shape != prediction.shape:

        raise RuntimeError(
            "Prediction and GT shape mismatch: "
            f"{prediction.shape} vs {ground_truth.shape}"
        )

    # -------------------------------------------------------------------------
    # Local action-chunk window
    # -------------------------------------------------------------------------

    start = max(
        0,
        sample.timestep - window,
    )

    end = min(
        len(timestep_mse),
        sample.timestep + window + 1,
    )

    x = list(
        range(start, end)
    )

    # -------------------------------------------------------------------------
    # One directory for this high-loss sample
    # -------------------------------------------------------------------------

    sample_dir = (
        output_dir
        / f"spike_{rank:02d}"
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_action_dims = (
        ground_truth.shape[-1]
    )

    image_count = 0

    # =========================================================================
    # Action dimensions
    # =========================================================================

    for action_dim in range(
        num_action_dims
    ):

        figure, axis = plt.subplots(
            figsize=(12, 5)
        )

        axis.plot(
            x,
            ground_truth[
                start:end,
                action_dim,
            ],
            label="Ground Truth",
            linewidth=2.0,
        )

        axis.plot(
            x,
            prediction[
                start:end,
                action_dim,
            ],
            linestyle="--",
            label="Base Policy Prediction",
            linewidth=2.0,
        )

        # Selected high-loss timestep
        axis.axvline(
            sample.timestep,
            linestyle=":",
            linewidth=1.5,
        )

        axis.set_xlabel(
            "Action timestep"
        )

        axis.set_ylabel(
            f"Action dim {action_dim}"
        )

        axis.set_title(
            f"Rank {rank} | Action dim {action_dim}\n"
            f"sample_index={sample.sample_index} | "
            f"timestep={sample.timestep} | "
            f"total MSE={sample.mse:.6f}"
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend()

        figure.tight_layout()

        figure.savefig(
            sample_dir
            / f"action_dim_{action_dim}.png",
            dpi=150,
        )

        plt.close(
            figure
        )

        image_count += 1

    # =========================================================================
    # Total action MSE
    # =========================================================================

    figure, axis = plt.subplots(
        figsize=(12, 5)
    )

    axis.plot(
        x,
        timestep_mse[start:end],
        linewidth=2.0,
        label="Mean Action MSE",
    )

    axis.scatter(
        [sample.timestep],
        [sample.mse],
        zorder=3,
        label="Selected timestep",
    )

    axis.axvline(
        sample.timestep,
        linestyle=":",
        linewidth=1.5,
    )

    axis.set_xlabel(
        "Action timestep"
    )

    axis.set_ylabel(
        "MSE"
    )

    axis.set_title(
        f"Rank {rank} | Mean Action MSE\n"
        f"sample_index={sample.sample_index} | "
        f"timestep={sample.timestep} | "
        f"MSE={sample.mse:.6f}"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        sample_dir / "mse.png",
        dpi=150,
    )

    plt.close(
        figure
    )

    image_count += 1

    return image_count


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    args = parse_args()

    # -------------------------------------------------------------------------
    # Validate arguments
    # -------------------------------------------------------------------------

    if args.top_k < 1:

        raise ValueError(
            "--top-k must be >= 1"
        )

    if args.window < 0:

        raise ValueError(
            "--window must be >= 0"
        )

    if (
        args.num_steps is not None
        and args.num_steps < 1
    ):

        raise ValueError(
            "--num-steps must be >= 1"
        )

    # -------------------------------------------------------------------------
    # Config + dataset
    # -------------------------------------------------------------------------

    config = load_yaml(
        args.config
    )

    dataset = build_base_dataset(
        config
    )

    # -------------------------------------------------------------------------
    # Base Policy
    # -------------------------------------------------------------------------

    (
        policy,
        preprocessor,
        postprocessor,
        device,
    ) = load_lerobot_policy(
        config,
        dataset,
    )

    policy.eval()

    # -------------------------------------------------------------------------
    # Get all valid indices
    # -------------------------------------------------------------------------

    all_valid_indices = (
        build_valid_indices(
            config,
            dataset,
            args.split,
        )
    )

    if not all_valid_indices:

        raise RuntimeError(
            "No valid samples found "
            "for requested split."
        )

    print(
        f"Total valid samples in split={args.split}: "
        f"{len(all_valid_indices)}"
    )

    # =========================================================================
    # RANDOM sampling
    # =========================================================================

    if args.num_steps is not None:

        sample_count = min(
            args.num_steps,
            len(all_valid_indices),
        )

        rng = random.Random(
            args.seed
        )

        valid_indices = rng.sample(
            all_valid_indices,
            sample_count,
        )

        # Sorting does NOT change which samples were randomly selected.
        # It only makes dataset/video access more sequential and efficient.
        valid_indices.sort()

        print(
            f"Randomly selected {sample_count} samples "
            f"from {len(all_valid_indices)} valid samples "
            f"using seed={args.seed}."
        )

    else:

        valid_indices = (
            all_valid_indices
        )

        print(
            "No --num-steps specified: "
            "analyzing the entire split."
        )

    # -------------------------------------------------------------------------
    # Dataset/model configuration
    # -------------------------------------------------------------------------

    dataset_cfg = (
        config["dataset"]
    )

    action_key = (
        dataset_cfg["keys"][
            "expert_action"
        ]
    )

    observation_keys = (
        dataset_cfg[
            "act_observation_keys"
        ]
    )

    action_horizon = int(
        config["sequence"][
            "action_horizon"
        ]
    )

    use_postprocessor = bool(
        config["policy"].get(
            "use_postprocessor",
            True,
        )
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(
            config["loader"][
                "batch_size"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # DataLoader
    #
    # shuffle=False because the random subset has already been selected above.
    # -------------------------------------------------------------------------

    loader = DataLoader(
        IndexedDataset(
            dataset,
            valid_indices,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=(
            collate_indexed_batch
        ),
    )

    # -------------------------------------------------------------------------
    # Global Top-K heap
    # -------------------------------------------------------------------------

    top_samples: list[
        tuple[
            float,
            int,
            HighLossSample,
        ]
    ] = []

    serial = 0

    print(
        f"\nAnalyzing {len(valid_indices)} samples "
        f"from split={args.split} "
        f"with batch_size={batch_size} "
        f"on {device}..."
    )

    # =========================================================================
    # Inference
    # =========================================================================

    with torch.inference_mode():

        for (
            batch_indices,
            cpu_batch,
        ) in tqdm(
            loader,
            total=len(loader),
            desc="Analyzing Base Policy",
            unit="batch",
        ):

            # -----------------------------------------------------------------
            # Observation
            # -----------------------------------------------------------------

            observation = {
                key: cpu_batch[key]
                for key in observation_keys
            }

            processed_observation = (
                preprocessor(
                    move_to_device(
                        observation,
                        device,
                    )
                )
            )

            # -----------------------------------------------------------------
            # Base Policy prediction
            # -----------------------------------------------------------------

            predicted = (
                policy.predict_action_chunk(
                    processed_observation
                )
            )

            if use_postprocessor:

                predicted = (
                    postprocessor(
                        predicted
                    )
                )

            prediction = (
                extract_action_tensor(
                    predicted
                )
            )

            prediction = (
                prediction[
                    :,
                    :action_horizon,
                ]
                .float()
                .cpu()
            )

            # -----------------------------------------------------------------
            # Ground Truth
            # -----------------------------------------------------------------

            ground_truth = (
                cpu_batch[action_key]
                .float()
            )

            if ground_truth.ndim != 3:

                raise RuntimeError(
                    "Expected expert action "
                    "[B, K, D], "
                    f"got {tuple(ground_truth.shape)}"
                )

            ground_truth = (
                ground_truth[
                    :,
                    :prediction.shape[1],
                ]
            )

            if (
                prediction.shape
                != ground_truth.shape
            ):

                raise RuntimeError(
                    "Prediction/GT shape mismatch: "
                    f"prediction="
                    f"{tuple(prediction.shape)}, "
                    f"GT="
                    f"{tuple(ground_truth.shape)}"
                )

            # =================================================================
            # Per-timestep MSE
            #
            # [B, K, D]
            #      ↓ mean over D
            # [B, K]
            # =================================================================

            timestep_mse = (
                (
                    prediction
                    - ground_truth
                )
                .square()
                .mean(dim=-1)
            )

            # -----------------------------------------------------------------
            # Only batch Top-K are needed.
            #
            # Any value outside a batch's Top-K cannot enter global Top-K
            # if that batch itself already contains >= K larger values.
            # -----------------------------------------------------------------

            flattened_mse = (
                timestep_mse.reshape(-1)
            )

            k = min(
                args.top_k,
                flattened_mse.numel(),
            )

            (
                batch_values,
                batch_flat_indices,
            ) = torch.topk(
                flattened_mse,
                k=k,
            )

            _, horizon = (
                timestep_mse.shape
            )

            # -----------------------------------------------------------------
            # Convert flattened index to:
            #
            # batch_index + action timestep
            # -----------------------------------------------------------------

            for (
                mse_value,
                flat_index,
            ) in zip(
                batch_values.tolist(),
                batch_flat_indices.tolist(),
            ):

                batch_index = (
                    flat_index
                    // horizon
                )

                timestep = (
                    flat_index
                    % horizon
                )

                sample = HighLossSample(
                    mse=float(
                        mse_value
                    ),
                    sample_index=int(
                        batch_indices[
                            batch_index
                        ]
                    ),
                    timestep=int(
                        timestep
                    ),
                    ground_truth=(
                        ground_truth[
                            batch_index
                        ]
                        .clone()
                    ),
                    prediction=(
                        prediction[
                            batch_index
                        ]
                        .clone()
                    ),
                    timestep_mse=(
                        timestep_mse[
                            batch_index
                        ]
                        .clone()
                    ),
                )

                keep_top_k(
                    heap=top_samples,
                    sample=sample,
                    top_k=args.top_k,
                    serial=serial,
                )

                serial += 1

    # =========================================================================
    # Sort global Top-K
    # =========================================================================

    samples = [
        item[2]
        for item in sorted(
            top_samples,
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    # =========================================================================
    # Output directories
    # =========================================================================

    output_dir = (
        args.output_dir
    )

    plots_dir = (
        output_dir
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # Save CSV + plots
    # =========================================================================

    csv_path = (
        output_dir
        / "high_loss_samples.csv"
    )

    total_images = 0

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rank",
                "sample_index",
                "timestep",
                "mse",
            ],
        )

        writer.writeheader()

        for rank, sample in enumerate(
            samples,
            start=1,
        ):

            writer.writerow(
                {
                    "rank": rank,
                    "sample_index": (
                        sample.sample_index
                    ),
                    "timestep": (
                        sample.timestep
                    ),
                    "mse": (
                        f"{sample.mse:.9f}"
                    ),
                }
            )

            generated_images = (
                plot_sample(
                    sample=sample,
                    rank=rank,
                    window=args.window,
                    output_dir=plots_dir,
                )
            )

            total_images += (
                generated_images
            )

    # =========================================================================
    # Summary
    # =========================================================================

    print()
    print(
        f"Saved CSV: {csv_path}"
    )

    print(
        f"Saved sample folders: "
        f"{plots_dir} "
        f"({len(samples)} samples)"
    )

    print(
        f"Saved images: "
        f"{total_images}"
    )

    if samples:

        num_action_dims = (
            samples[0]
            .ground_truth
            .shape[-1]
        )

        print(
            f"Images per sample: "
            f"{num_action_dims + 1} "
            f"({num_action_dims} action plots "
            f"+ 1 MSE plot)"
        )


if __name__ == "__main__":
    main()