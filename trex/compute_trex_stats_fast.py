"""Compute per-finger per-channel stats directly from parquet (fast path).

Same output format as trex/compute_trex_stats.py but iterates the raw parquet
files vectorized instead of through the (slow) dataloader.

Usage:
    python trex/compute_trex_stats_fast.py \
        --data_dir dataset/trex_force_data \
        --n_fingers 10 --force_dim 6 \
        --output trex/trex_10finger_stats.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_fingers", type=int, default=10)
    parser.add_argument("--force_dim", type=int, default=6)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(glob.glob(str(data_dir / "data" / "chunk-000" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}/data")

    n_f, d = args.n_fingers, args.force_dim
    sum_vals = np.zeros((n_f, d))
    sum_sq_vals = np.zeros((n_f, d))
    min_vals = np.full((n_f, d), np.inf)
    max_vals = np.full((n_f, d), -np.inf)
    count = 0

    for f in files:
        df = pd.read_parquet(f, columns=["observation.tactile_force"])
        arr = np.stack(df["observation.tactile_force"].values)  # [N, n_f*d]
        arr = arr.reshape(-1, n_f, d)  # [N, n_f, d]
        sum_vals += arr.sum(axis=0)
        sum_sq_vals += (arr ** 2).sum(axis=0)
        min_vals = np.minimum(min_vals, arr.min(axis=0))
        max_vals = np.maximum(max_vals, arr.max(axis=0))
        count += arr.shape[0]

    mean = sum_vals / count
    var = np.maximum((sum_sq_vals / count) - mean ** 2, 1e-8)
    std = np.maximum(np.sqrt(var), 1e-6)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": min_vals.tolist(),
        "max": max_vals.tolist(),
        "count": int(count),
        "n_samples": int(count),
        "n_fingers": n_f,
        "force_dim": d,
        "source": str(data_dir.resolve()),
        "description": "Per-finger per-channel statistics for T-Rex VQ-VAE "
                       f"({n_f}-finger bi-manual). Computed from all parquet frames.",
        "channel_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        "finger_order": [f"finger_{i}" for i in range(n_f)],
        "units": "raw sensor units (pre-normalization)",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"✓ saved {out}  (frames={count:,})")


if __name__ == "__main__":
    main()
