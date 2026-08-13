"""Build the LeRobot v3.0 meta/ directory for dataset/trex_force_data.

The dataset currently only has data/chunk-000/*.parquet (no videos, no meta/).
This script regenerates meta/info.json, meta/episodes and meta/tasks.parquet
from the existing parquet data so LeRobotDataset can open it.

Usage:
    python dataloader/build_trex_meta.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "dataset" / "trex_force_data"
CHUNK_DIR = DATA_DIR / "data" / "chunk-000"


def main():
    files = sorted(CHUNK_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {CHUNK_DIR}")

    print(f"Found {len(files)} data files")

    # Determine features and total frames from the first file.
    df0 = pd.read_parquet(files[0])
    feature_cols = [c for c in df0.columns if c not in ("index", "timestamp", "frame_index", "episode_index", "task_index")]

    features = {}
    for c in feature_cols:
        v = np.asarray(df0[c].iloc[0])
        features[c] = {
            "dtype": str(v.dtype) if v.dtype != object else "float32",
            "shape": list(v.shape) if v.ndim else [1],
            "names": None,
        }
    for c in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[c] = {
            "dtype": str(df0[c].dtype),
            "shape": [1],
            "names": None,
        }

    # fps from first two timestamps
    ts = df0["timestamp"].values
    fps = int(round(1.0 / float(ts[1] - ts[0])))

    # Build per-episode metadata across all files.
    rows = []
    total_frames = 0
    all_task_indices = set()
    for f in files:
        df = pd.read_parquet(f, columns=["index", "episode_index", "task_index"])
        total_frames += len(df)
        all_task_indices.update(df["task_index"].unique().tolist())
        for ep_idx, grp in df.groupby("episode_index"):
            rows.append((
                int(ep_idx),
                int(grp["index"].min()),
                int(grp["index"].max()) + 1,
                int(grp["task_index"].iloc[0]),
            ))

    rows.sort()
    n_episodes = len(rows)

    # tasks: one entry per distinct task_index in the data
    task_indices = sorted(all_task_indices)
    task_names = {t: f"task_{t}" for t in task_indices}
    tasks_df = pd.DataFrame(
        {"task_index": task_indices},
        index=pd.Index([task_names[t] for t in task_indices], name="task"),
    )

    episodes_rows = []
    for ep_idx, from_idx, to_idx, task_idx in rows:
        episodes_rows.append({
            "episode_index": ep_idx,
            "tasks": [task_names[task_idx]],
            "length": to_idx - from_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
        })
    episodes_df = pd.DataFrame(episodes_rows)
    episodes_df = episodes_df.sort_values("dataset_from_index").reset_index(drop=True)

    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "splits": {"train": f"0:{n_episodes}"},
        "features": features,
    }

    meta_dir = DATA_DIR / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    episodes_df.to_parquet(episodes_dir / "file-000.parquet", index=False)
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    print(f"  fps: {fps}")
    print(f"  episodes: {n_episodes}")
    print(f"  total_frames: {total_frames}")
    print(f"  tasks: {len(task_indices)}")
    print(f"  features: {sorted(features)}")
    print(f"✓ meta/ written to {meta_dir}")


if __name__ == "__main__":
    main()
