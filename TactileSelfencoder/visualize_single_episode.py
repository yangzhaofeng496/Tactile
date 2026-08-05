"""
Visualize a single episode from the dataset with VQ-VAE token predictions.

Generates a video showing:
- Camera feed
- Current discrete token ID
- Token timeline
- Token distribution histogram
"""

import argparse
from pathlib import Path
import sys
import os

import torch
import numpy as np
import cv2
from tqdm import tqdm
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import dataloader.dataloader as dl
from TactileSelfencoder.vqvae_model import build_vqvae_from_config


def load_checkpoint(checkpoint_path, device):
    """Load VQ-VAE checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Val loss: {checkpoint['metrics']['loss']:.6f}")
    print(f"  Codebook usage: {checkpoint['metrics']['codebook_usage']*100:.2f}%")

    return checkpoint


def get_camera_image(batch, camera_keys):
    """Extract camera image from batch, trying multiple keys."""
    for key in camera_keys:
        if key in batch:
            img = batch[key]
            if isinstance(img, torch.Tensor):
                # Handle different image formats
                if img.ndim == 4:  # [B, C, H, W]
                    img = img[0]  # Take first batch
                if img.ndim == 3:  # [C, H, W]
                    img = img.permute(1, 2, 0)  # [H, W, C]
                img = img.cpu().numpy()

                # Normalize to 0-255 if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                # Convert RGB to BGR for OpenCV
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                return img

    # If no camera key found, return placeholder
    return np.zeros((480, 640, 3), dtype=np.uint8)


def create_visualization_frame(camera_img, token_id, token_history, token_counts, num_embeddings):
    """Create a 4-panel visualization frame."""
    # Panel dimensions
    camera_h, camera_w = camera_img.shape[:2]
    panel_w = 400
    panel_h = camera_h

    # Create panels
    token_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    timeline_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    dist_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Panel 1: Current token
    cv2.putText(token_panel, "Current Token", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(token_panel, f"ID: {token_id}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(token_panel, f"/ {num_embeddings}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

    # Panel 2: Token timeline
    cv2.putText(timeline_panel, "Token Timeline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if len(token_history) > 1:
        timeline_h = panel_h - 100
        timeline_w = panel_w - 40
        x_step = timeline_w / max(len(token_history) - 1, 1)
        y_scale = timeline_h / (num_embeddings + 1)

        # Draw timeline
        for i in range(len(token_history) - 1):
            x1 = int(20 + i * x_step)
            y1 = int(60 + (num_embeddings - token_history[i]) * y_scale)
            x2 = int(20 + (i + 1) * x_step)
            y2 = int(60 + (num_embeddings - token_history[i + 1]) * y_scale)

            color = (100, 200, 255) if i < len(token_history) - 2 else (0, 255, 255)
            thickness = 1 if i < len(token_history) - 2 else 2
            cv2.line(timeline_panel, (x1, y1), (x2, y2), color, thickness)

    # Panel 3: Token distribution
    cv2.putText(dist_panel, "Token Distribution", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if token_counts.sum() > 0:
        hist_h = panel_h - 100
        hist_w = panel_w - 40
        bar_w = hist_w / num_embeddings
        max_count = token_counts.max()

        if max_count > 0:
            for i in range(num_embeddings):
                bar_h = int((token_counts[i] / max_count) * hist_h)
                x = int(20 + i * bar_w)
                y = panel_h - 20 - bar_h

                color = (0, 255, 255) if i == token_id else (100, 100, 255)
                cv2.rectangle(dist_panel, (x, y),
                             (int(x + bar_w - 2), panel_h - 20), color, -1)

    # Combine panels horizontally
    top_row = np.hstack([camera_img, token_panel])
    bottom_row = np.hstack([timeline_panel, dist_panel])

    # Stack vertically
    frame = np.vstack([top_row, bottom_row])

    return frame


def load_episode_data(config, split='train', frames_per_episode=None, start_frame=None, end_frame=None):
    """Load episode data from dataset."""
    dataloader_config_path = project_root / config['data']['dataloader_config']
    dataloader_config = dl.load_yaml(dataloader_config_path)

    # Override batch size and workers for single-sample loading
    dataloader_config['loader']['batch_size'] = 1
    dataloader_config['loader']['num_workers'] = 0

    # Build base dataset
    base_dataset = dl.build_base_dataset(dataloader_config)

    print(f"Loading dataset: {dataloader_config['dataset']['repo_id']}")
    print(f"Total frames in dataset: {len(base_dataset)}")

    # Check if dataset has episode information
    has_episodes = hasattr(base_dataset, 'episodes') and base_dataset.episodes is not None

    if not has_episodes:
        print("⚠️  No episodes information, using frame-based splitting")

        # Calculate split boundaries
        total_frames = len(base_dataset)
        train_ratio = dataloader_config['split']['train']
        val_ratio = dataloader_config['split']['val']

        num_train = int(total_frames * train_ratio)
        num_val = int(total_frames * val_ratio)

        split_ranges = {
            'train': (0, num_train),
            'val': (num_train, num_train + num_val),
            'test': (num_train + num_val, total_frames)
        }

        split_start, split_end = split_ranges[split]
        print(f"Frame split: train={split_ranges['train'][1]-split_ranges['train'][0]}, "
              f"val={split_ranges['val'][1]-split_ranges['val'][0]}, "
              f"test={split_ranges['test'][1]-split_ranges['test'][0]}")

        # Determine frame range
        if start_frame is not None and end_frame is not None:
            # Manual frame range
            frame_start = split_start + start_frame
            frame_end = min(split_start + end_frame, split_end)
            print(f"\n✓ Manual frame range: [{start_frame}, {end_frame})")
            print(f"  Episode length: {frame_end - frame_start} frames")
        elif frames_per_episode is not None:
            # Fixed episode length
            frame_start = split_start
            frame_end = min(split_start + frames_per_episode, split_end)
            print(f"\n✓ Fixed episode length: {frames_per_episode} frames")
            print(f"  Actual length: {frame_end - frame_start} frames")
        else:
            # Use entire split
            frame_start = split_start
            frame_end = split_end
            print(f"\n✓ Using entire {split} split")
            print(f"  Episode length: {frame_end - frame_start} frames")

        frame_indices = list(range(frame_start, frame_end))
    else:
        # Use episode-based splitting (original logic)
        episode_bounds = dl.get_episode_bounds(base_dataset)

        # Split episodes
        episode_splits = dl.split_episode_ids(
            episode_ids=sorted(episode_bounds.keys()),
            train_ratio=float(dataloader_config['split']['train']),
            val_ratio=float(dataloader_config['split']['val']),
            test_ratio=float(dataloader_config['split']['test']),
            seed=int(dataloader_config['split']['seed'])
        )

        episode_ids = episode_splits[split]
        episode_id = episode_ids[0]  # Use first episode

        start_idx, end_idx = episode_bounds[episode_id]

        if frames_per_episode is not None:
            end_idx = min(start_idx + frames_per_episode, end_idx)

        frame_indices = list(range(start_idx, end_idx))
        print(f"Episode {episode_id}: {len(frame_indices)} frames")

    # Camera keys to try
    camera_keys = [
        'observation.images.realsense',
        'wrist_cam',
        'top',
        'side',
        'observation.images.top',
    ]

    return base_dataset, frame_indices, camera_keys


def visualize_episode(model, config, split='train', output_path='episode_visualization.mp4',
                     frames_per_episode=None, start_frame=None, end_frame=None):
    """Generate video visualization for an episode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load episode data
    base_dataset, frame_indices, camera_keys = load_episode_data(
        config, split, frames_per_episode, start_frame, end_frame
    )

    num_embeddings = config['model']['quantizer']['num_embeddings']
    history_steps = config['model']['input']['history_steps']

    # Storage for visualization
    token_history = []
    token_counts = np.zeros(num_embeddings, dtype=np.int32)

    # Video writer (will be initialized with first frame)
    video_writer = None
    fps = 30

    print(f"\nRendering episode to video...")
    print(f"Output: {output_path}")
    print(f"Total frames: {len(frame_indices)}")

    with torch.no_grad():
        for idx, frame_idx in enumerate(tqdm(frame_indices, desc="Rendering")):
            # Get data from dataset
            sample = base_dataset[frame_idx]

            # Extract tactile history
            tactile_history = sample['tactile_history'].unsqueeze(0).to(device)  # [1, T, D]

            # Trim to model's expected history length
            if tactile_history.shape[1] > history_steps:
                tactile_history = tactile_history[:, :history_steps, :]

            # Get token prediction
            indices, z_q = model.encode(tactile_history)
            token_id = indices[0].item()

            # Update statistics
            token_history.append(token_id)
            token_counts[token_id] += 1

            # Get camera image
            camera_img = get_camera_image(sample, camera_keys)

            # Create visualization frame
            frame = create_visualization_frame(
                camera_img, token_id, token_history,
                token_counts, num_embeddings
            )

            # Initialize video writer with first frame
            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            # Write frame
            video_writer.write(frame)

            if (idx + 1) % 50 == 0:
                print(f"  Progress: {idx + 1}/{len(frame_indices)} frames")

    if video_writer is not None:
        video_writer.release()

    print(f"\n✓ Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize single episode with VQ-VAE tokens")
    parser.add_argument('--config', type=str,
                       default='TactileSelfencoder/vqvae_config.yaml',
                       help='Path to VQ-VAE config')
    parser.add_argument('--checkpoint', type=str,
                       default='TactileSelfencoder/vqvae_checkpoints/checkpoint_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--output', type=str, default='episode_visualization.mp4',
                       help='Output video path')
    parser.add_argument('--frames-per-episode', type=int, default=None,
                       help='Number of frames to visualize (default: entire split)')
    parser.add_argument('--start-frame', type=int, default=None,
                       help='Start frame index within split (requires --end-frame)')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='End frame index within split (requires --start-frame)')

    args = parser.parse_args()

    # Validate frame range arguments
    if (args.start_frame is not None) != (args.end_frame is not None):
        parser.error("--start-frame and --end-frame must be used together")

    if args.start_frame is not None and args.end_frame is not None:
        if args.start_frame >= args.end_frame:
            parser.error("--start-frame must be less than --end-frame")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = build_vqvae_from_config(args.config)

    checkpoint = load_checkpoint(args.checkpoint, torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate visualization
    visualize_episode(
        model, config,
        split=args.split,
        output_path=args.output,
        frames_per_episode=args.frames_per_episode,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )


if __name__ == '__main__':
    main()
