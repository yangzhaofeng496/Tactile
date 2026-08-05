"""
Visualize VQ-VAE token predictions on validation set.

Generates a video showing camera feed alongside the discrete token ID
predicted by the VQ-VAE model for each frame's tactile force history.
"""

import argparse
from pathlib import Path
import sys
import os

import torch
import numpy as np
import cv2
from tqdm import tqdm

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
    """Extract camera image from batch, trying multiple possible keys."""
    for key in camera_keys:
        if key in batch:
            img = batch[key]
            if isinstance(img, torch.Tensor):
                # Handle different image tensor formats
                if img.ndim == 4:  # [B, C, H, W]
                    img = img[0]  # Take first batch element
                if img.ndim == 3:  # [C, H, W]
                    img = img.permute(1, 2, 0)  # Convert to [H, W, C]
                img = img.cpu().numpy()

                # Normalize to 0-255 range if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                # Convert RGB to BGR for OpenCV
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                return img

    # If no camera key found, return black placeholder
    return np.zeros((480, 640, 3), dtype=np.uint8)


def create_visualization_frame(camera_img, token_id, token_history, token_counts, num_embeddings):
    """
    Create a 4-panel visualization frame:
    - Top-left: Camera feed
    - Top-right: Current token ID display
    - Bottom-left: Token timeline
    - Bottom-right: Token distribution histogram
    """
    # Get camera dimensions
    camera_h, camera_w = camera_img.shape[:2]

    # Create panels for token info
    panel_w = 400
    panel_h = camera_h

    # Panel 1: Current token (top-right)
    token_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(token_panel, "Current Token", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(token_panel, f"ID: {token_id}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(token_panel, f"Codebook: {num_embeddings}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

    # Panel 2: Token timeline (bottom-left)
    timeline_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(timeline_panel, "Token Timeline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if len(token_history) > 1:
        # Draw timeline
        timeline_h = panel_h - 100
        timeline_w = panel_w - 40
        x_step = timeline_w / max(len(token_history) - 1, 1)
        y_scale = timeline_h / (num_embeddings + 1)

        for i in range(len(token_history) - 1):
            x1 = int(20 + i * x_step)
            y1 = int(60 + (num_embeddings - token_history[i]) * y_scale)
            x2 = int(20 + (i + 1) * x_step)
            y2 = int(60 + (num_embeddings - token_history[i + 1]) * y_scale)

            # Current segment in bright color
            color = (100, 200, 255) if i < len(token_history) - 2 else (0, 255, 255)
            thickness = 1 if i < len(token_history) - 2 else 2
            cv2.line(timeline_panel, (x1, y1), (x2, y2), color, thickness)

    # Panel 3: Token distribution (bottom-right)
    dist_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(dist_panel, "Token Distribution", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if token_counts.sum() > 0:
        # Draw histogram
        hist_h = panel_h - 100
        hist_w = panel_w - 40
        bar_w = hist_w / num_embeddings
        max_count = token_counts.max()

        if max_count > 0:
            for i in range(num_embeddings):
                bar_h = int((token_counts[i] / max_count) * hist_h)
                x = int(20 + i * bar_w)
                y = panel_h - 20 - bar_h

                # Highlight current token
                color = (0, 255, 255) if i == token_id else (100, 100, 255)
                cv2.rectangle(dist_panel, (x, y),
                            (int(x + bar_w - 2), panel_h - 20), color, -1)

    # Combine panels
    top_row = np.hstack([camera_img, token_panel])
    bottom_row = np.hstack([timeline_panel, dist_panel])

    # Stack vertically
    frame = np.vstack([top_row, bottom_row])

    return frame


def visualize_validation_set(model, config, output_path='test_visualization.mp4', max_frames=None):
    """Generate video visualization for validation set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Load dataloader configuration
    dataloader_config_path = project_root / config['data']['dataloader_config']
    dataloader_config = dl.load_yaml(dataloader_config_path)

    # Override settings for visualization
    dataloader_config['loader']['batch_size'] = 1
    dataloader_config['loader']['num_workers'] = 0

    print("Building dataloader...")
    base_dataset = dl.build_base_dataset(dataloader_config)

    dataloaders_dict, datasets_dict = dl.build_normal_dataloaders(
        dataloader_config,
        base_dataset
    )

    val_loader = dataloaders_dict['val']

    print(f"Validation set: {len(val_loader)} batches")

    # Camera keys to try (in priority order)
    camera_keys = [
        'observation.images.realsense',
        'wrist_cam',
        'top',
        'side',
        'observation.images.top',
    ]

    num_embeddings = config['model']['quantizer']['num_embeddings']

    # Storage for visualization data
    token_history = []
    token_counts = np.zeros(num_embeddings, dtype=np.int32)

    # Video writer (will be initialized with first frame)
    video_writer = None
    fps = 30

    print(f"\nGenerating visualization video: {output_path}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing")):
            if max_frames is not None and batch_idx >= max_frames:
                break

            # Extract tactile history
            tactile_history = batch['tactile_history'].to(device)  # [1, T, D]

            # Get token prediction
            indices, z_q = model.encode(tactile_history)
            token_id = indices[0].item()

            # Update statistics
            token_history.append(token_id)
            token_counts[token_id] += 1

            # Get camera image
            camera_img = get_camera_image(batch, camera_keys)

            # Create visualization frame
            frame = create_visualization_frame(
                camera_img, token_id, token_history,
                token_counts, num_embeddings
            )

            # Initialize video writer with first frame dimensions
            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            # Write frame
            video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()

    print(f"\n✓ Video saved to: {output_path}")
    print(f"  Total frames: {len(token_history)}")
    print(f"  Unique tokens used: {(token_counts > 0).sum()}/{num_embeddings}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE tokens on validation set")
    parser.add_argument('--config', type=str,
                       default='TactileSelfencoder/vqvae_config.yaml',
                       help='Path to VQ-VAE config file')
    parser.add_argument('--checkpoint', type=str,
                       default='TactileSelfencoder/vqvae_checkpoints/checkpoint_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='test_visualization.mp4',
                       help='Output video path')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to visualize (default: all)')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = build_vqvae_from_config(args.config)

    checkpoint = load_checkpoint(args.checkpoint, torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate visualization
    visualize_validation_set(model, config, args.output, args.max_frames)


if __name__ == '__main__':
    main()
