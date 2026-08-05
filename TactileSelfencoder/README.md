# Tactile VQ-VAE

Vector Quantized Variational AutoEncoder (VQ-VAE) for encoding tactile force history into discrete tokens.

## Model Architecture

- **Encoder**: 1D CNN that processes temporal force history [B, T, D] → continuous embedding [B, 256]
- **Quantizer**: EMA-based vector quantizer with RMS normalization and dead code replacement
- **Decoder**: MLP that reconstructs force history from quantized embeddings

## Key Features

- **RMS Normalization**: Applied before quantization for scale consistency
- **EMA Codebook Update**: Exponential moving average instead of gradient descent
- **Dead Code Replacement**: Automatically reinitializes unused codebook entries
- **Commitment Loss**: Encourages encoder to commit to codebook entries

## Files

```
TactileSelfencoder/
├── vqvae_config.yaml           # Model and training configuration
├── vqvae_model.py              # Model implementation
├── train_vqvae.py              # Training script
├── visualize_tokens.py         # Validation set visualization
├── visualize_single_episode.py # Single episode visualization
└── vqvae_checkpoints/          # Saved model checkpoints
```

## Training

```bash
python TactileSelfencoder/train_vqvae.py
```

Resume from checkpoint:
```bash
python TactileSelfencoder/train_vqvae.py --resume TactileSelfencoder/vqvae_checkpoints/checkpoint_best.pth
```

## Visualization

Visualize validation set:
```bash
python TactileSelfencoder/visualize_tokens.py --output test_visualization.mp4
```

Visualize specific frames:
```bash
python TactileSelfencoder/visualize_single_episode.py --split val --start-frame 0 --end-frame 500
```

Visualize entire split:
```bash
python TactileSelfencoder/visualize_single_episode.py --split train
```

## Configuration

Key parameters in `vqvae_config.yaml`:

- `history_steps: 15` - Temporal window size
- `force_dim: 12` - Force sensor dimensions
- `num_embeddings: 16` - Codebook size
- `embedding_dim: 256` - Latent dimension
- `commitment_cost: 500.0` - Commitment loss weight
- `ema_decay: 0.9` - EMA update rate

## Loss Formulas

**Total Loss**:
```
total_loss = recon_loss + vq_loss
```

**Reconstruction Loss**:
```
recon_loss = MSE(x_recon, x_original)
```

**VQ Loss** (EMA mode):
```
vq_loss = commitment_loss
commitment_loss = β × MSE(z_e_normalized, stop_gradient(z_q))
```

**Codebook Update** (no gradients):
```
cluster_size_t = decay × cluster_size_{t-1} + (1-decay) × current_usage
embedding_sum_t = decay × embedding_sum_{t-1} + (1-decay) × Σ(z_e_normalized)
codebook_t = embedding_sum_t / cluster_size_t
```

## Requirements

- PyTorch
- OpenCV (for visualization)
- wandb (optional, for logging)
- LeRobot dataset format

## Output

The model outputs:
- Discrete token indices for each force history window
- Quantized embeddings for downstream tasks
- Reconstruction of input force history
