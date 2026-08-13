# T-Rex Integration

This directory contains all T-Rex (two-finger tactile sensor) related code and configurations, kept separate from the core TactileSelfencoder implementation.

## Quick Start

```bash
# Train T-Rex VQ-VAE
./run_trex_vqvae.sh

# Or train step by step
python train_trex_vqvae.py \
    --config trex_2finger_config.yaml \
    --data_config ../dataloader/vqvae_tactile.yaml \
    --stats trex_tactile_stats.json \
    --output_dir ../outputs/trex_vqvae
```

## Structure

```
trex/
├── README.md                          # This file
├── run_trex_vqvae.sh                  # One-click training script
│
├── train_trex_vqvae.py                # Training script
├── inference_trex.py                  # Inference script
├── inference_trex_vqvae.py            # VQ-VAE specific inference
├── compute_trex_stats.py              # Compute dataset statistics
│
├── trex_vqvae_model.py                # Model definition
├── trex_2finger_config.yaml           # Training config (30 epochs)
├── trex_2finger_test_config.yaml      # Test config (3 epochs)
├── trex_vqvae_config.yaml             # Model config
├── trex_tactile_stats.json            # Dataset statistics
│
├── trex_official/                     # Official T-Rex implementation
│   ├── encoder.py
│   ├── decoder.py
│   ├── quantizer.py
│   └── tactile_vqvae.py
│
├── trex_docs/                         # Test files
├── T-Rex-official/                    # Original repository
└── docs/                              # Detailed documentation
    ├── TREX_DATALOADER_INTEGRATION.md
    ├── TREX_INTEGRATION_COMPLETE.md
    └── TREX_QUICK_REFERENCE.md
```

## Key Files

- **Training**: `train_trex_vqvae.py`, `run_trex_vqvae.sh`
- **Inference**: `inference_trex.py`, `inference_trex_vqvae.py`
- **Model**: `trex_vqvae_model.py`, `trex_official/`
- **Config**: `trex_2finger_config.yaml`, `trex_vqvae_config.yaml`
- **Data**: `trex_tactile_stats.json`

## Documentation

Detailed documentation is in the [docs/](docs/) directory.
