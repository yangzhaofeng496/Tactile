#!/bin/bash
# VQ-VAE Training Script Launcher

# Navigate to project root
cd /home/yang/TactileEncoder

# Run training script from project root
python -c "
import sys
sys.path.insert(0, '.')

# Now execute the training script
exec(open('TactileSelfencoder/train_vqvae.py').read())
" --config TactileSelfencoder/vqvae_config.yaml "$@"
