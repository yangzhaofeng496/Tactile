#!/bin/bash

# T-Rex VQ-VAE Training Pipeline (Adapted for 2-finger system)
# This script must be run from the project root directory

set -e  # Exit on error

echo "========================================"
echo "T-Rex VQ-VAE 2-Finger Training Pipeline"
echo "========================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root (parent of trex/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"
echo "Working directory: $PWD"
echo ""

# Paths (relative to project root)
DATA_CONFIG="dataloader/trex_force_dataloader.yaml"
MODEL_CONFIG="trex/trex_10finger_config.yaml"
STATS_FILE="trex/trex_10finger_stats.json"
OUTPUT_DIR="outputs/trex_vqvae"

# Step 1: Compute statistics (skip if already exists)
if [ ! -f "$STATS_FILE" ]; then
    echo ""
    echo "Step 1: Computing tactile statistics..."
    python trex/compute_trex_stats.py \
        --data_config "$DATA_CONFIG" \
        --output "$STATS_FILE"
    echo "✓ Statistics saved to: $STATS_FILE"
else
    echo ""
    echo "Step 1: Statistics file already exists, skipping..."
    echo "  File: $STATS_FILE"
fi

# Step 2: Train VQ-VAE
echo ""
echo "Step 2: Training T-Rex VQ-VAE..."
python trex/train_trex_vqvae.py \
    --config "$MODEL_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --stats "$STATS_FILE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✓ Training complete! Checkpoint saved to: $OUTPUT_DIR"

# Step 3: Run inference and visualization
echo ""
echo "Step 3: Running inference and visualization..."
python trex/inference_trex.py \
    --checkpoint "$OUTPUT_DIR/latest.pt" \
    --data_config "$DATA_CONFIG" \
    --output "$OUTPUT_DIR/visualizations" \
    --n_samples 20 \
    --max_encode 5000

echo ""
echo "✓ Visualizations saved to: $OUTPUT_DIR/visualizations"
echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Model checkpoint: $OUTPUT_DIR/latest.pt"
echo "  - Best checkpoint: $OUTPUT_DIR/best.pt"
echo "  - Visualizations: $OUTPUT_DIR/visualizations/"
echo "  - Training stats: $STATS_FILE"
echo ""
