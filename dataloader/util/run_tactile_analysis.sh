#!/usr/bin/env bash
python "tactile_analysis.py" \
  --config "../tactile_dataloader.yaml" \
  --analysis-config "tactile_analysis.yaml" \
  --output-dir "tactile_analysis_output"
