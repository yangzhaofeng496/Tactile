# Residual ACT Overfit Experiments Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with experiment checkpoints.

**Goal:** Identify and reduce residual ACT overfitting on the 457-episode dataset using reproducible, paper-backed, single-variable experiments.

**Architecture:** Keep the episode-disjoint train/validation/test split fixed at seed 11. Run each configuration in its own checkpoint directory, select checkpoints by validation loss, and use test loss only as an overfitting alarm and final diagnostic.

**Tech Stack:** Python, PyTorch, YAML configuration, tmux, CSV/PNG experiment logs.

**Spec:** Approved in the conversation on 2026-08-21.

## Global Constraints

- Dataset: `/home/yang/TactileEncoder/dataset/so101/400_50hil_820`.
- Cache: `/home/yang/TactileEncoder/outputs/act_cache/act_cache_400_50hil_820.pt`.
- Split: train 0.80, validation 0.10, test 0.10, seed 11, split by complete episode.
- One experimental variable changes at a time.
- Early stop when test loss rises for three consecutive epochs while train or validation continues improving.
- `checkpoint_best.pth` is selected by validation loss, never by test loss.

### Task 1: Audit and freeze the split

- [x] Verify the dataset has 457 episodes and 290905 frames.
- [x] Verify the split contains 365/46/46 episodes.
- [x] Verify all pairwise episode-set intersections are empty.

### Task 2: Temporal smoothness sweep

- [ ] Run the approved baseline with temporal smoothness weight `0.005`.
- [ ] Compare train, validation, and test curves against the `0.01` baseline.
- [ ] Keep the validation-selected checkpoint and record the raw test minimum separately.
- [ ] Repeat with `0.02` if the weaker constraint does not improve generalization.

### Task 3: DART-inspired state-noise sweep

- [ ] Test state relative noise `0.025`, `0.05`, and `0.075` one at a time.
- [ ] Keep force noise at `0.05` and temporal weight fixed during this sweep.
- [ ] Compare the validation-selected checkpoints.

### Task 4: Action-chunk consistency experiment

- [ ] Add an overlap-consistency loss for adjacent action chunks only after the parameter sweeps.
- [ ] Verify the loss with a focused unit test before training.
- [ ] Run the method with a small coefficient and compare against the best preceding configuration.

### Task 5: Replication and report

- [ ] Re-run the best configuration with a second seed.
- [ ] Generate the final loss curves and checkpoint summary.
- [ ] Report whether the evidence supports reduced overfitting or only a lower test loss.
