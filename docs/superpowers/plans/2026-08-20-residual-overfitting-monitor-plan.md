# Residual ACT Overfitting Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reliable train/validation/test isolation, epoch-level test-loss overfitting detection, automatic stopping, and reproducible experiment artifacts to the residual ACT trainer.

**Architecture:** Keep the existing training loop and add small pure helpers for overfitting state and curve persistence. Validation and test remain calls to the existing evaluator, but both run with `update_weights=False`; test metrics are observed only and never modify training state. Each run writes a unique experiment directory containing a metrics CSV, plot, stop reason, and checkpoint.

**Tech Stack:** Python, PyTorch, PyYAML, matplotlib, existing W&B logging.

**Spec:** `docs/superpowers/specs/2026-08-20-residual-overfitting-monitor-design.md`

## Global Constraints

- Preserve existing model input/output shapes and cache format.
- Do not delete existing checkpoints, caches, or logs.
- Do not use test loss for optimizer updates, dropout changes, learning-rate changes, or best-checkpoint selection.
- Stop only after an epoch completes and its metrics are persisted.
- Keep the default detector configurable and disabled only by an explicit configuration value.

### Task 1: Add failing detector and metrics tests

**Files:**
- Create: `test_overfitting_monitor.py`

- [ ] **Step 1: Write tests for the required detector behavior**

```python
from train import OverfitMonitor


def test_detector_waits_for_three_rising_test_epochs():
    monitor = OverfitMonitor(patience=3, min_relative_increase=0.01)
    assert not monitor.update(10.0, 9.0, 8.0)
    assert not monitor.update(9.0, 8.5, 8.1)
    assert not monitor.update(8.0, 8.0, 8.2)
    assert not monitor.update(7.0, 7.5, 8.3)
    assert monitor.update(6.0, 7.0, 8.4)


def test_detector_does_not_stop_when_test_rise_is_below_threshold():
    monitor = OverfitMonitor(patience=3, min_relative_increase=0.01)
    for value in (8.0, 8.01, 8.02, 8.03, 8.04):
        assert not monitor.update(7.0, 7.0, value)


def test_detector_serializes_epoch_metrics():
    monitor = OverfitMonitor(patience=3, min_relative_increase=0.01)
    monitor.update(10.0, 9.0, 8.0)
    assert monitor.history[0]["epoch"] == 1
    assert monitor.history[0]["test_loss"] == 8.0
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run: `python -m pytest -q test_overfitting_monitor.py`

Expected: FAIL because `OverfitMonitor` is not yet defined.

### Task 2: Implement detector and artifact persistence

**Files:**
- Modify: `train.py`
- Test: `test_overfitting_monitor.py`

- [ ] **Step 1: Implement `OverfitMonitor`**

Add a class with constructor `OverfitMonitor(patience=3, min_relative_increase=0.01)`, method `update(train_loss, val_loss, test_loss) -> bool`, public `history`, `best_test_loss`, and `stop_reason`. The method must count only consecutive test increases after the best test value and require train or val improvement.

- [ ] **Step 2: Run detector tests**

Run: `python -m pytest -q test_overfitting_monitor.py`

Expected: PASS.

- [ ] **Step 3: Add CSV and PNG persistence**

Add helpers that write monitor history to `<experiment_dir>/loss_history.csv` and plot train/val/test curves to `<experiment_dir>/loss_curve.png`. Save the stop reason to `<experiment_dir>/stop_reason.json`.

- [ ] **Step 4: Compile the modified trainer**

Run: `python -m py_compile train.py test_overfitting_monitor.py`

Expected: exit code 0.

### Task 3: Isolate validation and test from training

**Files:**
- Modify: `train.py`
- Test: `test_overfitting_monitor.py`

- [ ] **Step 1: Add a source-level regression assertion**

Assert that the main validation call uses `update_weights=False` and that no test branch changes `reference_dropout`.

- [ ] **Step 2: Change the validation call**

Remove optimizer/scaler from the validation call and pass `update_weights=False`.

- [ ] **Step 3: Remove test-driven dropout adaptation**

Delete the branch that compares `test_loss` with `previous_test_loss` and changes `reference_dropout`; retain test logging only.

- [ ] **Step 4: Run the regression and compile checks**

Run: `python -m pytest -q test_overfitting_monitor.py && python -m py_compile train.py`

Expected: PASS and exit code 0.

### Task 4: Integrate epoch-boundary stopping and experiment artifacts

**Files:**
- Modify: `train.py`
- Modify: `config/model_config.yaml`

- [ ] **Step 1: Add detector configuration**

Add:

```yaml
overfit_monitor:
  enabled: true
  patience: 3
  min_relative_test_increase: 0.01
```

- [ ] **Step 2: Instantiate the monitor after loading the training configuration**

Use the configured values and a run-local output directory. Do not reuse an existing checkpoint directory for a new experiment.

- [ ] **Step 3: Update the monitor after test evaluation**

After train, val, and test losses are available, call `monitor.update(...)`, log the result, and persist CSV/PNG/JSON before breaking the epoch loop when it returns true.

- [ ] **Step 4: Save a stop checkpoint before exiting**

When triggered, save the current checkpoint with `stop_reason` and monitor history metadata, then exit the training loop cleanly.

- [ ] **Step 5: Run a one-epoch diagnostic smoke test**

Run the existing diagnostic mode with the current cache and verify the process emits train, val, and test metrics without updating validation weights.

### Task 5: Execute the baseline experiment and document evidence

**Files:**
- Create: `outputs/residual_overfit_experiments/<run-id>/`
- Modify: none unless experiment evidence identifies a root cause

- [ ] **Step 1: Stop the old uninstrumented training process at an epoch-safe boundary**

Record its log path and preserve its checkpoints. Do not delete files.

- [ ] **Step 2: Start the instrumented baseline with `act_cache_450.pt`**

Use the reduced 253,204-parameter model and action noise `0.02`. Capture stdout and the run configuration in the experiment directory.

- [ ] **Step 3: Monitor the generated metrics artifacts**

Verify the detector stops only on its configured condition, or continue to the configured epoch limit if no overfitting is detected.

- [ ] **Step 4: Analyze the first stopped run before changing code**

Compare train/val/test curves, episode counts, cache keys, and per-split distributions. Select exactly one next hypothesis.

### Task 6: Run evidence-driven single-variable experiments

**Files:**
- Modify: the smallest relevant config or dataset-index code for each experiment
- Create: one unique output directory per experiment

- [ ] **Step 1: Test window redundancy**

Run a configuration that keeps one window every 10 frames or limits windows per episode, leaving model and optimizer unchanged.

- [ ] **Step 2: Test modality dependence**

Run action-only and full-input controls, changing only the selected input branches.

- [ ] **Step 3: Test regularization**

Change only weight decay or action-noise strength, keeping the split and cache fixed.

- [ ] **Step 4: Compare all experiments using the same detector and metrics**

Record the best validation loss, corresponding test loss, stop epoch, and test-loss minimum for every run.

- [ ] **Step 5: Stop after three unsuccessful evidence-driven modifications**

If all three fail, stop tuning and write the architecture/data-distribution conclusion.

### Task 7: Send the final Gmail report

**Files:**
- Create: `outputs/residual_overfit_experiments/final_report.md`

- [ ] **Step 1: Assemble the report**

Include experiment table, plots, detector events, root-cause evidence, and recommended next action.

- [ ] **Step 2: Send the report to the authenticated user mailbox**

Use Gmail self-delivery with subject `Residual ACT 过拟合分析报告` and include the report body plus local artifact paths.

- [ ] **Step 3: Verify the send result**

Confirm Gmail reports the message was sent and report the user-meaningful subject and destination.
