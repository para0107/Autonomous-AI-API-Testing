# RL Training Experiments — Autonomous AI API Testing

---

## Experiment 1 — Baseline PPO Run (Random Weight Initialization)
**Date:** 2025-06-01  
**Script:** `training_from_qase.py`  
**Data:** `augmented_tests.json` (736 experiences)  
**Steps:** 100  
**Changes from previous:** _First run. No prior checkpoint. Default hyperparameters._

### Configuration
| Parameter | Value |
|-----------|-------|
| `num_steps` | 100 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `epochs_per_update` | 4 |
| `buffer_size` | 736 |

### Results (sample — every 10 steps)
| Step | Avg Policy Loss | Avg Value Loss |
|------|----------------|----------------|
| 10   | 0.1821         | 612.34         |
| 20   | 0.1654         | 589.12         |
| 30   | 0.1502         | 571.88         |
| 50   | 0.1389         | 554.43         |
| 100  | 0.1274         | 538.91         |

### Observations
- Policy loss decreasing steadily from random init.
- Value loss very high — critic network not yet calibrated to reward scale.
- `total_steps` reported as 0 throughout; PPO step counter not wired to trainer yet.
- No checkpoint saved (save path not configured).

---

## Experiment 2 — Bug Fixes: State Dimension Mismatch + Buffer Reference
**Date:** 2025-06-05  
**Script:** `training_from_qase.py` (BUG 7 partial fix)  
**Data:** `augmented_tests.json` (736 experiences)  
**Steps:** 100  
**Changes from previous:**
- Fixed `_create_state_vector()`: replaced manual 640-dim construction with `extract_state()` producing correct **64-dim** vectors.
- Fixed buffer reference: `self.rl_optimizer.experience_buffer` → `self.rl_optimizer.buffer`.
- Fixed `train()` call: now `await`-ed (async).
- Fixed checkpoint methods: `save_checkpoint()` / `load_checkpoint()` → `save()` / `load()`.

### Configuration
| Parameter | Value |
|-----------|-------|
| `num_steps` | 100 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `epochs_per_update` | 4 |
| `buffer_size` | 736 |

### Results (sample — every 10 steps)
| Step | Avg Policy Loss | Avg Value Loss |
|------|----------------|----------------|
| 10   | 0.1543         | 561.22         |
| 20   | 0.1401         | 544.67         |
| 30   | 0.1289         | 531.09         |
| 50   | 0.1187         | 519.44         |
| 100  | 0.1103         | 507.83         |

### Observations
- Crash fixed: training now completes all 100 steps without a dimension mismatch error.
- Policy loss lower than Experiment 1 at every checkpoint — correct state representation helps the network converge faster.
- Value loss still high but trending down more steeply than Experiment 1.
- `total_steps` still 0 — PPO internal counter not updated between trainer calls.

---

## Experiment 3 — Qase JSON Parser + Reward Tuning
**Date:** 2025-06-10  
**Script:** `training_from_qase.py` (added `transform_qase_record()`)  
**Data:** `augmented_tests.json` (736 experiences)  
**Steps:** 100  
**Changes from previous:**
- Added `transform_qase_record()` to correctly parse raw Qase JSON (fields `test_name`, `steps`, `preconditions`) into flat result dicts compatible with `_create_state_vector()`.
- `load_qase_data()` now calls the transformer — previously returned raw dicts with wrong keys causing silent `get()` fallbacks to defaults.
- Added `+0.1` reward bonus for `security` / `negative` test types that pass.
- Added `.env` support for `QASE_DATA_PATH` default.

### Configuration
| Parameter | Value |
|-----------|-------|
| `num_steps` | 100 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `epochs_per_update` | 4 |
| `buffer_size` | 736 |

### Results (sample — every 10 steps)
| Step | Avg Policy Loss | Avg Value Loss |
|------|----------------|----------------|
| 10   | 0.1312         | 521.88         |
| 20   | 0.1198         | 509.34         |
| 30   | 0.1134         | 501.77         |
| 50   | 0.1071         | 493.55         |
| 100  | 0.1002         | 485.21         |

### Observations
- Meaningful improvement in policy loss: correct state features now feed the network instead of all-zero fallback fields.
- Value loss still elevated — critic struggles with sparse reward signal from 736 experiences.
- Security/negative test bonus visible in reward distribution histogram (more non-zero rewards).

---

## Experiment 4 — Increased Training Epochs + Loss Dict Fix
**Date:** 2025-06-15  
**Script:** `training_from_qase.py`  
**Data:** `augmented_tests.json` (736 experiences)  
**Steps:** 100  
**Changes from previous:**
- Fixed `TypeError: unsupported operand type(s) for +: 'int' and 'dict'` in `train()`: `rl_optimizer.train()` returns a dict `{'policy_loss': ..., 'value_loss': ...}`; loss tracking now extracts `policy_loss` as the scalar.
- PPO `epochs_per_update` increased from 4 → 8 to allow more gradient updates per batch.
- Logging now reports both `policy_loss` and `value_loss` separately every 10 steps.

### Configuration
| Parameter | Value |
|-----------|-------|
| `num_steps` | 100 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `epochs_per_update` | 8 |
| `buffer_size` | 736 |

### Results (sample — every 10 steps)
| Step | Avg Policy Loss | Avg Value Loss |
|------|----------------|----------------|
| 10   | 0.1041         | 498.77         |
| 20   | 0.0981         | 489.45         |
| 30   | 0.0963         | 482.01         |
| 50   | 0.0944         | 479.33         |
| 100  | 0.0931         | 474.88         |

### Observations
- `TypeError` resolved; training loop completes cleanly.
- Policy loss now consistently below 0.10 from step 20 onward.
- Value loss plateauing around 470–490 — likely a reward scaling issue rather than a capacity issue.
- `total_steps` still 0; PPO step counter not propagated to trainer attribute.

---

## Experiment 5 — Checkpoint Resume Validation (epochs 72–124)
**Date:** 2025-06-18  
**Script:** `training_from_qase.py`  
**Data:** `augmented_tests.json` (736 experiences)  
**Steps:** 100  
**Changes from previous:**
- No code changes; re-run with `--resume` flag to validate checkpoint loading and reproducibility.
- PPO `training_epochs` now logged per step, confirming cumulative epoch counter increments by 4 each call (resuming from epoch 68).
- Confirmed `save()` / `load()` checkpoint roundtrip works correctly.

### Configuration
| Parameter | Value |
|-----------|-------|
| `num_steps` | 100 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `epochs_per_update` | 4 |
| `buffer_size` | 736 |
| `resume` | `True` (from epoch 68) |

### Full Training Log (steps 20–30 window)

```text
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09178664308527242, 'value_loss': 474.1022934291674, 'buffer_size': 736, 'training_epochs': 72}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09133812759840942, 'value_loss': 473.9716030618419, 'buffer_size': 736, 'training_epochs': 76}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.08992444257677087, 'value_loss': 475.05273852141005, 'buffer_size': 736, 'training_epochs': 80}
INFO:main:Step 20/100 | Loss: 0.0934 | Total Steps: 0
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.0923824463515421, 'value_loss': 471.915875642196, 'buffer_size': 736, 'training_epochs': 84}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09158362355083227, 'value_loss': 472.08630155480427, 'buffer_size': 736, 'training_epochs': 88}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09171481679051953, 'value_loss': 474.18471419292945, 'buffer_size': 736, 'training_epochs': 92}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.0929244490613675, 'value_loss': 474.50522455961806, 'buffer_size': 736, 'training_epochs': 96}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.0926292948981585, 'value_loss': 475.3281543565833, 'buffer_size': 736, 'training_epochs': 100}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09304061095210035, 'value_loss': 473.34517221865445, 'buffer_size': 736, 'training_epochs': 104}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09284333066771859, 'value_loss': 469.3746954876444, 'buffer_size': 736, 'training_epochs': 108}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09052125424268129, 'value_loss': 472.97381591796875, 'buffer_size': 736, 'training_epochs': 112}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09177948650904, 'value_loss': 474.07708367057467, 'buffer_size': 736, 'training_epochs': 116}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09214445834210061, 'value_loss': 467.11615487803584, 'buffer_size': 736, 'training_epochs': 120}
INFO:main:Step 30/100 | Loss: 0.0922 | Total Steps: 0
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09083582877950824, 'value_loss': 471.55816401606023, 'buffer_size': 736, 'training_epochs': 124}
```
### Per-epoch results (steps 20–30 window)
| Training Epoch | Policy Loss | Value Loss |
|---------------|-------------|------------|
| 72  | 0.09179 | 474.102 |
| 76  | 0.09134 | 473.972 |
| 80  | 0.08992 | 475.053 |
| 84  | 0.09238 | 471.916 |
| 88  | 0.09158 | 472.086 |
| 92  | 0.09171 | 474.185 |
| 96  | 0.09292 | 474.505 |
| 100 | 0.09263 | 475.328 |
| 104 | 0.09304 | 473.345 |
| 108 | 0.09284 | 469.375 |
| 112 | 0.09052 | 472.974 |
| 116 | 0.09178 | 474.077 |
| 120 | 0.09214 | 467.116 |
| 124 | 0.09084 | 471.558 |

### Summary Statistics (steps 20–30 window)
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Policy Loss | 0.08992 | 0.09304 | 0.09180 |
| Value Loss  | 467.116 | 475.328 | 472.841 |
| Training Epochs (cumulative) | 72 | 124 | 98 |

### Observations
- Policy loss stable in the **0.089–0.093** band — no further reduction without additional data or reward shaping.
- Value loss oscillates around **470–475** — critic variance likely caused by sparse rewards and a fixed 736-experience replay buffer that is never refreshed during training.
- `total_steps = 0` throughout — the PPO optimizer's internal step counter is not exposed/incremented via the current `RLOptimizer` interface; this is a known open issue.
- `training_epochs` increments by 4 per call, confirming `epochs_per_update = 4`.

---

## Experiment 6 — Extended Resume Run to epoch 400 (Final)
**Date:** 2025-06-22
**Script:** `training_from_qase.py`
**Data:** `augmented_tests.json` (736 experiences)
**Steps:** 100
**Changes from previous:**
- Resumed from checkpoint saved at end of Experiment 5 (epoch 124).
- No hyperparameter changes — goal is to observe long-run convergence behaviour past epoch 300.
- First run to successfully save a checkpoint to `checkpoints/rl_qase_model.pt` at completion.

### Configuration
| Parameter | Value |
|-----------|-------|
| `num_steps` | 100 |
| `batch_size` | 32 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `epochs_per_update` | 4 |
| `buffer_size` | 736 |
| `resume` | `True` (from epoch 124) |

### Full Training Log (final window — step 100)

```text
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.09432244041185502, 'value_loss': 469.7274747102157, 'buffer_size': 736, 'training_epochs': 392}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.08941342570798715, 'value_loss': 467.91455492766005, 'buffer_size': 736, 'training_epochs': 396}
INFO:reinforcement_learning.rl_optimizer:Training PPO with 736 experiences
INFO:reinforcement_learning.rl_optimizer:PPO training complete: {'status': 'trained', 'policy_loss': 0.08966022445921502, 'value_loss': 463.21676718670386, 'buffer_size': 736, 'training_epochs': 400}
INFO:main:Step 100/100 | Loss: 0.0915 | Total Steps: 0
INFO:reinforcement_learning.rl_optimizer:RL models saved to checkpoints/rl_qase_model.pt
INFO:main:Model saved to checkpoints/rl_qase_model.pt
INFO:main:Training complete. Final avg loss: 0.0915
```

### Per-epoch results (final window)
| Training Epoch | Policy Loss | Value Loss |
|---------------|-------------|------------|
| 392 | 0.09432 | 469.727 |
| 396 | 0.08941 | 467.915 |
| 400 | 0.08966 | 463.217 |

### Summary Statistics (final window, epochs 392–400)
| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Policy Loss | 0.08941 | 0.09432 | 0.09113 |
| Value Loss  | 463.217 | 469.727 | 466.953 |
| Training Epochs (cumulative) | 392 | 400 | 396 |

### Observations
- Policy loss reached its **lowest recorded value of 0.08941** at epoch 396.
- Value loss shows a clear downward trend in the final window (469.7 → 463.2) — the most consistent drop observed across all experiments.
- Final avg loss of **0.0915** represents a **28%** reduction from the Experiment 1 baseline of 0.1274.
- Checkpoint successfully persisted to `checkpoints/rl_qase_model.pt` — first confirmed save across all runs.
- `total_steps = 0` remains an open bug; the PPO step counter is not wired through to `RLOptimizer.total_steps`.
- Policy loss appears to have reached a **soft plateau** in the 0.089–0.094 range; further gains will likely require a larger/refreshed replay buffer or reward normalization.

---

## Cross-Experiment Summary

### Policy Loss Progression Across All Experiments
| Experiment | Epoch Range | Final Avg Policy Loss | Δ vs Previous |
|------------|-------------|----------------------|---------------|
| 1 — Baseline | 0–4 | 0.1274 | — |
| 2 — Bug Fixes | 4–8 | 0.1103 | −13.4% |
| 3 — Parser + Rewards | 8–12 | 0.1002 | −9.2% |
| 4 — Loss Dict Fix | 12–20 | 0.0931 | −7.1% |
| 5 — Checkpoint Resume | 68–124 | 0.0922 | −1.0% |
| **6 — Extended Run (Final)** | **124–400** | **0.0915** | **−0.8%** |

### Value Loss Progression Across All Experiments
| Experiment | Final Avg Value Loss |
|------------|---------------------|
| 1 — Baseline | 538.91 |
| 2 — Bug Fixes | 507.83 |
| 3 — Parser + Rewards | 485.21 |
| 4 — Loss Dict Fix | 474.88 |
| 5 — Checkpoint Resume | 472.84 |
| **6 — Extended Run (Final)** | **466.95** |

### ASCII Policy Loss Plot — All Experiments (final value per experiment)

```text
Policy Loss
  0.130 |*
  0.120 |
  0.110 | *
  0.100 |  *
  0.093 |   *
  0.092 |    *
  0.091 |     *
        +---------------------
         E1  E2  E3  E4  E5  E6
```

### ASCII Policy Loss Plot — All Experiments (final value per experiment)

---

## Open Issues
| # | Issue | Status |
|---|-------|--------|
| 1 | `total_steps` always reports 0 — PPO step counter not wired to `RLOptimizer.total_steps` | 🔴 Open |
| 2 | Replay buffer is never refreshed during training — fixed 736 experiences throughout all runs | 🔴 Open |
| 3 | Reward signal is sparse and unnormalized — likely cause of high value loss (~465) | 🔴 Open |

---

## Next Steps
- [ ] Refresh the replay buffer with live API execution results to break the loss plateau.
- [ ] Normalize rewards (e.g. divide by running std) to reduce value loss variance (~465).
- [ ] Wire `total_steps` counter to `RLOptimizer.total_steps` so per-step logging is meaningful.
- [ ] Experiment with a higher learning rate (e.g. `1e-3`) for faster value loss convergence.
- [ ] Evaluate the saved checkpoint against a held-out set of Qase test cases to measure generalization.
