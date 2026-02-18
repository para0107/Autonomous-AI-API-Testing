# file: scripts/train.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from config import rl_config
from reinforcement_learning.experience_buffer import Experience
from reinforcement_learning.rl_optimizer import RLOptimizer


class RLTrainer:
    """
    Trainer that wraps RLOptimizer.
    RLOptimizer owns its own networks, buffer, and optimizers internally —
    do not pass them in as constructor arguments.
    """

    def __init__(
        self,
        save_dir: Optional[Path] = None,
    ) -> None:
        # RLOptimizer.__init__() takes no arguments — it reads everything from rl_config
        self.optimizer = RLOptimizer()

        base_dir = Path(getattr(rl_config, "training_dir", "data/training"))
        self.save_dir = Path(save_dir) if save_dir else base_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Push a single experience into the optimizer's buffer."""
        self.optimizer.experience_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Dict[str, Any]:
        """
        Run one PPO training step if the buffer has enough data.
        Returns a dict of training stats, or {'skipped': True} if buffer not ready.
        """
        min_size = int(getattr(rl_config, "min_buffer_size", 32))
        if len(self.optimizer.experience_buffer) < min_size:
            return {"skipped": True}

        step_before = self.optimizer.training_step
        self.optimizer.train()
        trained = self.optimizer.training_step > step_before

        return {
            "skipped": not trained,
            "training_step": self.optimizer.training_step,
            "exploration_rate": self.optimizer.exploration_rate,
        }

    def fit(self, steps: int) -> None:
        """Run multiple train steps with lightweight logging."""
        steps = int(steps)
        log_interval = int(getattr(rl_config, "log_interval", 50))
        t0 = time.time()

        for i in range(1, steps + 1):
            stats = self.train_step()
            if i % log_interval == 0 and not stats.get("skipped"):
                print(
                    f"[step={i}] "
                    f"training_step={stats['training_step']} "
                    f"exploration_rate={stats['exploration_rate']:.4f}"
                )

        dt = time.time() - t0
        print(f"Training finished in {dt:.2f}s ({steps} steps)")

    def save_checkpoint(self, name: str = "checkpoint.pt") -> Path:
        """Save model checkpoint via RLOptimizer."""
        path = self.save_dir / name
        self.optimizer.save_checkpoint(str(path))
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint via RLOptimizer."""
        self.optimizer.load_checkpoint(str(path))