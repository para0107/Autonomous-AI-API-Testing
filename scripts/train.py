"""
FIXED scripts/train.py — BUG 8 FIX
====================================
Problem: Original train.py calls non-existent methods:
    - self.optimizer.experience_buffer  → should be self.optimizer.buffer
    - self.optimizer.train() (sync)     → should be await self.optimizer.train() (async)
    - self.optimizer.save_checkpoint()  → should be self.optimizer.save()
    - self.optimizer.load_checkpoint()  → should be self.optimizer.load()
    - self.optimizer.training_step      → should be self.optimizer.total_steps
    - self.optimizer.exploration_rate   → does not exist on RLOptimizer

This is a complete rewrite matching the actual RLOptimizer interface.
"""

import asyncio
import logging
import argparse
from pathlib import Path

from reinforcement_learning import RLOptimizer, ExperienceBuffer, Experience
from reinforcement_learning.state_extractor import extract_state

logger = logging.getLogger(__name__)


class RLTrainer:
    """Training harness for the RL optimizer, matching actual RLOptimizer API."""

    def __init__(self, checkpoint_path: str = None):
        self.optimizer = RLOptimizer()
        self.checkpoint_path = checkpoint_path

    async def load_if_exists(self):
        """Load a previous checkpoint if path is provided and exists."""
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            self.optimizer.load(self.checkpoint_path)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            logger.info(f"Resuming from step {self.optimizer.total_steps}")
        else:
            logger.info("Starting fresh training run")

    async def add_experience(self, experience: Experience):
        """Add a single experience to the optimizer's buffer."""
        self.optimizer.buffer.append(experience)

    async def train_step(self):
        """
        Run a single training step.

        RLOptimizer.train() is async and handles:
        - Sampling from self.buffer (a deque)
        - Computing policy/value losses
        - Updating networks
        - Incrementing self.total_steps
        """
        if len(self.optimizer.buffer) < self.optimizer.batch_size:
            logger.warning(
                f"Buffer has {len(self.optimizer.buffer)} experiences, "
                f"need at least {self.optimizer.batch_size} for training"
            )
            return None

        loss = await self.optimizer.train()
        return loss

    async def train_epochs(self, num_epochs: int = 10, steps_per_epoch: int = 100):
        """Run multiple training epochs."""
        await self.load_if_exists()

        for epoch in range(num_epochs):
            epoch_losses = []

            for step in range(steps_per_epoch):
                loss = await self.train_step()
                if loss is not None:
                    epoch_losses.append(loss)

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Total Steps: {self.optimizer.total_steps} | "
                f"Buffer Size: {len(self.optimizer.buffer)}"
            )

            # Save checkpoint after each epoch
            if self.checkpoint_path:
                self.optimizer.save(self.checkpoint_path)
                logger.info(f"Saved checkpoint to {self.checkpoint_path}")

    async def save(self):
        """Save the current model state."""
        if self.checkpoint_path:
            self.optimizer.save(self.checkpoint_path)
            logger.info(f"Final checkpoint saved to {self.checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the RL optimizer")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/rl_model.pt",
        help="Path to save/load model checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=100, help="Training steps per epoch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Ensure checkpoint directory exists
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    trainer = RLTrainer(checkpoint_path=args.checkpoint)
    asyncio.run(trainer.train_epochs(num_epochs=args.epochs, steps_per_epoch=args.steps))


if __name__ == "__main__":
    main()