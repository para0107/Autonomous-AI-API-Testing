"""
FIXED training_from_qase.py — BUG 7 FIX
=========================================
Problems in original:
    1. _create_state_vector() returns 640-dim, but RLOptimizer expects 64-dim
    2. References self.rl_optimizer.experience_buffer → should be self.rl_optimizer.buffer
    3. Calls self.rl_optimizer.train() synchronously → must be awaited (async)
    4. Uses save_checkpoint() / load_checkpoint() → should be save() / load()

This is a complete rewrite matching the actual RLOptimizer interface
and the 64-dim state from state_extractor.py.
"""

import asyncio
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from reinforcement_learning import RLOptimizer, ExperienceBuffer, Experience
from reinforcement_learning.state_extractor import extract_state
from reinforcement_learning.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)


class QaseTrainer:
    """
    Train the RL optimizer from Qase test management data.

    Fixed to match the actual RLOptimizer interface:
    - Uses 64-dim state vectors (via extract_state)
    - Accesses buffer via self.rl_optimizer.buffer (not experience_buffer)
    - Calls train() as async
    - Uses save()/load() for checkpoints
    """

    def __init__(self, checkpoint_path: str = "checkpoints/rl_qase_model.pt"):
        self.rl_optimizer = RLOptimizer()
        self.reward_calculator = RewardCalculator()
        self.checkpoint_path = checkpoint_path
        self.training_history: List[Dict[str, Any]] = []

    def load_qase_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load test case results from Qase export."""
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} test results from {data_path}")
        return data

    def _create_state_vector(self, test_result: Dict[str, Any]) -> np.ndarray:
        """
        Create a 64-dim state vector from a Qase test result.

        FIXED: Uses extract_state() which produces the correct 64-dim vector
        that matches RLOptimizer's TOTAL_STATE_DIM = 64.

        The old version manually constructed a 640-dim vector which caused
        dimension mismatch crashes.
        """
        # Build a test_case dict from the Qase result
        test_case = {
            'method': test_result.get('method', 'GET'),
            'endpoint': test_result.get('endpoint', '/'),
            'status_code': test_result.get('expected_status', 200),
            'description': test_result.get('title', ''),
            'parameters': test_result.get('parameters', {}),
        }

        # Build minimal api_spec from available info
        api_spec = {
            'endpoints': [{
                'method': test_case['method'],
                'path': test_case['endpoint'],
            }]
        }

        # Build execution history context
        history = self.training_history[-20:]  # Last 20 results for context

        # Use extract_state() to get the correct 64-dim vector
        state = extract_state([test_case], api_spec, history, None)
        return state

    def _compute_reward(self, test_result: Dict[str, Any]) -> float:
        """Compute reward from a Qase test execution result."""
        status = test_result.get('status', 'unknown').lower()

        if status in ('passed', 'success'):
            base_reward = 1.0
        elif status in ('failed', 'error'):
            base_reward = -0.5
        elif status in ('blocked', 'skipped'):
            base_reward = -0.2
        else:
            base_reward = 0.0

        # Bonus for finding real bugs (failed tests are valuable)
        if status == 'failed' and test_result.get('defect_found', False):
            base_reward = 0.8  # Bug-finding is rewarded

        # Penalty for flaky tests
        if test_result.get('is_flaky', False):
            base_reward *= 0.5

        return base_reward

    def _determine_action(self, test_result: Dict[str, Any]) -> int:
        """Map a Qase test type to an action index."""
        test_type = test_result.get('test_type', 'functional').lower()

        # Map to action indices matching ACTION_TYPES in rl_optimizer
        type_to_action = {
            'functional': 0,
            'edge_case': 1,
            'security': 2,
            'performance': 3,
            'boundary': 4,
            'negative': 5,
            'integration': 6,
        }
        return type_to_action.get(test_type, 0)

    async def process_qase_results(self, results: List[Dict[str, Any]]):
        """
        Convert Qase results into RL experiences and add to the optimizer's buffer.

        FIXED: Uses self.rl_optimizer.buffer (not experience_buffer)
        """
        for i, result in enumerate(results):
            state = self._create_state_vector(result)
            action = self._determine_action(result)
            reward = self._compute_reward(result)

            # Next state: use next result or terminal
            if i + 1 < len(results):
                next_state = self._create_state_vector(results[i + 1])
                done = False
            else:
                next_state = np.zeros_like(state)
                done = True

            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            # FIXED: Use buffer (deque), not experience_buffer
            self.rl_optimizer.buffer.append(experience)
            self.training_history.append(result)

        logger.info(
            f"Processed {len(results)} results. "
            f"Buffer size: {len(self.rl_optimizer.buffer)}"
        )

    async def train(self, num_steps: int = 100) -> List[float]:
        """
        Run training steps using buffered experiences.

        FIXED: Calls await self.rl_optimizer.train() (async, not sync)
        """
        losses = []

        if len(self.rl_optimizer.buffer) < self.rl_optimizer.batch_size:
            logger.warning(
                f"Not enough experiences ({len(self.rl_optimizer.buffer)}) "
                f"for training (need {self.rl_optimizer.batch_size}). "
                "Add more Qase data first."
            )
            return losses

        for step in range(num_steps):
            # FIXED: await async train()
            loss = await self.rl_optimizer.train()
            if loss is not None:
                losses.append(loss)

            if (step + 1) % 10 == 0:
                avg = sum(losses[-10:]) / min(10, len(losses[-10:]))
                logger.info(
                    f"Step {step + 1}/{num_steps} | "
                    f"Loss: {avg:.4f} | "
                    f"Total Steps: {self.rl_optimizer.total_steps}"
                )

        return losses

    def save(self):
        """Save checkpoint. FIXED: Uses save() not save_checkpoint()."""
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        self.rl_optimizer.save(self.checkpoint_path)
        logger.info(f"Model saved to {self.checkpoint_path}")

    def load(self):
        """Load checkpoint. FIXED: Uses load() not load_checkpoint()."""
        if Path(self.checkpoint_path).exists():
            self.rl_optimizer.load(self.checkpoint_path)
            logger.info(
                f"Model loaded from {self.checkpoint_path} "
                f"(step {self.rl_optimizer.total_steps})"
            )
        else:
            logger.info("No checkpoint found, starting fresh")


async def main():
    """Main training loop from Qase data."""
    import argparse

    parser = argparse.ArgumentParser(description="Train RL from Qase test results")
    parser.add_argument("--data", type=str, required=True, help="Path to Qase results JSON")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rl_qase_model.pt")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    trainer = QaseTrainer(checkpoint_path=args.checkpoint)

    if args.resume:
        trainer.load()

    # Load and process Qase data
    results = trainer.load_qase_data(args.data)
    await trainer.process_qase_results(results)

    # Train
    losses = await trainer.train(num_steps=args.steps)

    # Save
    trainer.save()

    if losses:
        logger.info(f"Training complete. Final avg loss: {sum(losses[-10:]) / min(10, len(losses)):.4f}")
    else:
        logger.info("Training complete (no loss data).")


if __name__ == "__main__":
    asyncio.run(main())