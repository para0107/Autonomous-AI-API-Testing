"""
FIXED scripts/evaluate.py — BUG 9 FIX
=======================================
Problem: Original evaluate.py calls self.optimizer.load_checkpoint()
which doesn't exist. Should be self.optimizer.load().

This is a complete rewrite matching the actual RLOptimizer interface.
"""

import asyncio
import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from reinforcement_learning import RLOptimizer
from reinforcement_learning.state_extractor import extract_state

logger = logging.getLogger(__name__)


class RLEvaluator:
    """Evaluation harness for the trained RL optimizer."""

    def __init__(self, checkpoint_path: str):
        self.optimizer = RLOptimizer()
        self.checkpoint_path = checkpoint_path

    def load_model(self):
        """Load the trained model. Uses optimizer.load() (not load_checkpoint)."""
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Train the model first with scripts/train.py"
            )
        # FIX: Use load() instead of load_checkpoint()
        self.optimizer.load(self.checkpoint_path)
        logger.info(f"Loaded model from {self.checkpoint_path}")
        logger.info(f"Model trained for {self.optimizer.total_steps} steps")

    async def evaluate_test_cases(
            self, test_cases: List[Dict[str, Any]], api_spec: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the RL policy on a set of test cases.

        For each test case, extracts a 64-dim state vector and gets the
        policy's action selection and confidence.
        """
        self.load_model()
        api_spec = api_spec or {}
        results = []

        for i, test_case in enumerate(test_cases):
            # Extract 64-dim state (matches TOTAL_STATE_DIM in rl_optimizer)
            state = extract_state([test_case], api_spec, [], None)

            # Get action from the optimizer
            action = await self.optimizer.select_action(state)

            results.append({
                'test_case_index': i,
                'test_case': test_case,
                'selected_action': action,
            })

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print evaluation summary."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluation Summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total test cases evaluated: {len(results)}")
        logger.info(f"Model checkpoint: {self.checkpoint_path}")
        logger.info(f"Model total training steps: {self.optimizer.total_steps}")

        # Count action distribution
        action_counts = {}
        for r in results:
            action = r['selected_action']
            action_counts[action] = action_counts.get(action, 0) + 1

        logger.info(f"\nAction Distribution:")
        for action, count in sorted(action_counts.items()):
            logger.info(f"  Action {action}: {count} ({100 * count / len(results):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RL optimizer")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/rl_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-data", type=str, default=None,
        help="Path to test data JSON file"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    evaluator = RLEvaluator(checkpoint_path=args.checkpoint)

    # Load test data if provided
    test_cases = []
    if args.test_data and Path(args.test_data).exists():
        with open(args.test_data, 'r') as f:
            test_cases = json.load(f)
        logger.info(f"Loaded {len(test_cases)} test cases from {args.test_data}")
    else:
        logger.warning("No test data provided. Use --test-data <path>")
        return

    results = asyncio.run(evaluator.evaluate_test_cases(test_cases))
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()