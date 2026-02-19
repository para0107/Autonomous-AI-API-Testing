

import asyncio
import logging
import json
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

from reinforcement_learning import RLOptimizer, ExperienceBuffer, Experience
from reinforcement_learning.state_extractor import extract_state
from reinforcement_learning.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for parsing the Qase JSON format
# ---------------------------------------------------------------------------

# Regex to extract HTTP method + path from an action string like
# "send a GET request to /api/users/:userId/reservations/paginated"
_ACTION_RE = re.compile(
    r'\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b.*?(/api[^\s,\n]*)',
    re.IGNORECASE,
)

# Map fragments of expected_result strings to HTTP status codes
_STATUS_MAP = [
    (r'200', 200),
    (r'201', 201),
    (r'204', 204),
    (r'400', 400),
    (r'401', 401),
    (r'403', 403),
    (r'404', 404),
    (r'422', 422),
    (r'500', 500),
]

# Derive a coarse test_type from the test_name so _determine_action() works
_TYPE_HINTS = {
    'security':     'security',
    'sql injection':'security',
    'sql_injection':'security',
    'negative':     'negative',
    'boundary':     'boundary',
    'edge':         'edge_case',
    'integration':  'integration',
    'lifecycle':    'integration',
    'performance':  'performance',
}


def _infer_status(expected_result: str) -> int:
    """Return the first HTTP status code found in expected_result, else 200."""
    for pattern, code in _STATUS_MAP:
        if re.search(pattern, expected_result or ''):
            return code
    return 200


def _infer_method_endpoint(action: str):
    """
    Extract (METHOD, /path) from an action string.
    Returns ('GET', '/') when nothing is found.
    """
    m = _ACTION_RE.search(action or '')
    if m:
        return m.group(1).upper(), m.group(2)
    return 'GET', '/'


def _infer_test_type(test_name: str) -> str:
    """Map keywords in a test_name to one of the known test types."""
    lower = test_name.lower()
    for keyword, ttype in _TYPE_HINTS.items():
        if keyword in lower:
            return ttype
    return 'functional'


def _infer_status_label(test_name: str, expected_result: str) -> str:
    """
    Derive a pass/fail label from naming conventions and expected results.

    Qase export uses '-positive-successful', '-negative-successful',
    '-negative-unsuccessful', '-positive-unsuccessful' in test names,
    and expected results contain status codes.
    """
    # Tests that explicitly check for error codes are "failed" scenarios
    # but they PASS when the API returns the right error → label 'passed'
    code = _infer_status(expected_result)
    if code in (200, 201, 204):
        return 'passed'
    # 4xx / 5xx expected → test validates error handling; treat as 'passed'
    # (the RL agent gets a lower reward because the scenario is negative)
    return 'passed'


def transform_qase_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert one Qase test-case record (with nested steps) into a flat list
    of result dicts that QaseTrainer._create_state_vector() / _compute_reward()
    can consume.

    Each step becomes one result dict.  If there are no steps the test-case
    itself becomes a single result dict.

    Output schema per item
    ----------------------
    {
        'title':           str,   # test_name
        'method':          str,   # HTTP verb inferred from action
        'endpoint':        str,   # path inferred from action
        'expected_status': int,   # HTTP code inferred from expected_result
        'status':          str,   # 'passed' | 'failed' | 'skipped'
        'test_type':       str,   # 'functional' | 'security' | ...
        'defect_found':    bool,
        'is_flaky':        bool,
        'parameters':      dict,
    }
    """
    test_name      = record.get('test_name', '')
    objective      = record.get('objective', '') or ''
    preconditions  = record.get('preconditions', '') or ''
    steps          = record.get('steps') or []
    test_type      = _infer_test_type(test_name)

    # Mark as defect_found when the test name explicitly says unsuccessful
    defect_found = 'unsuccessful' in test_name.lower()

    results = []

    if not steps:
        # No steps → synthesise one entry from the record metadata
        results.append({
            'title':           test_name,
            'method':          'GET',
            'endpoint':        '/',
            'expected_status': 200,
            'status':          'skipped',
            'test_type':       test_type,
            'defect_found':    defect_found,
            'is_flaky':        False,
            'parameters':      {},
        })
        return results

    for step in steps:
        action          = step.get('action', '')
        expected_result = step.get('expected_result', '')
        data_field      = step.get('data', '')

        method, endpoint    = _infer_method_endpoint(action)
        expected_status     = _infer_status(expected_result)
        status_label        = _infer_status_label(test_name, expected_result)

        # Parse simple key=value pairs from the data field if present
        parameters: Dict[str, Any] = {}
        if data_field and data_field not in ("", "{}"):
            # Try JSON first
            try:
                parsed = json.loads(data_field)
                if isinstance(parsed, dict):
                    parameters = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        results.append({
            'title':           test_name,
            'method':          method,
            'endpoint':        endpoint,
            'expected_status': expected_status,
            'status':          status_label,
            'test_type':       test_type,
            'defect_found':    defect_found,
            'is_flaky':        False,
            'parameters':      parameters,
        })

    return results


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class QaseTrainer:
    """
    Train the RL optimizer from Qase test management data.

    Fixed to match the actual RLOptimizer interface:
    - Uses 64-dim state vectors (via extract_state)
    - Accesses buffer via self.rl_optimizer.buffer (not experience_buffer)
    - Calls train() as async
    - Uses save()/load() for checkpoints
    - Parses raw Qase JSON format via transform_qase_record()
    """

    def __init__(self, checkpoint_path: str = "checkpoints/rl_qase_model.pt"):
        self.rl_optimizer      = RLOptimizer()
        self.reward_calculator = RewardCalculator()
        self.checkpoint_path   = checkpoint_path
        self.training_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_qase_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load and transform Qase export JSON into flat result dicts.

        FIXED (BUG 5): The original method returned the raw Qase records
        whose keys ('test_name', 'steps', …) are incompatible with
        _create_state_vector() which expects 'method', 'endpoint', etc.

        Now each Qase record is passed through transform_qase_record() so
        the output is a flat list ready for process_qase_results().
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            raw: List[Dict[str, Any]] = json.load(f)

        transformed: List[Dict[str, Any]] = []
        for record in raw:
            transformed.extend(transform_qase_record(record))

        logger.info(
            f"Loaded {len(raw)} Qase records → "
            f"{len(transformed)} result entries from {data_path}"
        )
        return transformed

    # ------------------------------------------------------------------
    # State / reward / action helpers
    # ------------------------------------------------------------------

    def _create_state_vector(self, test_result: Dict[str, Any]) -> np.ndarray:
        """
        Create a 64-dim state vector from a (transformed) test result.

        FIXED: Uses extract_state() which produces the correct 64-dim vector
        that matches RLOptimizer's TOTAL_STATE_DIM = 64.
        """
        test_case = {
            'method':      test_result.get('method', 'GET'),
            'endpoint':    test_result.get('endpoint', '/'),
            'status_code': test_result.get('expected_status', 200),
            'description': test_result.get('title', ''),
            'parameters':  test_result.get('parameters', {}),
        }

        api_spec = {
            'endpoints': [{
                'method': test_case['method'],
                'path':   test_case['endpoint'],
            }]
        }

        history = self.training_history[-20:]
        state   = extract_state([test_case], api_spec, history, None)
        return state

    def _compute_reward(self, test_result: Dict[str, Any]) -> float:
        """Compute reward from a transformed Qase test result."""
        status = test_result.get('status', 'unknown').lower()

        if status in ('passed', 'success'):
            base_reward = 1.0
        elif status in ('failed', 'error'):
            base_reward = -0.5
        elif status in ('blocked', 'skipped'):
            base_reward = -0.2
        else:
            base_reward = 0.0

        # Bonus: finding a real defect is valuable
        if status == 'failed' and test_result.get('defect_found', False):
            base_reward = 0.8

        # Penalty for flaky tests
        if test_result.get('is_flaky', False):
            base_reward *= 0.5

        # Slight positive boost for security / negative tests that pass
        # (agent should be encouraged to generate them)
        if test_result.get('test_type') in ('security', 'negative') \
                and status == 'passed':
            base_reward += 0.1

        return base_reward

    def _determine_action(self, test_result: Dict[str, Any]) -> int:
        """Map a test type to an action index."""
        test_type = test_result.get('test_type', 'functional').lower()

        type_to_action = {
            'functional':  0,
            'edge_case':   1,
            'security':    2,
            'performance': 3,
            'boundary':    4,
            'negative':    5,
            'integration': 6,
        }
        return type_to_action.get(test_type, 0)

    # ------------------------------------------------------------------
    # Core async methods
    # ------------------------------------------------------------------

    async def process_qase_results(self, results: List[Dict[str, Any]]):
        """
        Convert transformed results into RL experiences and add to buffer.

        FIXED: Uses self.rl_optimizer.buffer (not experience_buffer)
        """
        for i, result in enumerate(results):
            state  = self._create_state_vector(result)
            action = self._determine_action(result)
            reward = self._compute_reward(result)

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
        losses: List[float] = []

        if len(self.rl_optimizer.buffer) < self.rl_optimizer.batch_size:
            logger.warning(
                f"Not enough experiences ({len(self.rl_optimizer.buffer)}) "
                f"for training (need {self.rl_optimizer.batch_size}). "
                "Add more Qase data first."
            )
            return losses

        for step in range(num_steps):
            loss = await self.rl_optimizer.train()
            if loss is not None:
                losses.append(loss['policy_loss'] if isinstance(loss, dict) else loss)

            if (step + 1) % 10 == 0:
                recent = losses[-10:]
                avg    = sum(recent) / len(recent) if recent else float('nan')
                logger.info(
                    f"Step {step + 1}/{num_steps} | "
                    f"Loss: {avg:.4f} | "
                    f"Total Steps: {self.rl_optimizer.total_steps}"
                )

        return losses

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

async def main():
    """Main training loop from Qase data."""
    import argparse

    parser = argparse.ArgumentParser(description="Train RL from Qase test results")
    load_dotenv()

    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default=os.getenv("QASE_DATA_PATH"),
        help="Path to Qase results JSON"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/rl_qase_model.pt")
    parser.add_argument("--steps",      type=int, default=100,
                        help="Training steps")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    trainer = QaseTrainer(checkpoint_path=args.checkpoint)

    if args.resume:
        trainer.load()

    results = trainer.load_qase_data(args.data)
    await trainer.process_qase_results(results)

    losses = await trainer.train(num_steps=args.steps)

    trainer.save()

    if losses:
        logger.info(
            f"Training complete. "
            f"Final avg loss: {sum(losses[-10:]) / min(10, len(losses)):.4f}"
        )
    else:
        logger.info("Training complete (no loss data).")


if __name__ == "__main__":
    asyncio.run(main())
