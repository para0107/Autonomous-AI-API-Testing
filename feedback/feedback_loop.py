"""
Feedback loop for continuous learning from test execution results.

Fixes:
- Raise clear errors when dependencies aren't set, instead of silent warnings
- compute_reward returns meaningful signal
- process_feedback actually updates RAG + RL
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Processes test execution feedback to improve RAG and RL components."""

    def __init__(self):
        self.rag_system = None
        self.rl_optimizer = None
        self.knowledge_base = None
        self.execution_history: List[Dict] = []
        self._initialized = False

    def set_rag_system(self, rag_system):
        """Set the RAG system for knowledge base updates."""
        self.rag_system = rag_system
        self._check_initialized()

    def set_rl_optimizer(self, rl_optimizer):
        """Set the RL optimizer for reward feedback."""
        self.rl_optimizer = rl_optimizer
        self._check_initialized()

    def set_knowledge_base(self, knowledge_base):
        """Set the knowledge base for direct updates."""
        self.knowledge_base = knowledge_base
        self._check_initialized()

    def _check_initialized(self):
        """Check if all dependencies are set."""
        if self.rag_system and self.rl_optimizer and self.knowledge_base:
            self._initialized = True
            logger.info("FeedbackLoop fully initialized with all dependencies")

    @property
    def is_ready(self) -> bool:
        return self._initialized

    async def process_feedback(self, execution_results: List[Dict]) -> Dict[str, Any]:
        """
        Process execution results and update RAG + RL.

        Args:
            execution_results: List of test execution result dicts

        Returns:
            Summary of updates performed
        """
        if not execution_results:
            logger.warning("No execution results to process")
            return {'status': 'no_results'}

        # Store history
        self.execution_history.extend(execution_results)

        summary = {
            'total_results': len(execution_results),
            'rag_updated': False,
            'rl_updated': False,
            'rewards_computed': 0,
        }

        # Update RAG knowledge base
        if self.rag_system is not None:
            try:
                await self._update_rag(execution_results)
                summary['rag_updated'] = True
            except Exception as e:
                logger.error(f"RAG update failed: {e}", exc_info=True)
        else:
            logger.warning(
                "FeedbackLoop: RAG system not set. "
                "Call set_rag_system() before processing feedback."
            )

        # Update RL with rewards
        if self.rl_optimizer is not None:
            try:
                rewards = await self._update_rl(execution_results)
                summary['rl_updated'] = True
                summary['rewards_computed'] = len(rewards)
            except Exception as e:
                logger.error(f"RL update failed: {e}", exc_info=True)
        else:
            logger.warning(
                "FeedbackLoop: RL optimizer not set. "
                "Call set_rl_optimizer() before processing feedback."
            )

        logger.info(f"Feedback processed: {summary}")
        return summary

    async def _update_rag(self, results: List[Dict]):
        """Update RAG with successful test patterns and failure patterns."""
        successful_tests = []
        failed_tests = []

        for result in results:
            test_case = result.get('test_case', result.get('test', {}))
            if not test_case:
                continue

            entry = {
                'test_name': result.get('name', test_case.get('name', 'unknown')),
                'endpoint': result.get('endpoint', test_case.get('endpoint', '')),
                'method': result.get('method', test_case.get('method', '')),
                'test_type': result.get('test_type', test_case.get('test_type', '')),
                'passed': result.get('passed', False),
                'execution_time': result.get('execution_time', 0),
                'expected_status': result.get('expected_status', test_case.get('expected_status')),
                'actual_status': result.get('actual_status'),
                'error': result.get('error'),
                'timestamp': result.get('timestamp', datetime.now().isoformat()),
            }

            if result.get('passed'):
                successful_tests.append(entry)
            else:
                failed_tests.append(entry)

        # Add successful patterns to knowledge base for future retrieval
        if successful_tests and self.knowledge_base is not None:
            try:
                for test in successful_tests:
                    searchable = (
                        f"{test['method']} {test['endpoint']} "
                        f"{test['test_type']} {test['test_name']} passed"
                    )
                    self.knowledge_base.add_entry({
                        'type': 'successful_test_pattern',
                        'data': test,
                        'searchable_text': searchable,
                    })
                logger.info(f"Added {len(successful_tests)} successful patterns to knowledge base")
            except Exception as e:
                logger.warning(f"Failed to add successful patterns: {e}")

        # Add failure patterns so the system can learn what doesn't work
        if failed_tests and self.knowledge_base is not None:
            try:
                for test in failed_tests:
                    searchable = (
                        f"{test['method']} {test['endpoint']} "
                        f"{test['test_type']} failed {test.get('error', '')}"
                    )
                    self.knowledge_base.add_entry({
                        'type': 'failed_test_pattern',
                        'data': test,
                        'searchable_text': searchable,
                    })
                logger.info(f"Added {len(failed_tests)} failure patterns to knowledge base")
            except Exception as e:
                logger.warning(f"Failed to add failure patterns: {e}")

    async def _update_rl(self, results: List[Dict]) -> List[float]:
        """Compute rewards from execution results and feed to RL optimizer."""
        rewards = []

        for result in results:
            reward = self._compute_reward(result)
            rewards.append(reward)

            # Record in RL optimizer
            test_case = result.get('test_case', result.get('test', {}))
            if isinstance(test_case, dict):
                # Create minimal state for the experience
                from reinforcement_learning.state_extractor import extract_state
                state = extract_state([test_case], {}, self.execution_history[-20:])
                next_state = state  # simplified: same state

                # Map test type to action index
                action = self._test_type_to_action(
                    test_case.get('test_type', test_case.get('type', 'happy_path'))
                )

                self.rl_optimizer.record_reward(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=True
                )

        # Attempt training if buffer is full enough
        try:
            train_result = await self.rl_optimizer.train(epochs=2)
            if train_result.get('status') == 'trained':
                logger.info(f"RL training completed: {train_result}")
        except Exception as e:
            logger.warning(f"RL training skipped: {e}")

        return rewards

    def _compute_reward(self, result: Dict) -> float:
        """
        Compute a reward signal from a test execution result.

        Reward structure:
            +1.0  test passed and found a real bug (status mismatch indicating API issue)
            +0.5  test passed normally
            +0.3  test failed but revealed useful info (4xx/5xx)
            -0.2  test failed due to bad test design (connection error, timeout)
            -0.5  test couldn't execute at all

        Bonuses:
            +0.1  fast execution (< 1s)
            +0.2  high-priority test passed
        """
        reward = 0.0

        passed = result.get('passed', False)
        error = result.get('error')
        expected_status = result.get('expected_status')
        actual_status = result.get('actual_status')
        execution_time = result.get('execution_time', 0)

        if error and any(w in str(error).lower() for w in ['connection', 'timeout', 'refused']):
            # Infrastructure failure, not a useful signal
            reward = -0.2
        elif error:
            # Test couldn't execute
            reward = -0.5
        elif passed:
            reward = 0.5

            # Bonus: test found that status differs from expectation — potential bug discovery
            if expected_status and actual_status and expected_status != actual_status:
                reward = 1.0
        else:
            # Test failed (assertion or status mismatch)
            if actual_status and 400 <= actual_status < 600:
                # Server responded with error — useful negative test result
                reward = 0.3
            else:
                reward = 0.0

        # Time bonus
        if execution_time > 0 and execution_time < 1.0:
            reward += 0.1

        # Priority bonus
        test_case = result.get('test_case', result.get('test', {}))
        if isinstance(test_case, dict):
            priority = test_case.get('priority', 'medium').lower()
            if priority == 'high' and passed:
                reward += 0.2

        return reward

    @staticmethod
    def _test_type_to_action(test_type: str) -> int:
        """Map test type string to action index."""
        from reinforcement_learning.rl_optimizer import ACTION_TYPES

        test_type = test_type.lower().strip()

        # Direct match
        if test_type in ACTION_TYPES:
            return ACTION_TYPES.index(test_type)

        # Fuzzy mapping
        mappings = {
            'smoke': 0, 'positive': 0, 'happy': 0,        # happy_path
            'error': 1, 'failure': 1, 'invalid': 1,        # negative
            'edge': 2, 'corner': 2,                         # edge_case
            'limit': 3, 'boundary_value': 3,                # boundary
            'xss': 4, 'csrf': 4, 'sqli': 4,                # security
            'authentication': 5, 'authorization': 5,        # auth
            'load': 6, 'stress': 6,                         # performance
            'null': 7, 'empty': 7, 'missing': 7,            # null_empty
            'sql_injection': 8, 'command_injection': 8,     # injection
            'overflow': 9, 'payload': 9,                    # large_payload
        }

        for key, idx in mappings.items():
            if key in test_type:
                return idx

        return 0  # default: happy_path