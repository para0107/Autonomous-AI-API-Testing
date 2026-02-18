"""
Calculate rewards for reinforcement learning
"""

import logging
from typing import Dict, Any, List
import numpy as np

from config import rl_config
logger = logging.getLogger(__name__)


class RewardCalculator:
    """Calculates rewards based on test execution results"""

    def __init__(self):
        self.reward_weights = rl_config.reward_weights
        self.metrics_history = []

    def calculate_reward(self, test_results: List[Dict[str, Any]],
                         metrics: Dict[str, Any]) -> float:
        """
        Calculate total reward from test execution

        Args:
            test_results: Test execution results
            metrics: Performance metrics

        Returns:
            Total reward value
        """
        reward = 0.0

        # Bug discovery reward
        bugs_found = metrics.get('bugs_found', 0)
        reward += bugs_found * self.reward_weights['bug_found']

        # Code coverage reward
        coverage = metrics.get('code_coverage', 0)
        reward += coverage * self.reward_weights['code_coverage']

        # Edge case coverage reward
        edge_cases = metrics.get('edge_cases_covered', 0)
        reward += edge_cases * self.reward_weights['edge_case_covered']

        # Unique scenario reward
        unique_scenarios = self._count_unique_scenarios(test_results)
        reward += unique_scenarios * self.reward_weights['unique_scenario']

        # Penalties
        false_positives = metrics.get('false_positives', 0)
        reward += false_positives * self.reward_weights['false_positive']

        redundant_tests = self._count_redundant_tests(test_results)
        reward += redundant_tests * self.reward_weights['redundant_test']

        test_failures = metrics.get('test_failures', 0)
        reward += test_failures * self.reward_weights['test_failed']

        api_errors = metrics.get('api_errors', 0)
        reward += api_errors * self.reward_weights['api_error']

        # Normalize reward
        reward = self._normalize_reward(reward)

        # Store metrics for analysis
        self.metrics_history.append({
            'reward': reward,
            'metrics': metrics,
            'timestamp': np.datetime64('now')
        })

        return reward

    def calculate_intermediate_reward(self, state: Dict[str, Any],
                                      action: Dict[str, Any]) -> float:
        """Calculate intermediate reward before execution"""
        reward = 0.0

        # Reward for test diversity
        test_type = action.get('test_type')
        if test_type in ['edge_case', 'security', 'boundary']:
            reward += 2.0
        elif test_type in ['validation', 'authentication']:
            reward += 1.5
        elif test_type == 'happy_path':
            reward += 1.0

        # Reward for parameter coverage
        param_coverage = action.get('parameter_coverage', 0)
        reward += param_coverage * 3.0

        # Penalty for redundancy
        if self._is_redundant(state, action):
            reward -= 2.0

        return reward

    def _count_unique_scenarios(self, test_results: List[Dict[str, Any]]) -> int:
        """Count unique test scenarios"""
        scenarios = set()

        for result in test_results:
            # Create scenario signature
            scenario = (
                result.get('endpoint'),
                result.get('method'),
                result.get('test_type'),
                frozenset(result.get('parameters', {}).keys())
            )
            scenarios.add(scenario)

        return len(scenarios)

    def _count_redundant_tests(self, test_results: List[Dict[str, Any]]) -> int:
        """Count redundant test cases"""
        seen = set()
        redundant = 0

        for result in test_results:
            # Create test signature
            signature = self._create_test_signature(result)

            if signature in seen:
                redundant += 1
            else:
                seen.add(signature)

        return redundant

    def _create_test_signature(self, test_result: Dict[str, Any]) -> str:
        """Create unique signature for test"""
        return f"{test_result.get('endpoint')}_{test_result.get('method')}_" \
               f"{test_result.get('test_type')}_{test_result.get('input')}"

    def _is_redundant(self, state: Dict[str, Any], action: Dict[str, Any]) -> bool:
        """Check if action would create redundant test"""
        existing_tests = state.get('existing_tests', [])

        for test in existing_tests:
            if (test.get('test_type') == action.get('test_type') and
                    test.get('parameters') == action.get('parameters')):
                return True

        return False

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to reasonable range"""
        # Clip to prevent extreme values
        reward = np.clip(reward, -100, 100)

        # Apply sigmoid for smooth normalization
        return 2 / (1 + np.exp(-reward / 10)) - 1

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about rewards"""
        if not self.metrics_history:
            return {}

        rewards = [m['reward'] for m in self.metrics_history]

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'total_episodes': len(rewards),
            'recent_trend': self._calculate_trend(rewards[-10:])
        }

    def _calculate_trend(self, recent_rewards: List[float]) -> str:
        """Calculate reward trend"""
        if len(recent_rewards) < 2:
            return 'stable'

        # Simple linear regression
        x = np.arange(len(recent_rewards))
        slope = np.polyfit(x, recent_rewards, 1)[0]

        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'