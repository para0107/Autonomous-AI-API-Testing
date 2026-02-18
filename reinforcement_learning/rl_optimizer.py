"""
PPO-based reinforcement learning optimizer
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from config import rl_config
from reinforcement_learning.policy_network import PolicyNetwork
from reinforcement_learning.value_network import ValueNetwork
from reinforcement_learning.experience_buffer import ExperienceBuffer
from reinforcement_learning.reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)
@dataclass
class State:
    """RL state representation"""
    api_features: np.ndarray
    test_coverage: np.ndarray
    historical_performance: np.ndarray
    current_test_set: np.ndarray

@dataclass
class Action:
    """RL action representation"""
    test_type: int
    parameter_selection: np.ndarray
    assertion_complexity: int

class RLOptimizer:
    """Reinforcement learning optimizer for test generation"""

    def __init__(self):
        logger.info("Initializing RL Optimizer")

        # Calculate state and action dimensions
        self.state_dim = sum(rl_config.state_dimensions.values())
        self.action_dim = len(rl_config.action_types)

        # Initialize networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_net = ValueNetwork(self.state_dim)

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=rl_config.initial_learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=rl_config.initial_learning_rate
        )

        # Initialize components
        self.experience_buffer = ExperienceBuffer(rl_config.buffer_size)
        self.reward_calculator = RewardCalculator()

        # Training state
        self.training_step = 0
        self.exploration_rate = rl_config.initial_exploration

    async def optimize(self, state: Dict[str, Any],
                      test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize test case selection and ordering

        Args:
            state: Current state representation
            test_cases: Generated test cases

        Returns:
            Optimized test cases
        """
        # Convert state to tensor
        state_tensor = self.create_state(test_cases, state)

        # Get action probabilities
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)

        # Select and reorder test cases
        optimized = self.select_test_cases(test_cases, action_probs)

        # Add exploration
        if np.random.random() < self.exploration_rate:
            optimized = self.add_exploration(optimized, test_cases)

        return optimized

    def create_state(self, test_cases: List[Dict[str, Any]],
                    api_spec: Dict[str, Any]) -> torch.Tensor:
        """Create state representation from test cases and API spec"""
        state_features = []

        # API complexity features
        api_features = self.extract_api_features(api_spec)
        state_features.append(api_features)

        # Test coverage features
        coverage_features = self.extract_coverage_features(test_cases)
        state_features.append(coverage_features)

        # Historical performance features
        history_features = self.extract_history_features(api_spec)
        state_features.append(history_features)

        # Current test set features
        test_features = self.extract_test_features(test_cases)
        state_features.append(test_features)

        # Combine features
        state_vector = np.concatenate(state_features)

        return torch.FloatTensor(state_vector).unsqueeze(0)

    def extract_api_features(self, api_spec: Dict[str, Any]) -> np.ndarray:
        """Extract features from API specification"""
        features = np.zeros(rl_config.state_dimensions['api_complexity'])

        # Number of parameters
        params = api_spec.get('parameters', [])
        features[0] = len(params) / 10.0  # Normalize

        # Parameter types distribution
        type_counts = {'string': 0, 'integer': 0, 'boolean': 0, 'object': 0, 'array': 0}
        for param in params:
            param_type = param.get('type', 'string')
            if param_type in type_counts:
                type_counts[param_type] += 1

        for i, count in enumerate(type_counts.values(), 1):
            if i < len(features):
                features[i] = count / max(len(params), 1)

        # HTTP method encoding
        method = api_spec.get('method', 'GET').upper()
        method_encoding = {'GET': 0.2, 'POST': 0.4, 'PUT': 0.6, 'DELETE': 0.8, 'PATCH': 1.0}
        features[6] = method_encoding.get(method, 0.5)

        # Authentication required
        features[7] = 1.0 if api_spec.get('security') else 0.0

        # Response codes
        responses = api_spec.get('responses', {})
        features[8] = len(responses) / 10.0

        # Validation rules
        validators = api_spec.get('x-test-metadata', {}).get('validators', [])
        features[9] = len(validators) / 10.0

        return features[:rl_config.state_dimensions['api_complexity']]

    def extract_coverage_features(self, test_cases: List[Dict[str, Any]]) -> np.ndarray:
        """Extract test coverage features"""
        features = np.zeros(rl_config.state_dimensions['test_coverage'])

        if not test_cases:
            return features

        # Test type distribution
        type_counts = {}
        for test in test_cases:
            test_type = test.get('test_type', 'unknown')
            type_counts[test_type] = type_counts.get(test_type, 0) + 1

        # Encode distribution
        for i, test_type in enumerate(rl_config.action_types[:10]):
            if i < len(features):
                features[i] = type_counts.get(test_type, 0) / len(test_cases)

        # Total test count
        features[10] = len(test_cases) / 100.0

        # Assertion coverage
        total_assertions = sum(len(t.get('assertions', [])) for t in test_cases)
        features[11] = total_assertions / (len(test_cases) * 5.0) if test_cases else 0

        return features[:rl_config.state_dimensions['test_coverage']]

    def extract_history_features(self, api_spec: Dict[str, Any]) -> np.ndarray:
        """Extract historical performance features from tracked reward history."""
        features = np.zeros(rl_config.state_dimensions['historical_bugs'])

        stats = self.reward_calculator.get_reward_statistics()
        if stats:
            features[0] = np.clip(stats.get('mean_reward', 0.0), -1, 1)
            features[1] = np.clip(stats.get('std_reward', 0.0), 0, 1)
            features[2] = np.clip(stats.get('max_reward', 0.0), -1, 1)
            features[3] = np.clip(stats.get('min_reward', 0.0), -1, 1)
            trend = stats.get('recent_trend', 'stable')
            features[4] = {'improving': 1.0, 'stable': 0.5, 'declining': 0.0}.get(trend, 0.5)
            features[5] = min(stats.get('total_episodes', 0) / 100.0, 1.0)

        return features[:rl_config.state_dimensions['historical_bugs']]

    def extract_test_features(self, test_cases: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from current test set"""
        features = np.zeros(rl_config.state_dimensions['execution_results'])

        if not test_cases:
            return features

        # Diversity score
        unique_types = len(set(t.get('test_type') for t in test_cases))
        features[0] = unique_types / len(rl_config.action_types)

        # Complexity score
        avg_assertions = np.mean([len(t.get('assertions', [])) for t in test_cases])
        features[1] = avg_assertions / 10

        # Priority score
        priority_scores = {'authentication': 1.0, 'validation': 0.8, 'happy_path': 0.6}
        avg_priority = np.mean([
            priority_scores.get(t.get('test_type'), 0.5) for t in test_cases
        ])
        features[2] = avg_priority

        return features[:rl_config.state_dimensions['execution_results']]

    def select_test_cases(self, test_cases: List[Dict[str, Any]],
                         action_probs: torch.Tensor) -> List[Dict[str, Any]]:
        """Select and order test cases based on action probabilities"""
        if not test_cases:
            return test_cases

        # Convert action probabilities to numpy
        probs = action_probs.squeeze().numpy()

        # Score each test case
        scored_tests = []
        for test in test_cases:
            test_type = test.get('test_type', 'unknown')

            # Get type index
            type_idx = 0
            for i, action_type in enumerate(rl_config.action_types):
                if action_type in test_type:
                    type_idx = i
                    break

            # Calculate score
            score = probs[type_idx] if type_idx < len(probs) else 0.5

            # Add priority bonus
            if test_type in ['authentication', 'validation']:
                score *= 1.5

            scored_tests.append((score, test))

        # Sort by score
        scored_tests.sort(key=lambda x: x[0], reverse=True)

        # Return sorted test cases
        return [test for _, test in scored_tests]

    def add_exploration(self, test_cases: List[Dict[str, Any]],
                       all_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add exploration by including random test cases"""
        if len(test_cases) >= len(all_tests):
            return test_cases

        # Add some random tests not in the current selection
        remaining = [t for t in all_tests if t not in test_cases]

        if remaining:
            # Add up to 20% random tests
            n_random = max(1, int(len(test_cases) * 0.2))
            random_tests = np.random.choice(remaining, min(n_random, len(remaining)),
                                          replace=False).tolist()
            test_cases.extend(random_tests)

        return test_cases

    def update_from_feedback(self, state: torch.Tensor, action: torch.Tensor,
                           reward: float, next_state: torch.Tensor, done: bool):
        """Update networks based on feedback"""
        # Store experience
        self.experience_buffer.add(state, action, reward, next_state, done)

        # Update if buffer is ready
        if len(self.experience_buffer) >= rl_config.min_buffer_size:
            self.train()

    def train(self):
        """Train networks using PPO"""
        # sample() returns (batch, indices); indices is None for uniform sampling
        batch, sample_indices = self.experience_buffer.sample(rl_config.batch_size)

        if not batch:
            return

        states = torch.stack([e.state for e in batch])
        actions = torch.stack([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])

        # Calculate advantages
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()

            # GAE calculation
            advantages = self.calculate_gae(rewards, values, next_values, dones)
            returns = advantages + values

        # Compute old log probs ONCE before the update loop (frozen policy snapshot)
        # Without this, ratio is always 1.0 and PPO clipping never activates
        with torch.no_grad():
            old_action_probs = self.policy_net(states)
            old_dist = torch.distributions.Categorical(old_action_probs)
            old_log_probs = old_dist.log_prob(actions.squeeze())

        # PPO update
        for _ in range(rl_config.n_epochs):
            # Get current policy
            action_probs = self.policy_net(states)
            dist = torch.distributions.Categorical(action_probs)

            # Ratio: current policy / frozen old policy
            current_log_probs = dist.log_prob(actions.squeeze())
            ratio = torch.exp(current_log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - rl_config.clip_epsilon,
                              1 + rl_config.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()
            policy_loss = policy_loss - rl_config.entropy_coefficient * entropy

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                          rl_config.max_grad_norm)
            self.policy_optimizer.step()

            # Update value network
            values_pred = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(values_pred, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(),
                                          rl_config.max_grad_norm)
            self.value_optimizer.step()

        # Update priorities using final TD errors (prioritized replay only)
        if sample_indices is not None:
            with torch.no_grad():
                final_values = self.value_net(states).squeeze()
                final_next_values = self.value_net(next_states).squeeze()
                td_errors = rewards + rl_config.gamma * final_next_values * (1 - dones) - final_values
            self.experience_buffer.update_priorities(sample_indices, td_errors)

        # Update learning rate
        self.update_learning_rate()

        # Update exploration rate
        self.exploration_rate = max(
            rl_config.min_exploration,
            self.exploration_rate * rl_config.exploration_decay
        )

        self.training_step += 1

    def calculate_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                      next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + rl_config.gamma * next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + rl_config.gamma * rl_config.gae_lambda * \
                           last_advantage * (1 - dones[t])
            advantages[t] = last_advantage

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def update_learning_rate(self):
        """Update learning rate based on schedule"""
        lr = rl_config.get_learning_rate(self.training_step)

        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'value_state': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'training_step': self.training_step,
            'exploration_rate': self.exploration_rate
        }, path)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)

        self.policy_net.load_state_dict(checkpoint['policy_state'])
        self.value_net.load_state_dict(checkpoint['value_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.training_step = checkpoint['training_step']
        self.exploration_rate = checkpoint['exploration_rate']

        logger.info(f"Loaded checkpoint from {path}")