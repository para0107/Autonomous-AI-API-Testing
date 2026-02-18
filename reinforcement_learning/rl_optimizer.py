"""
RL Optimizer using PPO - Fixed version

Fixes:
- Consistent async interface (all public methods are async)
- Uses compact 64-dim state vector instead of 640-dim (95% zeros)
- Fixed action selection: exact match instead of loose `in` substring matching
- train() is now async to match optimize()
"""

import logging
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

from reinforcement_learning.state_extractor import (
    extract_state, TOTAL_STATE_DIM
)

logger = logging.getLogger(__name__)

# Action types for test generation strategy
ACTION_TYPES = [
    "happy_path",
    "negative",
    "edge_case",
    "boundary",
    "security",
    "auth",
    "performance",
    "null_empty",
    "injection",
    "large_payload",
]

NUM_ACTIONS = len(ACTION_TYPES)


@dataclass
class Experience:
    """Single RL experience tuple"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


class PolicyNetwork(nn.Module):
    """Actor network - outputs action probabilities"""

    def __init__(self, state_dim: int = TOTAL_STATE_DIM, action_dim: int = NUM_ACTIONS):
        super().__init__()
        # Smaller network for smaller state space
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ValueNetwork(nn.Module):
    """Critic network - estimates state value"""

    def __init__(self, state_dim: int = TOTAL_STATE_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RLOptimizer:
    """PPO-based optimizer for test generation strategy"""

    def __init__(self, lr: float = 3e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, min_buffer_size: int = 100,
                 batch_size: int = 32):
        """
        Args:
            lr: Learning rate
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            min_buffer_size: Minimum experiences before training (lowered from 1000)
            batch_size: Training batch size
        """
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size

        # Networks
        self.policy = PolicyNetwork()
        self.value_net = ValueNetwork()
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Experience buffer
        self.buffer: deque = deque(maxlen=10000)
        self.episode_rewards: List[float] = []

        # Training stats
        self.total_steps = 0
        self.training_epochs = 0

        logger.info(
            f"RLOptimizer initialized: state_dim={TOTAL_STATE_DIM}, "
            f"actions={NUM_ACTIONS}, min_buffer={min_buffer_size}"
        )

    def create_state(self, test_cases: List[Dict], api_spec: Dict,
                     history: Optional[List[Dict]] = None,
                     context: Optional[Dict] = None) -> np.ndarray:
        """Create a state vector from current information."""
        return extract_state(test_cases, api_spec, history, context)

    async def optimize(self, state: np.ndarray,
                       test_cases: List[Dict]) -> List[Dict]:
        """
        Use the policy to select which test types to prioritize,
        then reorder/filter test cases accordingly.

        Args:
            state: Current state vector (64-dim)
            test_cases: Generated test cases

        Returns:
            Optimized/reordered test cases
        """
        if not test_cases:
            return test_cases

        try:
            # Get action probabilities from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_probs = self.policy(state_tensor).squeeze().numpy()

            # Rank action types by probability
            ranked_actions = np.argsort(-action_probs)
            logger.info(
                f"RL policy top actions: "
                f"{[ACTION_TYPES[a] for a in ranked_actions[:3]]} "
                f"(probs: {action_probs[ranked_actions[:3]].tolist()})"
            )

            # Reorder test cases: prioritize types the policy recommends
            prioritized = self._prioritize_tests(test_cases, ranked_actions, action_probs)
            return prioritized

        except Exception as e:
            logger.warning(f"RL optimization failed: {e}, returning original order")
            return test_cases

    def _prioritize_tests(self, test_cases: List[Dict],
                          ranked_actions: np.ndarray,
                          action_probs: np.ndarray) -> List[Dict]:
        """
        Reorder tests based on policy-recommended action types.

        FIX: Uses exact type matching instead of loose `in` substring matching.
        """
        scored_tests = []

        for test in test_cases:
            test_type = test.get('test_type', test.get('type', 'unknown')).lower().strip()
            score = 0.0

            # FIX: Exact match or well-defined prefix matching
            for action_idx in range(NUM_ACTIONS):
                action_type = ACTION_TYPES[action_idx]

                # Exact match
                if test_type == action_type:
                    score = action_probs[action_idx]
                    break

                # Known prefix mappings (explicit, not substring)
                if self._types_match(test_type, action_type):
                    score = action_probs[action_idx] * 0.9  # slight penalty for fuzzy
                    break

            # Priority boost
            priority = test.get('priority', 'medium').lower()
            if priority == 'high':
                score += 0.1
            elif priority == 'low':
                score -= 0.05

            scored_tests.append((score, test))

        # Sort by score descending
        scored_tests.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_tests]

    @staticmethod
    def _types_match(test_type: str, action_type: str) -> bool:
        """
        Check if a test type matches an action type with well-defined rules.
        No loose substring matching.
        """
        # Define explicit equivalences
        equivalences = {
            'happy_path': {'happy_path', 'smoke', 'positive', 'happy'},
            'negative': {'negative', 'error', 'failure', 'invalid'},
            'edge_case': {'edge_case', 'edge', 'corner_case'},
            'boundary': {'boundary', 'boundary_value', 'limit'},
            'security': {'security', 'xss', 'csrf', 'sqli'},
            'auth': {'auth', 'authentication', 'authorization', 'authz', 'authn'},
            'performance': {'performance', 'load', 'stress', 'perf'},
            'null_empty': {'null_empty', 'null', 'empty', 'missing_field'},
            'injection': {'injection', 'sql_injection', 'command_injection'},
            'large_payload': {'large_payload', 'overflow', 'payload_size'},
        }

        if action_type in equivalences:
            return test_type in equivalences[action_type]
        return False

    def add_experience(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.total_steps += 1

    def record_reward(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool = False):
        """Record a reward from test execution for training."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            probs = self.policy(state_tensor).squeeze()
            log_prob = torch.log(probs[action] + 1e-10).item()
            value = self.value_net(state_tensor).item()

        exp = Experience(
            state=state, action=action, reward=reward,
            next_state=next_state, done=done,
            log_prob=log_prob, value=value
        )
        self.add_experience(exp)

    async def train(self, epochs: int = 4) -> Dict[str, float]:
        """
        Train policy and value networks using PPO.

        FIX: Now async to be consistent with optimize().
        """
        if len(self.buffer) < self.min_buffer_size:
            logger.info(
                f"Buffer size {len(self.buffer)}/{self.min_buffer_size} — "
                f"need more experiences before training"
            )
            return {'status': 'insufficient_data', 'buffer_size': len(self.buffer)}

        logger.info(f"Training PPO with {len(self.buffer)} experiences")

        # Run the actual training in a thread to not block event loop
        metrics = await asyncio.to_thread(self._train_sync, epochs)
        return metrics

    def _train_sync(self, epochs: int) -> Dict[str, float]:
        """Synchronous training logic (called from thread)."""
        experiences = list(self.buffer)

        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        old_log_probs = torch.FloatTensor([e.log_prob for e in experiences])

        # Compute returns and advantages
        returns = self._compute_returns(rewards, [e.done for e in experiences])
        with torch.no_grad():
            values = self.value_net(states).squeeze()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0

        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(experiences))

            for start in range(0, len(indices), self.batch_size):
                end = min(start + self.batch_size, len(indices))
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # Policy loss (PPO clipped)
                probs = self.policy(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                policy_loss = policy_loss - 0.01 * entropy

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

                # Value loss
                values = self.value_net(batch_states).squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        self.training_epochs += epochs
        n_batches = max(1, (len(experiences) // self.batch_size) * epochs)

        metrics = {
            'status': 'trained',
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'buffer_size': len(self.buffer),
            'training_epochs': self.training_epochs,
        }
        logger.info(f"PPO training complete: {metrics}")
        return metrics

    def _compute_returns(self, rewards: torch.Tensor, dones: List[bool]) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        return returns

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Sample an action from the policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            probs = self.policy(state_tensor).squeeze()
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value_net.state_dict(),
            'training_epochs': self.training_epochs,
            'total_steps': self.total_steps,
        }, path)
        logger.info(f"RL models saved to {path}")

    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_net.load_state_dict(checkpoint['value'])
        self.training_epochs = checkpoint.get('training_epochs', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        logger.info(f"RL models loaded from {path}")