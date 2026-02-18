"""
Reinforcement Learning configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning optimization"""

    # Algorithm settings
    algorithm: str = "PPO"  # Proximal Policy Optimization
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    entropy_coefficient: float = 0.01  # Entropy bonus
    value_loss_coefficient: float = 0.5  # Value loss weight

    # Network architecture
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    value_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"

    # State space dimensions
    state_dimensions: Dict[str, int] = field(default_factory=lambda: {
        'api_complexity': 128,
        'test_coverage': 64,
        'historical_bugs': 256,
        'execution_results': 128,
        'parameter_space': 64
    })

    # Action space
    action_types: List[str] = field(default_factory=lambda: [
        'happy_path',
        'boundary_value',
        'null_empty',
        'type_mismatch',
        'format_violation',
        'business_logic',
        'security_test',
        'concurrent_access',
        'state_transition',
        'integration_test'
    ])

    # Learning rate schedule
    initial_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    lr_schedule: str = "cosine_annealing"  # "linear", "exponential", "cosine_annealing"
    warmup_steps: int = 1000
    total_steps: int = 100000
    restart_period: int = 5000  # For cosine annealing with restarts

    # Training settings
    batch_size: int = 64
    mini_batch_size: int = 32
    n_epochs: int = 10
    max_grad_norm: float = 0.5

    # Experience buffer
    buffer_size: int = 10000
    min_buffer_size: int = 1000  # Minimum before training
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4

    # Reward function weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'bug_found': 10.0,
        'code_coverage': 5.0,
        'edge_case_covered': 8.0,
        'unique_scenario': 6.0,
        'false_positive': -3.0,
        'redundant_test': -2.0,
        'test_failed': -1.0,
        'api_error': -5.0
    })

    # Exploration settings
    initial_exploration: float = 1.0
    min_exploration: float = 0.01
    exploration_decay: float = 0.995

    # Checkpointing
    checkpoint_frequency: int = 1000
    keep_checkpoints: int = 5
    save_best_model: bool = True

    def get_learning_rate(self, step: int) -> float:
        """Calculate learning rate based on schedule"""
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_learning_rate * (step / self.warmup_steps)

        if self.lr_schedule == "linear":
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.initial_learning_rate * (1 - progress) + self.min_learning_rate * progress

        elif self.lr_schedule == "exponential":
            decay_steps = step - self.warmup_steps
            return max(self.min_learning_rate,
                       self.initial_learning_rate * (0.99 ** (decay_steps / 1000)))

        elif self.lr_schedule == "cosine_annealing":
            import math
            progress = (step - self.warmup_steps) % self.restart_period
            return self.min_learning_rate + (self.initial_learning_rate - self.min_learning_rate) * \
                0.5 * (1 + math.cos(math.pi * progress / self.restart_period))

        return self.initial_learning_rate

    def calculate_reward(self, metrics: Dict[str, Any]) -> float:
        """Calculate total reward from metrics"""
        total_reward = 0.0
        for key, weight in self.reward_weights.items():
            if key in metrics:
                total_reward += metrics[key] * weight
        return total_reward


# Global instance
rl_config = RLConfig()