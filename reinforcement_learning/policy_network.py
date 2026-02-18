"""
Policy network for action selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from config import rl_config

class PolicyNetwork(nn.Module):
    """Neural network for policy (action selection)"""

    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()

        # Get hidden layer sizes from config
        hidden_sizes = rl_config.policy_hidden_sizes

        # Build layers
        layers = []
        prev_size = state_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            state: State tensor

        Returns:
            Action probabilities
        """
        logits = self.network(state)
        return F.softmax(logits, dim=-1)

    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action from policy

        Args:
            state: State tensor
            deterministic: If True, return most probable action

        Returns:
            Selected action
        """
        probs = self.forward(state)

        if deterministic:
            return torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            return dist.sample()

    def evaluate_actions(self, states: torch.Tensor,
                         actions: torch.Tensor) -> tuple:
        """
        Evaluate actions for PPO

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Log probabilities and entropy
        """
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy