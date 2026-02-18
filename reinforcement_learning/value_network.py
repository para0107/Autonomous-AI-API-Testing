"""
Value network for state evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import rl_config

class ValueNetwork(nn.Module):
    """Neural network for value function (state evaluation)"""

    def __init__(self, state_dim: int):
        super(ValueNetwork, self).__init__()

        # Get hidden layer sizes from config
        hidden_sizes = rl_config.value_hidden_sizes

        # Build layers
        layers = []
        prev_size = state_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            state: State tensor

        Returns:
            State value
        """
        return self.network(state)

    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.01)

    def get_value(self, state: torch.Tensor) -> float:
        """
        Get value for a single state

        Args:
            state: State tensor

        Returns:
            State value as float
        """
        with torch.no_grad():
            value = self.forward(state)
        return value.item()