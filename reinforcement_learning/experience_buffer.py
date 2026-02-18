"""
Experience replay buffer for RL training
"""

import logging
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass
import torch

from config import rl_config

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple"""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    priority: float = 1.0


class ExperienceBuffer:
    """Experience replay buffer with prioritization support"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0

        # Prioritized replay parameters
        self.alpha = rl_config.priority_alpha
        self.beta = rl_config.priority_beta
        self.epsilon = 1e-6

        # Statistics
        self.total_added = 0
        self.total_sampled = 0

    def add(self, state: torch.Tensor, action: torch.Tensor,
            reward: float, next_state: torch.Tensor, done: bool):
        """Add experience to buffer"""
        # Calculate initial priority
        max_priority = max(self.priorities) if self.priorities else 1.0

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            priority=max_priority
        )

        self.buffer.append(experience)
        self.priorities.append(max_priority)

        self.total_added += 1

        if self.total_added % 1000 == 0:
            logger.info(f"Added {self.total_added} experiences to buffer")

    def sample(self, batch_size: int):
        """
        Sample batch of experiences.

        Returns:
            (experiences, indices) tuple. indices is None for uniform sampling.
        """
        if rl_config.prioritized_replay:
            return self._prioritized_sample(batch_size)
        else:
            return self._uniform_sample(batch_size), None

    def _uniform_sample(self, batch_size: int) -> List[Experience]:
        """Uniform random sampling"""
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        self.total_sampled += batch_size
        return batch

    def _prioritized_sample(self, batch_size: int):
        """
        Prioritized experience replay sampling.

        Returns:
            (experiences, indices) tuple so callers can update priorities via TD errors.
        """
        if len(self.buffer) == 0:
            return [], []

        batch_size = min(batch_size, len(self.buffer))

        priorities = np.asarray(self.priorities, dtype=np.float64) + self.epsilon
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probs
        )

        batch = [self.buffer[i] for i in indices]
        self.total_sampled += len(batch)
        return batch, indices.tolist()

    def update_priorities(self, indices: List[int], td_errors) -> None:
        """
        Update priorities for prioritized replay.

        Args:
            indices: Indices of sampled experiences.
            td_errors: TD-error per sampled experience (tensor, list or ndarray).
        """
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.asarray(td_errors, dtype=np.float64)

        new_priorities = np.abs(td_errors) + self.epsilon

        for idx, p in zip(indices, new_priorities):
            p = float(p if np.isfinite(p) and p > 0 else self.epsilon)
            # Update backing priority store
            self.priorities[idx] = p
            # Keep the Experience's priority in sync (optional)
            try:
                self.buffer[idx].priority = p
            except IndexError:
                pass

    def __len__(self) -> int:
        return len(self.buffer)