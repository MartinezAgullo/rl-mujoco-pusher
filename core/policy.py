"""Defines policy abstractions and implementations for continuous control."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class Policy(ABC):
    """Abstract base class for all policies."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Return an action given the current state."""
        pass


class PolicyNetwork(Policy):
    """Neural network-based policy for continuous action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
    ):
        self.action_high = action_high
        self.action_low = action_low

        # Simple feedforward network
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Output between -1 and 1
        )
        # We end up with "action_dim" floats in the [-1, 1] range

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Compute an action using the policy network and scale it to the environment's range."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_dim)
        raw_action = self.model(state_tensor).detach().numpy()[0]  # Output in [-1, 1]

        # Scale action to environment's action space
        scaled_action = self._scale_action(raw_action)
        return scaled_action

    def _scale_action(self, raw_action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to [action_low, action_high]."""
        # In the case of Pusher-v5 this scale the results from [-1, 1] to [-2, 2]
        return self.action_low + (raw_action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )
