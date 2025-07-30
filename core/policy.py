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


class PolicyNetwork(nn.Module, Policy):
    """Neural network-based policy for continuous action spaces."""

    # Hybrid Inheritance: both Policy and nn.Module

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
    ):
        super().__init__()  # Initialize nn.Module
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.action_low = torch.tensor(action_low, dtype=torch.float32)

        # Simple feedforward policy network
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Compute an action for the given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_dim)
        scaled_action = self.forward(state_tensor).detach().numpy()[0]
        return scaled_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        raw_action = self.model(state)  # Output in [-1, 1]
        return self._scale_action(raw_action)  #  Output in [-2, 2]

    def _scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to [action_low, action_high]."""
        return self.action_low + (raw_action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )
