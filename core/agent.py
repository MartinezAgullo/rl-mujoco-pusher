"""Defines the Agent class responsible for decision-making."""

import numpy as np

from core.policy import Policy


class Agent:
    """Agent that uses a policy to select actions."""

    def __init__(self, policy: Policy):
        self.policy = policy

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Delegate action selection to the policy."""
        return self.policy.select_action(state)
