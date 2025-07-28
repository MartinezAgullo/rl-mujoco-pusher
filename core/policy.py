"""Defines policy abstractions and implementations."""

import random
from abc import ABC, abstractmethod


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def select_action(self, state):
        pass


class EpsilonGreedyPolicy(Policy):
    """Epsilon-greedy policy implementation."""

    def __init__(self, action_space, epsilon: float = 0.1):
        self.action_space = action_space
        self.epsilon = epsilon

    def select_action(self, state):
        """Return an action based on epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        return self.action_space.sample()  # Replace with actual Q-value logic
