"""Defines the Agent class responsible for decision-making."""

import random


class Agent:
    """Simple agent with epsilon-greedy strategy."""

    def __init__(self, action_space, epsilon: float = 0.1):
        self.action_space = action_space
        self.epsilon = epsilon

    def select_action(self, state):
        """Select an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        # TODO: Replace with a learned policy
        return self.action_space.sample()
