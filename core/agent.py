"""Agent that uses a hybrid Policy for action selection and training."""

from typing import Tuple

import numpy as np
import torch

from core.policy import PolicyNetwork


class Agent:
    """Agent for continuous control using a policy."""

    def __init__(self, policy: PolicyNetwork, lr: float = 1e-3, noise_std: float = 0.1):
        self.policy = policy
        self.noise_std = noise_std
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory: list[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select an action from the policy and optionally add Gaussian noise."""
        action = self.policy.select_action(state)
        if explore:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(
                action + noise,
                self.policy.action_low.numpy(),
                self.policy.action_high.numpy(),
            )
        return action

    def store_transition(self, state, action, reward, next_state):
        """Store experience for training."""
        self.memory.append((state, action, reward, next_state))

    def train_step(self):
        """Perform a single policy update (placeholder)."""

        # If there are less than 10 transitions stored, do nothing
        # To avoid training where there is not enough data.
        if len(self.memory) < 10:
            return

        # Sample last 10 transitions:
        batch = self.memory[-10:]  # It could be better to do random sampling
        states, actions, rewards, next_states = zip(*batch)

        # Convert states and rewards to tensors:
        states = torch.FloatTensor(states)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)

        # Forward pass through the policy
        # predicted_actions = self.policy(states)

        # Placeholder loss: encourage positive rewards
        loss = -rewards_tensor.mean()  # Very naive, replace later

        # Backpropagation and optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
