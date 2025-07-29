"""Adapter to interact with Gymnasium environments."""

import gymnasium as gym


class GymEnvironmentAdapter:
    """Wraps a Gymnasium environment with a simple interface."""

    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        """Reset the environment and return the initial observation."""
        state, info = self.env.reset()
        return state

    def step(self, action):
        """Perform an action and return next state, reward, done, truncated, info."""
        return self.env.step(action)

    def render(self):
        """Render the environment."""
        self.env.render()
