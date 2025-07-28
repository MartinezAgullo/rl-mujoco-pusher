"""Adapter to interact with Gymnasium environments."""

import gymnasium as gym


class GymEnvironmentAdapter:
    """Wraps a Gymnasium environment with a simple interface."""

    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        # action_space: gymnasium object that defines the space of valid actions

    def reset(self):
        """Reset the environment and return the initial state."""
        state, info = self.env.reset()
        # reset returns:
        #       - state: initial state
        #       - info: additional metadata
        return state

    def step(self, action):
        """Perform an action and return the next state, reward, and status."""
        return self.env.step(action)

    def render(self):
        """Render the environment."""
        self.env.render()
