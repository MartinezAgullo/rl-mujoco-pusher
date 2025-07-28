"""Use case: Train the agent using a Gym environment."""

from config.settings import ENV_NAME, EPISODES, EPSILON, MAX_STEPS
from core.agent import Agent
from core.trainer import Trainer
from infra.gym_environment import GymEnvironmentAdapter


def train_agent():
    """Train an agent on the specified environment."""
    env = GymEnvironmentAdapter(ENV_NAME)
    agent = Agent(action_space=env.action_space, epsilon=EPSILON)
    trainer = Trainer(env, agent, episodes=EPISODES, max_steps=MAX_STEPS)
    trainer.run()
