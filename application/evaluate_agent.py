"""Use case: Evaluate a trained agent."""

from config.settings import ENV_NAME
from core.agent import Agent
from infra.gym_environment import GymEnvironmentAdapter


def evaluate_agent():
    """Run evaluation episodes using a trained agent."""
    env = GymEnvironmentAdapter(ENV_NAME)
    agent = Agent(action_space=env.action_space, epsilon=0.0)  # Greedy policy
    # TODO: Load trained parameters here if implemented
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
        env.render()
