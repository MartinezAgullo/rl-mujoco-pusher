"""Use case: Train the agent using a Gym environment."""

from config.settings import ENV_NAME, EPISODES, MAX_STEPS
from core.agent import Agent
from core.policy import PolicyNetwork
from core.trainer import Trainer
from infra.gym_environment import GymEnvironmentAdapter


def train_agent():
    """Train an agent using a policy network on the specified environment."""
    env = GymEnvironmentAdapter(ENV_NAME)

    # Extract state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high
    action_low = env.action_space.low

    # Initialize policy network
    policy_network = PolicyNetwork(state_dim, action_dim, action_high, action_low)

    # Create agent with the policy network
    agent = Agent(policy=policy_network)

    # Create trainer and run training loop
    trainer = Trainer(env, agent, episodes=EPISODES, max_steps=MAX_STEPS)
    trainer.run()
