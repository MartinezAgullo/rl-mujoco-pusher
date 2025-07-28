"""Handles the training loop for the agent."""


class Trainer:
    """Trainer class to run training episodes."""

    def __init__(self, env, agent, episodes: int, max_steps: int):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps = max_steps

    def run(self):
        """Run the training loop."""
        for episode in range(self.episodes):
            state = (
                self.env.reset()
            )  # Reset environment to start a new episode. State is what the agent can see
            done = False
            total_reward = 0

            print(f"Starting Episode {episode + 1}/{self.episodes}")
            print(f"Initial Observation: {state}")

            for step in range(self.max_steps):
                # Agent selects an action based on the current state
                action = self.agent.select_action(state)

                # Interact with the environment using the chosen action
                next_state, reward, done, truncated, info = self.env.step(action)

                # Accumulate the reward
                total_reward += reward

                # Update the current state
                state = next_state

                if done:
                    print(
                        "Episode ended early: environment signaled termination (done=True)."
                    )
                    break
                if truncated:
                    print("Episode reached time limit (truncated=True).")
                    break

            print(f"Episode {episode+1}/{self.episodes} - Total Reward: {total_reward}")
