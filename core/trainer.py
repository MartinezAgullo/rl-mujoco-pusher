"""Handles the training loop for the agent."""


class Trainer:
    """Trainer class to run training episodes."""

    def __init__(self, env, agent, episodes: int, max_steps: int):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps = max_steps

    def run(self):
        """Run the training loop with basic policy updates."""

        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0

            print(f"Starting Episode {episode + 1}/{self.episodes}")
            print(f"Initial Observation: {state}")

            steps = 0
            for step in range(self.max_steps):
                steps += 1

                # Agent selects an action based on the current state
                action = self.agent.select_action(state, explore=True)

                # Interact with the environment using the chosen action
                next_state, reward, done, truncated, info = self.env.step(action)

                # Store experience
                self.agent.store_transition(state, action, reward, next_state)

                self.agent.train_step()

                # Accumulate the reward
                total_reward += reward

                # Update the current state
                state = next_state

                if done:
                    print(
                        f"Episode ended early after {steps} steps: environment signaled termination (done=True)."
                    )
                    break
                if truncated:
                    print(
                        f"Episode reached time limit (truncated=True) after {steps} steps:"
                    )
                    break

            print(f"Episode {episode+1}/{self.episodes} - Total Reward: {total_reward}")
