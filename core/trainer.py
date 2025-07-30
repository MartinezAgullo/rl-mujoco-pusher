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

        print("[trainer.py - run()] Start training")
        for episode in range(self.episodes):
            """An episode represents a full attempt by the agent to complete the tasku"""
            state = self.env.reset()
            done = False
            total_reward = 0.0

            print(
                f"[trainer.py - run()]   \nStarting Episode {episode + 1}/{self.episodes}"
            )
            print(f"[trainer.py - run()]   Initial Observation: {state}")

            steps = 0
            for step in range(self.max_steps):
                steps += 1
                print(f"[trainer.py - run()]   - Step {steps}:")

                # Agent selects an action based on the current state
                action = self.agent.select_action(state, explore=True)
                print(f"[trainer.py - run()]   - Decided action {action}:")

                # Interact with the environment using the chosen action
                next_state, reward, done, truncated, info = self.env.step(action)
                print("[trainer.py - run()]   - Action taken:")
                # print(f"[trainer.py - run()]        - next_state: {next_state}")
                print(f"[trainer.py - run()]        - reward: {reward}")
                # print(f"[trainer.py - run()]        - info: {info}")

                # Store experience
                self.agent.store_transition(state, action, reward, next_state)

                print("[trainer.py - run()]   - Train step")
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
