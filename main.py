"""Entry point for the RL project."""

from application.train_agent import train_agent

if __name__ == "__main__":
    # Run the training use case
    train_agent()

    # After traning we evaluate the model
    # evaluate_agent(model_path="trained_policy.pth", render=True)
