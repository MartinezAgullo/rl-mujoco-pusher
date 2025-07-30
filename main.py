"""Entry point for the RL project."""

import argparse

from application.evaluate_agent import evaluate_agent
from application.train_agent import train_agent


def main():
    parser = argparse.ArgumentParser(
        description="RL Project: Train or Evaluate an Agent on Pusher-v5"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="Choose whether to train a new agent or evaluate an existing one.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained model file (required for evaluation).",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation.",
    )

    args = parser.parse_args()

    # Train the RL agent
    if args.mode == "train":
        print("[main.py] Mode: Training")
        train_agent()

    # Evaluate the RL agent
    elif args.mode == "eval":
        print("[main.py] Mode: Evaluation")
        if not args.model_path:
            raise ValueError("You must provide --model-path when evaluating.")
        evaluate_agent(model_path=args.model_path, render=args.render)


if __name__ == "__main__":
    main()
