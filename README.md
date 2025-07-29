# Pusher

Reinforcement Learning exercise using [Gymmnasium](https://gymnasium.farama.org/)'s [MuJoCo](https://mujoco.org/) envirorment.

“Pusher” is a multi-jointed robot arm that is very similar to a human arm. The goal is to move a target cylinder (called object) to a goal position using the robot’s end effector (called fingertip). The robot consists of shoulder, elbow, forearm and wrist joints.


## Project structure
```

rl-mujoco-pusher/
├── pyproject.toml
├── README.md
├── main.py                        # Entry point to run training/evaluation
├── config/
│   └── settings.py                # Hyperparameters and environment settings
├── application/
│   ├── train_agent.py             # Use case: Train agent
│   └── evaluate_agent.py          # Use case: Evaluate agent
├── core/
│   ├── agent.py                   # Agent logic (action selection, update)
│   ├── policy.py                  # Policy abstraction and implementations
│   └── trainer.py                 # Training loop
├── infra/
│   ├── gym_environment.py         # Adapter for Gymnasium environments
│   └── logger.py                  # Simple logging utility
└── tests/
    ├── test_agent.py
    ├── test_policy.py
    └── test_trainer.py
```


## Pusher RL

### Action space

-   **Number of actions:** 7

-   **Action type:** Joint torques applied to a robotic arm

-   **Control range:** Each joint can apply torque between **-2 Nm and +2 Nm**

-   **Joints controlled:**

    1.  Shoulder pan (rotate horizontally)

    2.  Shoulder lift (move up/down)

    3.  Upper arm roll (rotate upper arm)

    4.  Elbow flex (bend elbow)

    5.  Forearm roll (rotate forearm)

    6.  Wrist flex (bend wrist)

    7.  Wrist roll (rotate wrist)

-   **Joint type:** All are **hinge joints**

-   **Action unit:** **Torque (Newton-meters)**

![pusher](https://github.com/MartinezAgullo/rl-mujoco-pusher/blob/main/docs/pusher.png)

### Observation space
The observation space consists of the following 23 parts (in order):

- qpos (7 elements): Position values of the robot’s body parts.

- qvel (7 elements): The velocities of these individual body parts (their derivatives).

- xpos (3 elements): The coordinates of the fingertip of the pusher.

- xpos (3 elements): The coordinates of the object to be moved.

-xpos (3 elements): The coordinates of the goal position.


### Rewards

### Starting State

### Episode End
