# Pusher

Reinforcement Learning exercise using Gymmnasium's MuJoCo envirorment. 

“Pusher” is a multi-jointed robot arm that is very similar to a human arm. The goal is to move a target cylinder (called object) to a goal position using the robot’s end effector (called fingertip). The robot consists of shoulder, elbow, forearm and wrist joints. 

### Gymnasium and MuJoCo

- Gymnasium is an open-source Python library that provides a standardized collection of reinforcement learning environments

- MuJoCo (Multi-Joint dynamics with Contact) is a physics engine that facilitates the rapid and accurate simulation of complex systems, especially those with multiple rigid bodies and intricate contact interactions, such as robots. 

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

### Observation space

### Rewards

### Starting State

### Episode End


