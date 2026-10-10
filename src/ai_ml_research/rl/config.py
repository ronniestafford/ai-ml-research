from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    model_device: torch.device
    env_device: torch.device
    gamma: float  # Used to evaluate the agent
    trajectory_count: int


@dataclass(frozen=True)
class AgentConfig:
    gamma: float  # Used to optimise the agent
    learning_rate: float
    mini_batch_size: int
    optim_steps_per_transition: int
