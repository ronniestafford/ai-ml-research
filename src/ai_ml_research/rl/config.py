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
class OptimisationConfig:
    # Hyperparameters of a gradient step. They live with whichever
    # Optimisable takes the step: an agent, or a component it composes.
    learning_rate: float
    mini_batch_size: int
    optim_steps_per_transition: int


@dataclass(frozen=True)
class AgentConfig:
    gamma: float  # Used to optimise the agent
