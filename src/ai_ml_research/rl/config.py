from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    model_device: torch.device
    env_device: torch.device
    gamma: float  # Used to evaluate the agent
    trajectory_count: int

    # Typical magnitude of each state dimension (one entry per
    # dimension of the env's observation). Feature maps measure
    # distances in units of this scale, so every dimension of the
    # state space contributes in its own units.
    state_scale: tuple[float, ...] | None


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
