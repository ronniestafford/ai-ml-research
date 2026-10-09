from dataclasses import dataclass
from typing import Union

import torch

# Sums are accumulated as device-side scalars (see DQNAgent.optimise) and
# only converted to floats once per trajectory, in Agent.print_metrics.
Scalar = Union[float, torch.Tensor]
Count = Union[int, torch.Tensor]


@dataclass(frozen=True)
class OptimisationMetrics:
    # optim_step_count is the cumulative count across the whole run;
    # trajectory_step_count is reset at the start of each trajectory.
    optim_step_count: int = 0
    trajectory_step_count: int = 0
    loss_sum: Scalar = 0.0
    grad_norm_sum: Scalar = 0.0
    grad_rms_sum: Scalar = 0.0


@dataclass(frozen=True)
class TrajectoryMetrics:
    transition_count: int = 0
    discounted_reward: float = 0.0


@dataclass(frozen=True)
class GlobalMetrics:
    trajectory_count: int = 0
    transition_count: int = 0
    seed: int = 0
