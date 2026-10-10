from abc import ABC, abstractmethod
from dataclasses import fields, replace
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from tensordict import TensorDictBase

from ai_ml_research.rl.metrics import (
    GlobalMetrics,
    OptimisationMetrics,
    TrajectoryMetrics,
)

# OptimisationMetrics can be specialised to type T
# e.g. DQNAgent inherits Agent[DQNOptimisationMetrics].
T = TypeVar("T", bound=OptimisationMetrics)


def get_epsilon_exp_decay(
    global_transition_idx: int,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
) -> float:
    return eps_end + (eps_start - eps_end) * np.exp(
        -global_transition_idx / eps_decay
    )


def get_epsilon_linear_decay(
    global_transition_idx: int,
    eps_start: float,
    eps_end: float,
    transition_idx_end: float,
) -> float:
    return (
        eps_start
        + global_transition_idx * (eps_end - eps_start) / transition_idx_end
    )


class Agent(ABC, Generic[T]):
    """Contract between an RL agent and the generic experiment runner.

    The runner steps the environment and calls the agent once per
    transition; the agent owns everything algorithm-specific:
    action selection (including exploration), what to do with each
    transition (replay buffer, trajectory accumulator, ...), when
    and how to take optimisation steps, and the optimisation
    metrics themselves.
    """

    def __init__(self, optimisation_metrics: T) -> None:
        self.optimisation_metrics: T = optimisation_metrics

    @abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """Return an action in the format env.step() expects."""

    @abstractmethod
    def observe(self, transition: TensorDictBase) -> None:
        """Store the full transition just produced by env.step()."""

    @abstractmethod
    def optimise(self) -> None:
        """Called after every transition."""

    def on_trajectory_start(self) -> None:
        """Called before env.reset() of each trajectory."""
        # field.default is Any-typed; every metrics field declares one,
        # so annotating the dict keeps replace() happy.
        reset: dict[str, Any] = {
            field.name: field.default
            for field in fields(self.optimisation_metrics)
            if field.name != "optim_step_count"
        }
        self.optimisation_metrics = replace(self.optimisation_metrics, **reset)

    def on_trajectory_end(self) -> None:
        """Called when the trajectory is done, before metrics are printed.

        Typically where on-policy agents run their updates over the collected
        trajectory.
        """

    def format_metrics(self) -> str:
        """Agent-specific suffix appended to the trajectory's metric line."""
        return ""

    def print_metrics(
        self,
        trajectory_idx: int,
        metrics_global: GlobalMetrics,
        metrics_trajectory: TrajectoryMetrics,
    ) -> None:
        metrics_optimisation = self.optimisation_metrics
        if metrics_optimisation.trajectory_step_count > 0:
            # Single device → host sync for the whole trajectory's metrics.
            step_count = metrics_optimisation.trajectory_step_count
            avg_loss = float(metrics_optimisation.loss_sum) / step_count
            avg_grad_norm = (
                float(metrics_optimisation.grad_norm_sum) / step_count
            )
            avg_grad_rms = (
                float(metrics_optimisation.grad_rms_sum) / step_count
            )

            discounted_reward = metrics_trajectory.discounted_reward
            transition_count = metrics_trajectory.transition_count
            total_transitions = metrics_global.transition_count
            total_optim_steps = metrics_optimisation.optim_step_count

            line = (
                f"Trajectory {trajectory_idx} │"
                f" discounted reward: {discounted_reward:.4f} │"
                f" transitions: {transition_count} │"
                f" total transitions: {total_transitions} │"
                f" total optim steps: {total_optim_steps} │"
                f" avg_loss: {avg_loss:.4f} │"
                f" avg_grad_norm: {avg_grad_norm:.4f} │"
                f" avg_grad_rms: {avg_grad_rms:.6f}"
            )
            line += self.format_metrics()
            print(line)
