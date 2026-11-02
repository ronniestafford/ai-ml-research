from abc import ABC, abstractmethod

import numpy as np
import torch
from tensordict import TensorDictBase

from ai_ml_research.rl.metrics import (
    GlobalMetrics,
    TrajectoryMetrics,
)
from ai_ml_research.rl.optimisable import T, Optimisable


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


class Agent(Optimisable[T], ABC):
    """Contract between an RL agent and the generic experiment runner.

    The runner steps the environment and calls the agent once per
    transition; the agent owns everything algorithm-specific:
    action selection (including exploration), what to do with each
    transition (replay buffer, trajectory accumulator, ...), when
    and how to take optimisation steps, and the optimisation
    metrics themselves.

    An agent is itself an Optimisable and may compose further
    Optimisable components (a transition model, an encoder, ...) whose
    lifecycle hooks and metrics it drives; see sub_optimisables().
    """

    @abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """Return an action in the format env.step() expects."""

    @abstractmethod
    def observe(self, transition: TensorDictBase) -> None:
        """Store the full transition just produced by env.step()."""

    def sub_optimisables(self) -> tuple[Optimisable, ...]:
        """Optimisable components owned by this agent (none by default)."""
        return ()

    def optimisation_components(self) -> tuple[Optimisable, ...]:
        """Every optimisable contributing to this agent's metric line."""
        return (self, *self.sub_optimisables())

    def on_trajectory_start(self) -> None:
        """Called before env.reset() of each trajectory."""
        super().on_trajectory_start()
        for component in self.sub_optimisables():
            component.on_trajectory_start()

    def on_trajectory_end(self) -> None:
        """Called when the trajectory is done, before metrics are printed.

        Typically where on-policy agents run their updates over the collected
        trajectory.
        """
        for component in self.sub_optimisables():
            component.on_trajectory_end()

    def print_metrics(
        self,
        trajectory_idx: int,
        metrics_global: GlobalMetrics,
        metrics_trajectory: TrajectoryMetrics,
    ) -> None:
        components = self.optimisation_components()
        if all(
            c.optimisation_metrics.trajectory_step_count == 0
            for c in components
        ):
            return

        discounted_reward = metrics_trajectory.discounted_reward
        transition_count = metrics_trajectory.transition_count
        total_transitions = metrics_global.transition_count
        total_optim_steps = sum(
            c.optimisation_metrics.optim_step_count for c in components
        )

        line = (
            f"Trajectory {trajectory_idx} │"
            f" discounted reward: {discounted_reward:.4f} │"
            f" transitions: {transition_count} │"
            f" total transitions: {total_transitions} │"
            f" total optim steps: {total_optim_steps} │"
        )
        line += "".join(c.format_metrics() for c in components)
        print(line)
