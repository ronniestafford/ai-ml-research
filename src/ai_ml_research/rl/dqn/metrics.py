from dataclasses import dataclass
from typing import Union

import torch

# Sums are accumulated as device-side scalars (see optimise) and only
# converted to floats once per trajectory, in print_metrics.
Scalar = Union[float, torch.Tensor]
Count = Union[int, torch.Tensor]


@dataclass(frozen=True)
class OptimisationMetrics:
    loss_sum: Scalar = 0.0
    grad_norm_sum: Scalar = 0.0
    grad_rms_sum: Scalar = 0.0
    optim_step_count: int = 0
    td_error_sum: Scalar = 0.0
    td_error_abs_sum: Scalar = 0.0
    td_error_sq_sum: Scalar = 0.0
    td_error_in_huber_count: Count = 0
    td_error_element_count: int = 0


@dataclass(frozen=True)
class TrajectoryMetrics:
    transition_count: int = 0
    discounted_reward: float = 0.0


@dataclass(frozen=True)
class GlobalMetrics:
    trajectory_count: int = 0
    transition_count: int = 0
    optim_step_count: int = 0
    epsilon: float = 1
    seed: int = 0


def print_metrics(
    trajectory_idx: int,
    metrics_global: GlobalMetrics,
    metrics_trajectory: TrajectoryMetrics,
    metrics_optimisation: OptimisationMetrics,
):
    if metrics_optimisation.optim_step_count > 0:
        # Single device → host sync for the whole trajectory's metrics.
        step_count = metrics_optimisation.optim_step_count
        element_count = metrics_optimisation.td_error_element_count
        avg_loss = float(metrics_optimisation.loss_sum) / step_count
        avg_grad_norm = float(metrics_optimisation.grad_norm_sum) / step_count
        avg_grad_rms = float(metrics_optimisation.grad_rms_sum) / step_count
        avg_td_error = (
            float(metrics_optimisation.td_error_sum) / element_count
        )
        td_error_rms = (
            float(metrics_optimisation.td_error_sq_sum) / element_count
        ) ** 0.5
        huber_share = (
            float(metrics_optimisation.td_error_in_huber_count) / element_count
        )

        print(
            f"Trajectory {trajectory_idx} │"
            f" discounted reward: {metrics_trajectory.discounted_reward:.4f} │"
            f" transitions: {metrics_trajectory.transition_count} │"
            f" total transitions: {metrics_global.transition_count} │"
            f" total optim steps: {metrics_global.optim_step_count} │"
            f" avg_loss: {avg_loss:.4f} │"
            f" avg_grad_norm: {avg_grad_norm:.4f} │"
            f" avg_grad_rms: {avg_grad_rms:.6f} │"
            f" avg_td_error: {avg_td_error:+.4f} │"
            f" td_error_rms: {td_error_rms:.4f} │"
            f" |td_error|<β: {huber_share:.0%} │"
            f" epsilon: {metrics_global.epsilon:.4f}"
        )
