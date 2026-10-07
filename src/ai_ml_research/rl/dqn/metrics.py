from dataclasses import dataclass


@dataclass(frozen=True)
class OptimisationMetrics:
    loss_sum: float = 0.0
    grad_norm_sum: float = 0.0
    grad_rms_sum: float = 0.0
    optim_step_count: int = 0
    td_error_sum: float = 0.0
    td_error_abs_sum: float = 0.0
    td_error_sq_sum: float = 0.0
    td_error_in_huber_count: int = 0
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
        print(
            f"Trajectory {trajectory_idx} │"
            f" discounted reward: {metrics_trajectory.discounted_reward:.4f} │"
            f" transitions: {metrics_trajectory.transition_count} │"
            f" total transitions: {metrics_global.transition_count} │"
            f" total optim steps: {metrics_global.optim_step_count} │"
            f" avg_loss: "
            f"{metrics_optimisation.loss_sum /
               metrics_optimisation.optim_step_count:.4f} │"
            f" avg_grad_norm: "
            f"{metrics_optimisation.grad_norm_sum /
               metrics_optimisation.optim_step_count:.4f} │"
            f" avg_grad_rms: "
            f"{metrics_optimisation.grad_rms_sum /
               metrics_optimisation.optim_step_count:.6f} │"
            f" avg_td_error: "
            f"{metrics_optimisation.td_error_sum /
               metrics_optimisation.td_error_element_count:+.4f} │"
            f" td_error_rms: "
            f"{(metrics_optimisation.td_error_sq_sum /
                metrics_optimisation.td_error_element_count) ** 0.5:.4f} │"
            f" |td_error|<β: "
            f"{metrics_optimisation.td_error_in_huber_count /
               metrics_optimisation.td_error_element_count:.0%} │"
            f" epsilon: {metrics_global.epsilon:.4f}"
        )
