from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class DPCMEOptimisationMetrics(OptimisationMetrics):
    # Every new field must declare a default: Agent.on_trajectory_start
    # resets all fields (except optim_step_count) to their field.default.

    # Device-side accumulation: updated in MbrlPcmeAgent.observe() with
    # device tensors — no .item() — so these stay on (and grow on) the
    # env/compute device. The single host sync happens in format_metrics.
    reward_sum: Scalar = 0.0
    reward_sq_sum: Scalar = 0.0

    # CPU-side data: plain host scalars, cheap Python arithmetic.
    reward_element_count: int = 0
    last_reward: float = 0.0
