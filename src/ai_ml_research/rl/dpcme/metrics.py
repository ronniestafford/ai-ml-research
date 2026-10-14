from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class DPCMEOptimisationMetrics(OptimisationMetrics):
    reward_sum: Scalar = 0.0
    reward_sq_sum: Scalar = 0.0

    # CPU-side data: plain host scalars, cheap Python arithmetic.
    reward_element_count: int = 0
    last_reward: Scalar = 0.0
