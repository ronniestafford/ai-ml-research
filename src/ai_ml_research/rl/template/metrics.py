from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class TemplateOptimisationMetrics(OptimisationMetrics):
    # GPU-side data: Stay on (and grow on) the env/compute device.
    reward_sum: Scalar = 0.0
    reward_sq_sum: Scalar = 0.0

    # CPU-side data: plain host scalars, cheap Python arithmetic.
    reward_element_count: int = 0
    last_reward: float = 0.0
