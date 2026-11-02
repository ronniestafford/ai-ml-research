from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class ValueFunctionOptimisationMetrics(OptimisationMetrics):
    # GPU-side data: stay on (and grow on) the compute device.
    # Per-step means of v(s)² and of its model-based Bellman backup
    # q(s,a)²: comparing the two checks that the value keeps pace with
    # the return scale it regresses onto.
    v_sq_sum: Scalar = 0.0
    q_sq_sum: Scalar = 0.0
