from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    Count,
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class DQNOptimisationMetrics(OptimisationMetrics):
    td_error_sum: Scalar = 0.0
    td_error_abs_sum: Scalar = 0.0
    td_error_sq_sum: Scalar = 0.0
    td_error_in_huber_count: Count = 0
    td_error_element_count: int = 0
