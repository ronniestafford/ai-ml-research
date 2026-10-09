from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    Count,
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class PCMEOptimisationMetrics(OptimisationMetrics):
    error_sum: Scalar = 0.0
    some_count: Count = 0
    element_count: int = 0
