from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class CMEOptimisationMetrics(OptimisationMetrics):
    # GPU-side data: stay on (and grow on) the compute device.
    # Per-step means of ‖μ(s,a)‖² and ‖φ(s′)‖²: comparing the two checks
    # that the embedding lives at the scale of its regression target.
    mu_norm_sq_sum: Scalar = 0.0
    phi_norm_sq_sum: Scalar = 0.0
