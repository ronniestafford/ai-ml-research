from dataclasses import dataclass

from ai_ml_research.rl.metrics import (
    OptimisationMetrics,
    Scalar,
)


@dataclass(frozen=True)
class PhiOptimisationMetrics(OptimisationMetrics):
    # GPU-side data: stay on (and grow on) the compute device.
    # Per-step means of ‖φ(s′)‖² (fast copy, pre-mask): the static
    # RFF map was exactly unit-norm, so this is the scale a learned
    # map should keep near O(1).
    phi_norm_sq_sum: Scalar = 0.0
    # Per-step means of the unscaled whitening penalty
    # ‖Ĉ − I‖²_F: watching collapse directly — it should stay
    # bounded, not decay to the trivial zero-feature value.
    whitening_sum: Scalar = 0.0
