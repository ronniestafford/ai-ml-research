from dataclasses import dataclass
from typing import Literal

UpdateRule = Literal["semigradient", "residual_gradient"]


@dataclass(frozen=True)
class ValueFunctionOptimisationConfig:
    learning_rate: float
    mini_batch_size: int
    optim_steps_per_transition: int
    max_grad_norm: float
    warm_up_transition_count: int

    # Gradient form for δ = w·φ(s) − q(s,a):
    # "semigradient": target q(s,a) = r(s,a) + γ·μ(s,a)·w is held constant,
    # so the step is δ·φ(s).
    # "residual_gradient": the target keeps its w dependence, so the step
    # is 2δ·(φ(s) − γ·μ(s,a)), gradient descent on the squared model
    # Bellman residual.
    update_rule: UpdateRule = "semigradient"


@dataclass(frozen=True)
class ValueFunctionConfig:
    # Bellman-target discount: q(s,a) = r(s,a) + γ·μ(s,a)·w.
    gamma: float

    # Optimisation
    optimisation: ValueFunctionOptimisationConfig
