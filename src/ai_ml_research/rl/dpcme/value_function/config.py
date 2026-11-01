from dataclasses import dataclass
from typing import Literal

UpdateRule = Literal["semigradient", "residual_gradient"]


@dataclass(frozen=True)
class ValueFunctionOptimisationConfig:
    learning_rate: float
    mini_batch_size: int

    # Policy iteration loop inside optimise(): each iteration fits
    # v with sgd_step_count SGD steps (policy evaluation of the
    # current Bellman target) and then commits a soft
    # (conservative_update_proportion) update of w.
    policy_iteration_count: int
    sgd_step_count: int

    max_grad_norm: float

    # Backup discount γ = γ_backup applied when fitting v:
    # q(s,a) = r(s,a) + γ·(1−is_terminated)·μ(s,a)·w. It lives
    # with the value fit only — the optimiser passes it to the
    # Bellman target on every call — never on the q module or the
    # agent.
    backup_gamma: float

    # Gradient form for δ = w·φ(s) − q(s,a):
    # "semigradient": target q(s,a) = r(s,a) + γ·μ(s,a)·w is held constant,
    # so the step is δ·φ(s).
    # "residual_gradient": the target keeps its w dependence, so the step
    # is 2δ·(φ(s) − γ·μ(s,a)), gradient descent on the squared model
    # Bellman residual.
    update_rule: UpdateRule = "semigradient"

    # Conservative (Polyak) outer update of w across optimise()
    # calls: the inner gradient descent produces w_inner, then
    #   w ← w_prev + conservative_update_proportion·(w_inner − w_prev).
    # 1.0 takes the inner result as is (no damping).
    conservative_update_proportion: float = 1.0


@dataclass(frozen=True)
class ValueFunctionConfig:
    # Random Fourier feature map φ: dimensionless RBF bandwidth,
    # a length scale measured in units of φ's per-dimension state
    # scale, so every state dimension contributes to the kernel in
    # its own units. v lives in φ's space, so φ belongs to the
    # value fit; the CME regresses onto the same (shared) φ.
    phi_bandwidth: float

    # Optimisation
    optimisation: ValueFunctionOptimisationConfig
