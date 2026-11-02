from dataclasses import dataclass


@dataclass(frozen=True)
class PhiOptimisationConfig:
    learning_rate: float
    mini_batch_size: int
    sgd_step_count: int
    max_grad_norm: float

    # Coefficient λ of the whitening penalty
    # ‖(1/B)Σ φ(sᵢ)φ(sᵢ)ᵀ − I‖²_F on the batch's features.
    # Anti-collapse is mandatory: a constant φ satisfies the
    # predictability loss exactly, so without this term the
    # learned space degenerates. The penalty keeps every feature
    # at O(1) variance (uncentred second moment: the mean
    # direction is whitened too).
    whitening_coefficient: float

    # EMA rate τ of the target copy, updated once per optimise()
    # call: φ̄ ← φ̄ + (1−τ)·(φ − φ̄). Consumers (the CME regression
    # target, the value fit) read φ̄ only, so the space they see
    # drifts on the 1/(1−τ)-call time scale — keep it far slower
    # than the value fit's blend rate.
    target_ema_rate: float


@dataclass(frozen=True)
class PhiConfig:
    # False (the default): the injected φ is wired to its
    # consumers directly and no PhiOptimiser runs — today's
    # frozen-feature behaviour, bit-identical.
    train_phi: bool

    # Optimisation
    optimisation: PhiOptimisationConfig
