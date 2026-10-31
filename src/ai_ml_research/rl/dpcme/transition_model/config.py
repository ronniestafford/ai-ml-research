from dataclasses import dataclass


@dataclass(frozen=True)
class CMEOptimisationConfig:
    learning_rate: float
    mini_batch_size: int
    optim_steps_per_transition: int
    max_grad_norm: float
    warm_up_transition_count: int


@dataclass(frozen=True)
class CMEConfig:
    # CME function: ψ (state-action features) then W (ψ -> φ space)
    state_action_dim: int
    psi_space_dim: int
    phi_space_dim: int
    hidden_dim: int
    hidden_layer_count: int

    # Random Fourier feature map φ: μ regresses onto its output space,
    # so phi_space_dim is shared with the CME function's output.
    phi_bandwidth: float

    # Optimisation
    optimisation: CMEOptimisationConfig
