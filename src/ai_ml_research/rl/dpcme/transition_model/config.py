from dataclasses import dataclass


@dataclass(frozen=True)
class CMEOptimisationConfig:
    learning_rate: float
    mini_batch_size: int
    sgd_step_count: int
    max_grad_norm: float


@dataclass(frozen=True)
class CMEConfig:
    # CME function: ψ (state-action features) then W (ψ -> φ space)
    state_action_dim: int
    psi_space_dim: int
    # Random Fourier feature map φ: μ regresses onto its output
    # space, so phi_space_dim is shared with φ's output.
    phi_space_dim: int
    hidden_dim: int
    hidden_layer_count: int

    # Optimisation
    optimisation: CMEOptimisationConfig
