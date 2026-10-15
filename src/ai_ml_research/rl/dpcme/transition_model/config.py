from dataclasses import dataclass

from ai_ml_research.rl.config import OptimisationConfig


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
    optimisation: OptimisationConfig
    max_grad_norm: float
    warm_up_transition_count: int
