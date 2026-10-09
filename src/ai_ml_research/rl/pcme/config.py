from dataclasses import dataclass

from ai_ml_research.rl.config import AgentConfig


@dataclass(frozen=True)
class PCMEConfig(AgentConfig):
    # Q Functions
    hidden_dim: int
    hidden_layer_count: int

    # Target Q function
    target_soft_update: float
    optim_steps_per_target_update: int

    # Loss and gradient clipping
    huber_beta: float
    max_grad_norm: float

    # Exploration
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float

    # Replay buffer
    replay_buffer_max_size: int
    warm_up_transition_count: int
