from dataclasses import dataclass

from ai_ml_research.rl.config import AgentConfig


@dataclass(frozen=True)
class DPCMEConfig(AgentConfig):
    # Exploration
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float

    # Replay buffer
    replay_buffer_max_size: int
    warm_up_transition_count: int
