from dataclasses import dataclass

from ai_ml_research.rl.config import AgentConfig


@dataclass(frozen=True)
class DPCMEConfig(AgentConfig):
    # This agent takes no gradient step of its own (the transition model
    # does), so there is no OptimisationConfig here; the CME owns its own.
    # gamma (inherited) is the Bellman-target discount.
    # Exploration
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float

    # Replay buffer
    replay_buffer_max_size: int
