from dataclasses import dataclass

from ai_ml_research.rl.config import AgentConfig, OptimisationConfig


@dataclass(frozen=True)
class TemplateConfig(AgentConfig):
    # Optimisation: unused by the random policy, here as the slot
    # where the real algorithm's gradient step lives.
    optimisation: OptimisationConfig

    # Exploration
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float

    # Replay buffer
    replay_buffer_max_size: int
    warm_up_transition_count: int
