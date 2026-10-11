from dataclasses import dataclass

from ai_ml_research.rl.config import AgentConfig


@dataclass(frozen=True)
class DPCMEConfig(AgentConfig):
    """A random policy needs no hyperparameters of its own; extend
    AgentConfig here with algorithm-specific fields."""
