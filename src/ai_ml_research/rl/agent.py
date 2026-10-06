from typing import Protocol
from torch import nn
from torch.optim import Optimizer


class QFunction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent(Protocol):
    q_function: nn.Module
    q_function_target: nn.Module
    optimiser: Optimizer

    def __init__(self, q_function: nn.Module, optimiser: Optimizer):
        self.q_function = q_function
        self.optimiser = optimiser
