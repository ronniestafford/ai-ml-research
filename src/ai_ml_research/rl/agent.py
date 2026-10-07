from torch import nn


class QFunction(nn.Module):
    def __init__(self, state_dim: int, action_count: int, hidden_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, x):
        return self.network(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
