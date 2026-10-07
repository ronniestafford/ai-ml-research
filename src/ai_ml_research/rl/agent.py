from torch import nn


class QFunction(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_count: int,
        hidden_dim: int,
        hidden_layers: int = 3,
    ):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim)]

        for _ in range(hidden_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_count))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
