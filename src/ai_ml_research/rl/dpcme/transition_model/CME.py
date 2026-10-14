import torch

from torch import nn


class CMEFunction(nn.Module):
    def __init__(
        self,
        state_action_dim: int,
        psi_space_dim: int,
        phi_space_dim: int,
        hidden_dim: int,
        hidden_layer_count: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(state_action_dim, hidden_dim)]
        layers.append(nn.ReLU())

        for _ in range(hidden_layer_count - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, psi_space_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(psi_space_dim, phi_space_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
