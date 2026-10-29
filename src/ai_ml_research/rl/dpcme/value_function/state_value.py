import torch
from torch import nn


class PhiFunction(nn.Module):
    """Fixed random Fourier feature map φ: R^state_dim -> R^phi_space_dim.

    φ(s) = √(2/d) · cos(Ws + b) with W_ij ~ N(0, 1/bandwidth²) and
    b_i ~ U(0, 2π), so ⟨φ(s), φ(s')⟩ estimates the RBF kernel k(s, s')
    and ‖φ(s)‖ ≈ 1.

    φ defines the space the CME regresses onto, it is not learned:
    the projection is frozen and forward runs without a graph.
    """

    def __init__(
        self,
        state_dim: int,
        phi_space_dim: int,
        bandwidth: float,
    ) -> None:
        super().__init__()

        self.normalisation: float = (2.0 / phi_space_dim) ** 0.5

        projection = nn.Linear(state_dim, phi_space_dim)
        with torch.no_grad():
            projection.weight.normal_(mean=0.0, std=1.0 / bandwidth)
            projection.bias.uniform_(0.0, 2.0 * torch.pi)
        self.projection = projection.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalisation * torch.cos(self.projection(x))
