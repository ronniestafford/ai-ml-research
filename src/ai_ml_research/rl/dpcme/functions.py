from abc import ABC, abstractmethod

import torch
from torch import nn


class FunctionModule(ABC, nn.Module):
    """A component that computes a pure function of tensors.

    The forward/inference convention (shared by every function
    module below): forward() — invoked as module(x) — is the
    evaluation itself, the path training code calls when it needs
    the autograd graph; inference() wraps it in no_grad and is the
    path evaluation code calls, treating the module as a constant.
    Subclasses implement forward() only.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor: ...

    def inference(self, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return self(*args, **kwargs)


class PhiFunction(FunctionModule):
    """Fixed random Fourier feature map φ: R^state_dim -> R^phi_space_dim.

    φ(s) = √(2/d) · cos(Ws + b) with W_ij ~ N(0, 1/bandwidth²) and
    b_i ~ U(0, 2π), so ⟨φ(s), φ(s')⟩ estimates the RBF kernel k(s, s')
    and ‖φ(s)‖ ≈ 1.

    φ defines the space the CME regresses onto, it is not learned:
    the projection is frozen.
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


class MuFunction(FunctionModule):
    """
    - CME μ:SxA->Fϕ is the expected next-state feature ϕ(S').
    - μ(s,a) = Wψ(s,a) where ψ is the state-action feature.

    Training calls forward (the CME optimiser trains μ through
    it); evaluation code calls inference.
    """

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
        self.psi_layer = layers[-1]
        layers.append(nn.Linear(psi_space_dim, phi_space_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ValueFunction(FunctionModule):
    """
    - Value function v:S->R parameterized by w in the frozen φ space:
      v(s) = w·φ(s).

    φ is frozen and injected by reference (evaluated through its
    inference), so only the value weights w are learnable here.

    The value optimiser trains w through forward; evaluation code
    calls inference.
    """

    def __init__(
        self,
        phi_function: PhiFunction,
        phi_space_dim: int,
    ) -> None:
        super().__init__()
        self.phi_function = phi_function
        self.linear_layer = nn.Linear(phi_space_dim, 1, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # v(s) = w·φ(s); the graph to w is kept (training path).
        # φ is fixed: evaluated through its no-graph inference.
        phi = self.phi_function.inference(state)
        return self.linear_layer(phi).squeeze(-1)

    def get_param_count(self) -> int:
        # parameters() also walks the shared φ; only w is learnable
        # by this component.
        return self.linear_layer.weight.numel()


class QFunctionModelBased(FunctionModule):
    """
    - State-action value function q:SxA->R defined through the CME:
      q(s,a) = r(s,a) + γ·(1−is_terminated)·μ(s,a)·w, the model-based
      Bellman backup of v(s) = w·φ(s). The termination mask kills the
      successor term when no S′ exists (terminated, not truncated:
      on a timeout the successor does exist).
    - The value lives in the frozen φ space, and because
      μ(s,a) ≈ E[φ(S′)|s,a], the successor value is
      μ(s,a)·w ≈ E[v(S′)|s,a].

    Owns no parameters: v, φ and μ are injected by reference and
    enter q as constants (μ is trained by the CME component and
    is evaluated through its inference). Bellman targets call
    forward — the residual-gradient rule needs the graph through
    w; pure evaluation calls inference.

    Also exposed alongside the Bellman optimality backup
    v(s) = r + γ·max_a μ(s,a)·w, the off-policy (DQN-style max)
    alternative to the on-policy backup, implemented as the
    model_based_optimal_value_target() function of the bellman
    module.
    """

    def __init__(
        self,
        value_function: ValueFunction,
        cme_function: MuFunction,
        gamma: float,
        action_count: int,
    ) -> None:
        super().__init__()
        self.value_function = value_function
        self.cme_function = cme_function
        self.gamma = gamma
        self.action_count = action_count

    def forward(
        self,
        state_action: torch.Tensor,
        reward: torch.Tensor,
        is_terminated: torch.Tensor,
    ) -> torch.Tensor:
        # q(s,a) = r(s,a) + γ·(1−is_terminated)·μ(s,a)·w for stacked
        # (s, a) pairs; is_terminated is the realised transition's bool
        # flag. At decision time no transition exists yet: pass a
        # zeros bool tensor (survival mask fully open).
        # μ is a constant here — evaluated through its no-graph
        # inference (it is trained by the CME, not by the value);
        # the graph to w is kept.
        mu = self.cme_function.inference(state_action)
        weights = self.value_function.linear_layer.weight
        successor_value = (mu @ weights.T).squeeze(-1)
        # Mask in tensor arithmetic (type promotion 0-/1-ifies).
        successor_value = successor_value * is_terminated.logical_not()
        return reward + self.gamma * successor_value
