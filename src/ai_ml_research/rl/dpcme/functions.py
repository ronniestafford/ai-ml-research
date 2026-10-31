from collections.abc import Callable

import torch
from tensordict import TensorDictBase
from torch import nn
from torch.nn.functional import one_hot


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


class MuFunction(nn.Module):
    """
    - CME μ:SxA->Fϕ is the expected next-state feature ϕ(S').
    - μ(s,a) = Wψ(s,a) where ψ is the state-action feature.
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


class ValueFunction(nn.Module):
    """
    - Value function v:S->R parameterized by w in the frozen φ space:
      v(s) = w·φ(s).

    φ is frozen and injected by reference (evaluated without a
    graph), so only the value weights w are learnable here.
    """

    def __init__(
        self,
        phi_function: PhiFunction,
        phi_space_dim: int,
    ) -> None:
        super().__init__()
        self.phi_function = phi_function
        self.weights = nn.Linear(phi_space_dim, 1, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # v(s) = w·φ(s).
        # φ is fixed, so it never needs a gradient.
        with torch.no_grad():
            phi = self.phi_function(state)
        return self.weights(phi).squeeze(-1)

    def get_param_count(self) -> int:
        # parameters() also walks the shared φ; only w is learnable
        # by this component.
        return self.weights.weight.numel()


class QFunctionModelBased(nn.Module):
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
    enter q as constants (μ is trained by the CME component and is
    evaluated without a graph), so nothing here trains w — that is
    the value function component's job.

    Also exposed alongside the Bellman optimality backup
    v(s) = r + γ·max_a μ(s,a)·w, the off-policy (DQN-style max)
    alternative to the on-policy backup, implemented as the
    model_based_optimal_value_target() function of this module.
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
        # μ is a constant here: it is trained by the CME, not by the value.
        with torch.no_grad():
            mu = self.cme_function(state_action)
        weights = self.value_function.weights.weight
        successor_value = (mu @ weights.T).squeeze(-1)
        # Mask in tensor arithmetic (type promotion 0-/1-ifies).
        successor_value = successor_value * is_terminated.logical_not()
        return reward + self.gamma * successor_value


# Shared Bellman-target interface: given the model-based q function
# (whose μ and w the targets read as their model/value constants)
# and a sampled minibatch of transitions, return the regression
# target [batch] for v(s) = w·φ(s). Each target reads the slice of
# the transition it needs: the policy-evaluation backup prices the
# realised action and masks terminated successors, the
# policy-improvement (max) and policy-expectation (ε-greedy)
# backups evaluate all actions at s and read only (s, r).
ValueTarget = Callable[[QFunctionModelBased, TensorDictBase], torch.Tensor]


def model_based_behaviour_value_target(
    q_function: QFunctionModelBased,
    minibatch: TensorDictBase,
) -> torch.Tensor:
    # Model-based backup along the realised action:
    # q(s,a) = r + γ·(1−is_terminated)·μ(s,a)·w (the q function's
    # forward). Policy-evaluation backup of the behaviour policy:
    # the realised action prices the target, so the fixed point is
    # the value of the policy that generated the data (on-policy).
    state: torch.Tensor = minibatch.get("observation")
    action: torch.Tensor = minibatch.get("action")
    reward: torch.Tensor = minibatch.get(("next", "reward")).squeeze(-1)
    # is_terminated (not done): on truncation a successor exists and
    # bootstrapping must continue; on termination it must not. Kept
    # bool — the q function masks in tensor arithmetic.
    is_terminated: torch.Tensor = minibatch.get(
        ("next", "terminated")
    ).squeeze(-1)
    state_action = torch.cat((state, action), dim=-1)
    return q_function(state_action, reward, is_terminated)


def _model_based_successor_values(
    q_function: QFunctionModelBased,
    state: torch.Tensor,
) -> torch.Tensor:
    # μ(s,a)·w ≈ E[v(S′)|s,a] evaluated for every action at each
    # state: [batch, action_count]. μ is a constant (trained by the
    # CME); the graph to w is kept for the residual gradient.
    action_count = q_function.action_count
    batch_size = state.shape[0]
    states = state.repeat_interleave(action_count, dim=0)
    actions = one_hot(
        torch.arange(
            action_count, device=state.device, dtype=torch.long
        ).repeat(batch_size),
        num_classes=action_count,
    ).to(state.dtype)
    with torch.no_grad():
        mu = q_function.cme_function(torch.cat((states, actions), dim=-1))
    weights = q_function.value_function.weights.weight
    return (mu @ weights.T).squeeze(-1).reshape(batch_size, action_count)


def model_based_optimal_value_target(
    q_function: QFunctionModelBased,
    minibatch: TensorDictBase,
) -> torch.Tensor:
    # Bellman optimality backup v(s) = r + γ·max_a μ(s,a)·w:
    # successor value maximised over all actions at s, so the
    # action in the data no longer prices the target.
    # State-value function equivalent to DQN off-policy target.
    # No termination mask needed: CME already prices
    # termination risk in μ.
    # Only (s, r) are read: all actions at s are evaluated, and
    # every buffered state is non-terminal.
    state: torch.Tensor = minibatch.get("observation")
    reward: torch.Tensor = minibatch.get(("next", "reward")).squeeze(-1)
    successor_values = _model_based_successor_values(q_function, state)
    return reward + q_function.gamma * successor_values.max(dim=-1).values


def model_based_expected_sarsa_target(
    q_function: QFunctionModelBased,
    minibatch: TensorDictBase,
    epsilon: float,
) -> torch.Tensor:
    # Expected backup of the current ε-greedy policy AKA expected SARSA.
    # a* = argmax_a r + γ·μ(s,a)·w
    # v(s) = r + γ·[(1−ε)·μ(s,a*)·w + ε·mean_a μ(s,a)·w]
    #      = r + γ·[max_a μ(s,a)·w − ε·(max_a μ(s,a)·w − mean_a μ(s,a)·w)]
    #
    # ε = 0:
    #   — pure greedy evaluation; fixed point v^*, off-policy.
    #   - recovers model_based_optimal_value_target exactly.
    # 0 < ε < 1
    #   — evaluation of the ε-greedy policy itself (model-based expected SARSA)
    #   - fixed point v_π, on-policy
    # ε = 1
    #   — value of the uniform-random policy: r + γ·mean_a μ(s,a)·w.
    #   - uniform-average backup.
    # Only (s, r) are read: all actions at s are evaluated.
    state: torch.Tensor = minibatch.get("observation")
    reward: torch.Tensor = minibatch.get(("next", "reward")).squeeze(-1)
    successor_values = _model_based_successor_values(q_function, state)
    greedy_value = successor_values.max(dim=-1).values
    uniform_mean = successor_values.mean(dim=-1)
    return reward + q_function.gamma * (
        (1.0 - epsilon) * greedy_value + epsilon * uniform_mean
    )
