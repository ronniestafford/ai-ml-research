"""Bellman regression targets for the DPCME value function.

Each target maps a sampled minibatch of transitions to the
regression target [batch] for v(s) = w·φ(s), under a choice of
Bellman operator (behaviour evaluation, optimality, expected
SARSA); the value function optimiser injects one and is agnostic
to which.
"""

from collections.abc import Callable

import torch
from tensordict import TensorDictBase
from torch.nn.functional import one_hot

from ai_ml_research.rl.dpcme.functions import QFunctionModelBased

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


def model_based_successor_values(
    q_function: QFunctionModelBased,
    state: torch.Tensor,
) -> torch.Tensor:
    # μ(s,a)·w ≈ E[v(S′)|s,a] evaluated for every action at each
    # state: [batch, action_count]. μ is a constant (trained by the
    # CME); the graph to w is kept for the residual gradient.
    # Also the shared all-action evaluation behind the ε-greedy
    # targets below and the agent's greedy policy
    # argmax_a q(s,a) = argmax_a μ(s,a)·w (at decision time the
    # reward term is unknown and drops out of the argmax).
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
    successor_values = model_based_successor_values(q_function, state)
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
    successor_values = model_based_successor_values(q_function, state)
    greedy_value = successor_values.max(dim=-1).values
    uniform_mean = successor_values.mean(dim=-1)
    return reward + q_function.gamma * (
        (1.0 - epsilon) * greedy_value + epsilon * uniform_mean
    )
