from dataclasses import replace

import torch
from torch import nn, optim
from torch.nn.functional import one_hot
from tensordict import TensorDictBase
from ai_ml_research.rl.dqn.metrics import OptimisationMetrics


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


def get_epsilon_greedy_action(
    q_function: QFunction,
    observation: torch.Tensor,
    epsilon: float,
    action_count: int,
):
    is_single_observation = observation.ndim == 1
    if is_single_observation:
        observation = observation.unsqueeze(0)
    observation_count = observation.shape[0]

    # What actions are greedy or random?
    random_mask = torch.rand(observation_count) < epsilon
    greedy_mask = ~random_mask
    action_idx = torch.randint(0, action_count, (observation_count,))

    if greedy_mask.any():
        with torch.no_grad():
            q_values = q_function(observation[greedy_mask])

        action_idx[greedy_mask] = q_values.argmax(dim=-1)

    action_one_hot = one_hot(action_idx, num_classes=action_count)

    if is_single_observation:
        action_one_hot = action_one_hot.squeeze(0)

    return action_one_hot


def optimise(
    q_function: QFunction,
    target_q_function: QFunction,
    optimiser: optim.Optimizer,
    minibatch: TensorDictBase,
    metrics: OptimisationMetrics,
    loss_fn,
    gamma: float,
    n_grad_elements: int,
    max_norm: float,
    huber_beta: float,
) -> OptimisationMetrics:

    # Begin autograd graph
    q_values: torch.Tensor = q_function(
        minibatch.get("observation")
    )  # [batch, n_actions]

    # backprop is through max action only
    action_idx = minibatch.get("action").argmax(dim=-1)
    # .gather adds all the actions to autograd graph
    current_q = q_values.gather(dim=-1, index=action_idx.unsqueeze(1)).squeeze(
        1
    )

    with torch.no_grad():
        successor_q: torch.Tensor = (
            target_q_function(minibatch.get(("next", "observation")))
            .max(dim=-1)
            .values
        )

    # "Terminated": There is no successor state if MDP trajectory is terminated
    # e.g. when cart-pole pole is dropped.
    target_q: torch.Tensor = minibatch.get(("next", "reward")).squeeze(-1) + (
        gamma
        * successor_q
        * ~minibatch.get(("next", "terminated")).squeeze(-1)
    )

    # TD error δ = r + γ·max_a' Q_target(s',a') − Q(s,a)
    with torch.no_grad():
        td_error: torch.Tensor = target_q - current_q
        td_error_abs = td_error.abs()
        td_error_element_count: int = td_error.numel()
        td_error_in_huber_count: int = int(
            (td_error_abs < huber_beta).sum().item()
        )

    loss = loss_fn(input=current_q, target=target_q, beta=huber_beta)

    optimiser.zero_grad(set_to_none=True)
    loss.backward()

    # Not really needed has TD error rarely exceeds
    # quadratic regime of Huber loss anyway.
    grad_norm: float = torch.nn.utils.clip_grad_norm_(
        q_function.parameters(), max_norm=max_norm
    ).item()
    post_clip_norm = min(grad_norm, max_norm)
    grad_rms: float = post_clip_norm / n_grad_elements**0.5

    optimiser.step()

    return replace(
        metrics,
        loss_sum=metrics.loss_sum + loss.item(),
        grad_norm_sum=metrics.grad_norm_sum + post_clip_norm,
        grad_rms_sum=metrics.grad_rms_sum + grad_rms,
        optim_step_count=metrics.optim_step_count + 1,
        td_error_sum=metrics.td_error_sum + td_error.sum().item(),
        td_error_abs_sum=metrics.td_error_abs_sum + td_error_abs.sum().item(),
        td_error_sq_sum=metrics.td_error_sq_sum
        + td_error.square().sum().item(),
        td_error_in_huber_count=metrics.td_error_in_huber_count
        + td_error_in_huber_count,
        td_error_element_count=metrics.td_error_element_count
        + td_error_element_count,
    )
