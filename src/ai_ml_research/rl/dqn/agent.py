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
        hidden_layer_count: int,
    ):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim)]
        layers.append(nn.ReLU())

        for _ in range(hidden_layer_count - 1):
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
    # [Performance] Random actions: keep on env device
    env_device = observation.device
    is_single_observation = observation.ndim == 1
    if is_single_observation:
        observation = observation.unsqueeze(0)

    observation_count = observation.shape[0]
    random_mask = torch.rand(observation_count, device=env_device) < epsilon
    action_idx = torch.randint(
        0, action_count, (observation_count,), device=env_device
    )

    # [Performance] Greedy actions: forward pass and greedy action indexes
    # calculated on the model's device. Final action indexes calculated on the
    # env device.
    if not random_mask.all():
        model_device = next(q_function.parameters()).device
        with torch.no_grad():
            # [Performance] Keep on model device
            greedy_idx = q_function(observation.to(model_device)).argmax(
                dim=-1
            )
        # [Performance] Final action indexes: calculate on env device,
        # inhibit CPU sync with torch.where, just in case env_device
        # is GPU-based.
        action_idx = torch.where(
            random_mask, action_idx, greedy_idx.to(env_device)
        )

    action_one_hot = one_hot(action_idx, num_classes=action_count)

    if is_single_observation:
        action_one_hot = action_one_hot.squeeze(0)

    return action_one_hot


def soft_update_target(
    q_function: QFunction,
    target_q_function: QFunction,
    tau: float,
) -> None:
    # Soft update: θ_target ← (1 − τ)·θ_target + τ·θ_current
    # τ=1: hard update
    # τ=0: no update (freeze target)
    with torch.no_grad():
        for target_param, param in zip(
            target_q_function.parameters(), q_function.parameters()
        ):
            target_param.mul_(1.0 - tau).add_(param, alpha=tau)


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
    # .gather appends all the actions to autograd graph
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
        td_error_in_huber_count: torch.Tensor = (
            td_error_abs < huber_beta
        ).sum()

    loss = loss_fn(input=current_q, target=target_q, beta=huber_beta)

    optimiser.zero_grad(set_to_none=True)
    loss.backward()

    # Not really needed has TD error rarely exceeds
    # quadratic regime of Huber loss anyway.
    grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
        q_function.parameters(), max_norm=max_norm
    )
    post_clip_norm: torch.Tensor = grad_norm.clamp(max=max_norm)
    grad_rms: torch.Tensor = post_clip_norm / n_grad_elements**0.5

    optimiser.step()

    # [Performance] Metrics are accumulated as device scalars. The single sync
    # happens in print_metrics, once per trajectory.
    return replace(
        metrics,
        loss_sum=metrics.loss_sum + loss.detach(),
        grad_norm_sum=metrics.grad_norm_sum + post_clip_norm,
        grad_rms_sum=metrics.grad_rms_sum + grad_rms,
        optim_step_count=metrics.optim_step_count + 1,
        td_error_sum=metrics.td_error_sum + td_error.sum(),
        td_error_abs_sum=metrics.td_error_abs_sum + td_error_abs.sum(),
        td_error_sq_sum=metrics.td_error_sq_sum + td_error.square().sum(),
        td_error_in_huber_count=metrics.td_error_in_huber_count
        + td_error_in_huber_count,
        td_error_element_count=metrics.td_error_element_count
        + td_error_element_count,
    )
