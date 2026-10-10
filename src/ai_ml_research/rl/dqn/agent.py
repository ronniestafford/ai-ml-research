import copy
from collections.abc import Callable
from dataclasses import replace

import torch
from torch import nn, optim
from torch.nn.functional import one_hot
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer

from ai_ml_research.rl.agent import Agent, get_epsilon_exp_decay
from ai_ml_research.rl.dqn.config import DQNConfig
from ai_ml_research.rl.dqn.metrics import DQNOptimisationMetrics


class QFunction(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_count: int,
        hidden_dim: int,
        hidden_layer_count: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(state_dim, hidden_dim)]
        layers.append(nn.ReLU())

        for _ in range(hidden_layer_count - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, action_count))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_epsilon_greedy_action(
    q_function: QFunction,
    observation: torch.Tensor,
    epsilon: float,
    action_count: int,
) -> torch.Tensor:
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


class DQNAgent(Agent[DQNOptimisationMetrics]):
    """Off-policy DQN with an ε-greedy policy and a soft-updated target net.

    Collaborators (q network, optimiser, replay buffer, loss) are injected
    by the caller; the target network is derived internally because it is
    a DQN-specific implementation detail, not a collaborator.
    """

    def __init__(
        self,
        q_function: QFunction,
        optimiser: optim.Optimizer,
        replay_buffer: TensorDictReplayBuffer,
        loss_fn: Callable[..., torch.Tensor],
        action_count: int,
        config: DQNConfig,
        optimisation_metrics: DQNOptimisationMetrics,
    ) -> None:
        super().__init__(optimisation_metrics)
        self.q_function = q_function
        self.target_q_function = copy.deepcopy(q_function).requires_grad_(
            False
        )
        self.target_q_function.eval()
        self.optimiser = optimiser
        self.replay_buffer = replay_buffer
        self.loss_fn = loss_fn
        self.action_count = action_count
        self.config = config

        self.model_device = next(q_function.parameters()).device
        self._n_grad_elements: int = q_function.get_param_count()
        self._transition_count: int = 0
        self._epsilon: float = config.epsilon_start

    def on_trajectory_start(self) -> None:
        # Resets this trajectory's optimisation metrics accumulator.
        super().on_trajectory_start()
        # Exploration rate fixed for the whole trajectory, decayed with
        # the number of transitions observed so far.
        self._epsilon = get_epsilon_exp_decay(
            self._transition_count,
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.epsilon_decay,
        )

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return get_epsilon_greedy_action(
            q_function=self.q_function,
            observation=observation,
            epsilon=self._epsilon,
            action_count=self.action_count,
        )

    def observe(self, transition: TensorDictBase) -> None:
        self.replay_buffer.add(transition)
        self._transition_count += 1

    def optimise(self) -> None:
        if len(self.replay_buffer) <= self.config.warm_up_transition_count:
            return

        for _ in range(self.config.optim_steps_per_transition):
            # [Performance] Commit batch to GPU here,
            # nowhere else.
            minibatch = self.replay_buffer.sample(
                batch_size=min(
                    len(self.replay_buffer), self.config.mini_batch_size
                )
            ).to(self.model_device)
            self._optimise_step(minibatch)

    def _optimise_step(self, minibatch: TensorDictBase) -> None:
        metrics = self.optimisation_metrics

        # Begin autograd graph
        q_values: torch.Tensor = self.q_function(
            minibatch.get("observation")
        )  # [batch, n_actions]

        # backprop is through max actions only
        action_idx = minibatch.get("action").argmax(dim=-1)
        # .gather appends all the actions to autograd graph
        current_q = q_values.gather(
            dim=-1, index=action_idx.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            successor_q: torch.Tensor = (
                self.target_q_function(minibatch.get(("next", "observation")))
                .max(dim=-1)
                .values
            )

        # if "Terminated": q_target = r,
        # else:            q_target = r + γ·max_a' Q_target(s',a')
        # e.g. no s' when cart-pole pole is dropped:
        target_q: torch.Tensor = minibatch.get(("next", "reward")).squeeze(
            -1
        ) + (
            self.config.gamma
            * successor_q
            * ~minibatch.get(("next", "terminated")).squeeze(-1)
        )

        # TD error δ = q_target − Q(s,a)
        #            = r + γ·max_a' Q_target(s',a') − Q(s,a)
        with torch.no_grad():
            td_error: torch.Tensor = target_q - current_q
            td_error_abs = td_error.abs()
            td_error_element_count: int = td_error.numel()
            td_error_in_huber_count: torch.Tensor = (
                td_error_abs < self.config.huber_beta
            ).sum()

        loss = self.loss_fn(
            input=current_q,
            target=target_q,
            beta=self.config.huber_beta,
        )

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()

        # Not really needed when using something like Huber loss
        # as δ rarely exceeds quadratic regime of Huber loss.
        grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
            self.q_function.parameters(), max_norm=self.config.max_grad_norm
        )
        post_clip_norm: torch.Tensor = grad_norm.clamp(
            max=self.config.max_grad_norm
        )
        grad_rms: torch.Tensor = post_clip_norm / self._n_grad_elements**0.5

        self.optimiser.step()

        self._update_optimisation_metrics(
            metrics,
            td_error,
            td_error_abs,
            td_error_element_count,
            td_error_in_huber_count,
            loss,
            post_clip_norm,
            grad_rms,
        )

        if (
            self.optimisation_metrics.optim_step_count
            % self.config.optim_steps_per_target_update
            == 0
        ):
            soft_update_target(
                q_function=self.q_function,
                target_q_function=self.target_q_function,
                tau=self.config.target_soft_update,
            )

    def _update_optimisation_metrics(
        self,
        metrics: DQNOptimisationMetrics,
        td_error: torch.Tensor,
        td_error_abs: torch.Tensor,
        td_error_element_count: int,
        td_error_in_huber_count: torch.Tensor,
        loss: torch.Tensor,
        post_clip_norm: torch.Tensor,
        grad_rms: torch.Tensor,
    ) -> None:
        # [Performance] Metrics are accumulated as device scalars. The single
        # sync happens in print_metrics, once per trajectory.
        self.optimisation_metrics = replace(
            metrics,
            loss_sum=metrics.loss_sum + loss.detach(),
            grad_norm_sum=metrics.grad_norm_sum + post_clip_norm,
            grad_rms_sum=metrics.grad_rms_sum + grad_rms,
            optim_step_count=metrics.optim_step_count + 1,
            trajectory_step_count=metrics.trajectory_step_count + 1,
            td_error_sum=metrics.td_error_sum + td_error.sum(),
            td_error_abs_sum=metrics.td_error_abs_sum + td_error_abs.sum(),
            td_error_sq_sum=metrics.td_error_sq_sum + td_error.square().sum(),
            td_error_in_huber_count=metrics.td_error_in_huber_count
            + td_error_in_huber_count,
            td_error_element_count=metrics.td_error_element_count
            + td_error_element_count,
        )

    def format_metrics(self) -> str:
        metrics = self.optimisation_metrics
        element_count = metrics.td_error_element_count
        avg_td_error = float(metrics.td_error_sum) / element_count
        td_error_rms = (float(metrics.td_error_sq_sum) / element_count) ** 0.5
        huber_share = float(metrics.td_error_in_huber_count) / element_count

        return (
            f" │ avg_td_error: {avg_td_error:+.4f} │"
            f" td_error_rms: {td_error_rms:.4f} │"
            f" |td_error|<β: {huber_share:.0%} │"
            f" epsilon: {self._epsilon:.4f}"
        )
