from collections.abc import Callable
from dataclasses import replace

import torch

from torch import nn, optim
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer

from ai_ml_research.rl.dpcme.transition_model.config import CMEConfig
from ai_ml_research.rl.dpcme.transition_model.metrics import (
    CMEOptimisationMetrics,
)
from ai_ml_research.rl.optimisable import Optimisable


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


class CME(Optimisable[CMEOptimisationMetrics]):
    """Conditional mean embedding μ(s,a) ≈ E[φ(S') | S=s, A=a], fit by SGD.

    Collaborators (CME function, φ feature map, optimiser, replay buffer,
    loss) are injected by the caller, which also puts the modules on the
    model device. φ is fixed, so learning μ is a regression problem:

        minimise  ‖μ_θ(s,a) − φ(s')‖²  over transitions in the buffer

    whose minimiser is the conditional mean E[φ(S') | s,a]. The buffer is
    shared with the agent that owns this component: observe() on the agent
    fills it, optimise() here consumes it.
    """

    def __init__(
        self,
        cme_function: CMEFunction,
        phi_function: PhiFunction,
        optimiser: optim.Optimizer,
        replay_buffer: TensorDictReplayBuffer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: CMEConfig,
        optimisation_metrics: CMEOptimisationMetrics,
        name: str = "cme",
    ) -> None:
        super().__init__(optimisation_metrics, name=name)
        self.cme_function = cme_function
        self.phi_function = phi_function
        self.optimiser = optimiser
        self.replay_buffer = replay_buffer
        self.loss_fn = loss_fn
        self.config = config

        self.model_device = next(cme_function.parameters()).device
        self._grad_element_count = cme_function.get_param_count()

    def optimise(self) -> None:
        if len(self.replay_buffer) <= self.config.warm_up_transition_count:
            return

        for _ in range(self.config.optimisation.optim_steps_per_transition):
            # [Performance] Commit batch to GPU here,
            # nowhere else.
            minibatch = self.replay_buffer.sample(
                batch_size=min(
                    len(self.replay_buffer),
                    self.config.optimisation.mini_batch_size,
                )
            ).to(self.model_device)

            self._optimise_step(minibatch)

    def _optimise_step(self, minibatch: TensorDictBase) -> None:
        metrics = self.optimisation_metrics

        state_action: torch.Tensor = torch.cat(
            (minibatch.get("observation"), minibatch.get("action")), dim=-1
        )  # [batch, state_dim + action_count]

        # Begin autograd graph
        mu: torch.Tensor = self.cme_function(state_action)  # [batch, phi_dim]

        # Regression target: φ is fixed, so it needs no gradient.
        with torch.no_grad():
            phi_successor: torch.Tensor = self.phi_function(
                minibatch.get(("next", "observation"))
            )  # [batch, phi_dim]

        loss = self.loss_fn(mu, phi_successor)

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
            self.cme_function.parameters(), max_norm=self.config.max_grad_norm
        )
        post_clip_norm: torch.Tensor = grad_norm.clamp(
            max=self.config.max_grad_norm
        )
        grad_rms: torch.Tensor = (
            post_clip_norm / self._grad_element_count**0.5
        )

        self.optimiser.step()

        with torch.no_grad():
            # Per-step means, so the trajectory averages are means too.
            mu_norm_sq: torch.Tensor = mu.detach().square().sum(dim=-1).mean()
            phi_norm_sq: torch.Tensor = phi_successor.square().sum(
                dim=-1
            ).mean()

        self._update_optimisation_metrics(
            metrics,
            loss,
            post_clip_norm,
            grad_rms,
            mu_norm_sq,
            phi_norm_sq,
        )

    def _update_optimisation_metrics(
        self,
        metrics: CMEOptimisationMetrics,
        loss: torch.Tensor,
        post_clip_norm: torch.Tensor,
        grad_rms: torch.Tensor,
        mu_norm_sq: torch.Tensor,
        phi_norm_sq: torch.Tensor,
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
            mu_norm_sq_sum=metrics.mu_norm_sq_sum + mu_norm_sq,
            phi_norm_sq_sum=metrics.phi_norm_sq_sum + phi_norm_sq,
        )

    def format_metrics(self) -> str:
        base = super().format_metrics()
        if not base:
            return ""

        metrics = self.optimisation_metrics
        # Tensor->Float parsing syncs host device with GPU data.
        # This is a performance hit, so only do here.
        step_count = metrics.trajectory_step_count
        prefix = self._metric_prefix()
        mu_rms = (float(metrics.mu_norm_sq_sum) / step_count) ** 0.5
        phi_rms = (float(metrics.phi_norm_sq_sum) / step_count) ** 0.5

        return base + (
            f" │ {prefix}mu_rms: {mu_rms:.4f} │ {prefix}phi_rms: {phi_rms:.4f}"
        )
