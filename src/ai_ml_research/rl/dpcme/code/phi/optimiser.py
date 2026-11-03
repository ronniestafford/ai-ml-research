import copy
from collections.abc import Callable
from dataclasses import replace

import torch

from torch import optim
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer

from ai_ml_research.rl.dpcme.code.functions import (
    MuFunction,
    PhiFunction,
)
from ai_ml_research.rl.dpcme.code.phi.config import PhiConfig
from ai_ml_research.rl.dpcme.code.phi.metrics import (
    PhiOptimisationMetrics,
)
from ai_ml_research.rl.optimisable import Optimisable


class PhiOptimiser(Optimisable[PhiOptimisationMetrics]):
    """Learned feature map φ: R^state_dim -> R^phi_space_dim, fit
    to be the predictable function of states, by SGD.

    Owns two copies of φ:

    - the fast φ, trained through its forward;
    - a slow exponential-moving-average target copy φ̄, which every
      consumer reads — the CME regression target and the value
      fit's feature map. The consumers therefore see a nearly
      static space, and φ's gradient is closed off from every loop
      by a detach (μ enters graph-free), so each update is a
      regression against constants.

    On buffered rows (s, a, s′):

        minimise  ‖φ(s′) − μ(s,a)‖²
                  + λ·‖(1/B)Σ φᵢφᵢᵀ − I‖²_F

    The predictability term makes φ whatever the current learned
    operator predicts — a successor feature of the learned
    dynamics; the whitening term is the mandatory anti-collapse
    guard (a constant φ is perfectly predictable). The successor
    is masked exactly as in the CME regression: after termination
    the successor measure is the zero measure, whose feature is 0.

    Collaborators (φ, μ, optimiser, replay buffer, loss) are
    injected by the caller, which also puts the modules on the
    model device. The agent ticks this component before the CME,
    which ticks before the value fit: w fastest, μ middle, φ̄
    slowest.
    """

    def __init__(
        self,
        phi_function: PhiFunction,
        cme_function: MuFunction,
        optimiser: optim.Optimizer,
        replay_buffer: TensorDictReplayBuffer,
        loss_fn: Callable[..., torch.Tensor],
        config: PhiConfig,
        optimisation_metrics: PhiOptimisationMetrics,
        name: str = "phi",
    ) -> None:
        super().__init__(optimisation_metrics, name=name)
        self.phi_function = phi_function
        self.cme_function = cme_function
        self.optimiser = optimiser
        self.replay_buffer = replay_buffer
        self.loss_fn = loss_fn
        self.config = config

        # The target copy: same initial weights, never a gradient.
        # Deep-copied (not shared) so the fast φ can move freely
        # underneath it.
        self._phi_target = copy.deepcopy(phi_function)
        self._phi_target.requires_grad_(False)

        self.model_device = next(phi_function.parameters()).device
        self._grad_element_count = phi_function.get_param_count()

    @property
    def target_phi(self) -> PhiFunction:
        """The copy every consumer reads (CME target, value fit)."""
        return self._phi_target

    def optimise(self) -> None:
        for _ in range(self.config.optimisation.sgd_step_count):
            # [Performance] Commit batch to GPU here,
            # nowhere else.
            minibatch = self.replay_buffer.sample(
                batch_size=min(
                    len(self.replay_buffer),
                    self.config.optimisation.mini_batch_size,
                )
            ).to(self.model_device)

            self._optimise_step(minibatch)

        self._update_target()

    def _optimise_step(self, minibatch: TensorDictBase) -> None:
        metrics = self.optimisation_metrics

        state_action: torch.Tensor = torch.cat(
            (minibatch.get("observation"), minibatch.get("action")),
            dim=-1,
        )  # [batch, state_dim + action_count]

        # μ is a constant: graph-free inference (it is trained by
        # the CME optimiser; φ is trained to be predictable from
        # it, never through it).
        mu: torch.Tensor = self.cme_function.inference(
            state_action
        )  # [batch, phi_dim]

        # Begin autograd graph: training path → φ's own forward.
        phi_successor: torch.Tensor = self.phi_function(
            minibatch.get(("next", "observation"))
        )  # [batch, phi_dim]

        # Whitening penalty on the batch's second moment, taken
        # before the termination mask (a zeroed row is a feature
        # geometry artefact, not a feature).
        whitening: torch.Tensor = _whitening_penalty(phi_successor)

        # is_terminated (not done): after a terminated transition
        # the successor is the zero feature; on truncation the
        # successor exists and its target stays. Masked in tensor
        # arithmetic, as in the CME regression.
        is_terminated: torch.Tensor = minibatch.get(
            ("next", "terminated")
        )  # bool [batch, 1]
        phi_successor = phi_successor * is_terminated.logical_not()

        loss = (
            self.loss_fn(phi_successor, mu)
            + self.config.optimisation.whitening_coefficient * whitening
        )

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
            self.phi_function.parameters(),
            max_norm=self.config.optimisation.max_grad_norm,
        )
        post_clip_norm: torch.Tensor = grad_norm.clamp(
            max=self.config.optimisation.max_grad_norm
        )
        grad_rms: torch.Tensor = post_clip_norm / self._grad_element_count**0.5

        self.optimiser.step()

        with torch.no_grad():
            # Per-step means, so the trajectory averages are means too.
            phi_norm_sq: torch.Tensor = (
                phi_successor.square().sum(dim=-1).mean()
            )

        self._update_optimisation_metrics(
            metrics,
            loss,
            post_clip_norm,
            grad_rms,
            phi_norm_sq,
            whitening.detach(),
        )

    def _update_target(self) -> None:
        # φ̄ ← τ·φ̄ + (1−τ)·φ — the Polyak/EMA recursion, one tick
        # per optimise() call. lerp_(src, w) = self + w·(src − self);
        # the public per-tensor method keeps this off torch's private
        # API, at a loop over the map's handful of parameter tensors.
        one_minus_tau: float = 1.0 - self.config.optimisation.target_ema_rate
        with torch.no_grad():
            for target_param, param in zip(
                self._phi_target.parameters(),
                self.phi_function.parameters(),
            ):
                target_param.lerp_(param.detach(), one_minus_tau)

    def _update_optimisation_metrics(
        self,
        metrics: PhiOptimisationMetrics,
        loss: torch.Tensor,
        post_clip_norm: torch.Tensor,
        grad_rms: torch.Tensor,
        phi_norm_sq: torch.Tensor,
        whitening: torch.Tensor,
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
            phi_norm_sq_sum=metrics.phi_norm_sq_sum + phi_norm_sq,
            whitening_sum=metrics.whitening_sum + whitening,
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
        phi_rms = (float(metrics.phi_norm_sq_sum) / step_count) ** 0.5
        avg_whitening = float(metrics.whitening_sum) / step_count

        return base + (
            f" │ {prefix}phi_rms: {phi_rms:.4f}"
            f" │ {prefix}whitening: {avg_whitening:.4f}"
        )


def _whitening_penalty(features: torch.Tensor) -> torch.Tensor:
    # ‖(1/B)Σ φᵢφᵢᵀ − I‖²_F on [batch, phi_dim] rows: penalises
    # any departure of the batch second moment from the identity —
    # feature collapse (rank-deficient covariance) and feature
    # blow-up alike. The identity target is a scale choice: the
    # predictability loss fixes the direction scale relative to μ,
    # the whitening the spread.
    gram = features.T @ features / features.shape[0]
    identity = torch.eye(
        features.shape[-1], device=features.device, dtype=features.dtype
    )
    return (gram - identity).square().sum()
