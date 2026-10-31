from collections.abc import Callable
from dataclasses import replace

import torch
from torch import optim
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer

from ai_ml_research.rl.dpcme.bellman import ValueTarget
from ai_ml_research.rl.dpcme.functions import (
    QFunctionModelBased,
    ValueFunction,
)
from ai_ml_research.rl.dpcme.value_function.config import (
    ValueFunctionConfig,
)
from ai_ml_research.rl.dpcme.value_function.metrics import (
    ValueFunctionOptimisationMetrics,
)
from ai_ml_research.rl.optimisable import Optimisable


class ValueFunctionOptimiser(
    Optimisable[ValueFunctionOptimisationMetrics],
):
    """Value function v(s) = w·φ(s), fit by model-based TD in the φ
    space; its Bellman backup
    q(s,a) = r(s,a) + γ·(1−is_terminated)·μ(s,a)·w
    is derived by the q function, never fit.

    Collaborators (value and q functions, optimiser, replay buffer,
    loss) are injected by the caller, which also puts the modules on
    the model device. Each step regresses the value v(s) = w·φ(s)
    (value_function) onto its model-based Bellman backup q(s,a)
    (q_function):

        minimise  ( w·φ(s) − [ r + γ·(1−is_term)·μ(s,a)·w ] )² = δ²

    so the successor value comes from the CME μ ≈ E[φ(S′)|s,a] rather
    than from a sampled successor state, and is masked off when the
    transition terminated (no successor exists; truncation keeps it).
    Which gradient of this residual is used is selected by
    config.optimisation.update_rule:

        "semigradient":      δ·φ(s)
            (target q(s,a) held constant; TD fixed point)
        "residual_gradient": 2δ·(φ(s) − γ·(1−is_term)·μ(s,a))
            (target keeps its w dependence; descent on δ²)

    In both forms μ enters as a constant, trained by the CME. The
    buffer is shared with the agent that owns this component:
    observe() on the agent fills it, optimise() here consumes it.

    The Bellman target itself is injected as value_target (a
    ValueTarget: q_function, minibatch -> [batch]), e.g.
    model_based_behaviour_value_target (policy evaluation
    along the realised action, on-policy),
    model_based_optimal_value_target (max over actions, off-policy)
    or model_based_expected_sarsa_target partially applied with the
    current ε. Each target reads its own slice of the sampled
    transition. The target switches which Bellman operator's fixed
    point w converges to; the optimisation machinery is identical.
    """

    def __init__(
        self,
        value_function: ValueFunction,
        q_function: QFunctionModelBased,
        optimiser: optim.Optimizer,
        replay_buffer: TensorDictReplayBuffer,
        loss_fn: Callable[..., torch.Tensor],
        config: ValueFunctionConfig,
        optimisation_metrics: ValueFunctionOptimisationMetrics,
        value_target: ValueTarget,
        name: str = "value",
    ) -> None:
        super().__init__(optimisation_metrics, name=name)
        self.value_function = value_function
        self.q_function = q_function
        self.optimiser = optimiser
        self.replay_buffer = replay_buffer
        self.loss_fn = loss_fn
        self.config = config
        self.value_target = value_target
        self.model_device = next(value_function.weights.parameters()).device
        self._grad_element_count = value_function.get_param_count()

    def optimise(self) -> None:
        if (
            len(self.replay_buffer)
            <= self.config.optimisation.warm_up_transition_count
        ):
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

        # Begin autograd graph; the injected target reads its own
        # slice of the minibatch.
        state: torch.Tensor = minibatch.get("observation")  # [batch, ...]
        v: torch.Tensor = self.value_function(state)  # [batch]

        loss, q = (
            self._residual_gradient_loss(v, minibatch)
            if self.config.optimisation.update_rule == "residual_gradient"
            else self._semigradient_loss(v, minibatch)
        )

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
            self.value_function.weights.parameters(),
            max_norm=self.config.optimisation.max_grad_norm,
        )
        post_clip_norm: torch.Tensor = grad_norm.clamp(
            max=self.config.optimisation.max_grad_norm
        )
        grad_rms: torch.Tensor = post_clip_norm / self._grad_element_count**0.5

        self.optimiser.step()

        with torch.no_grad():
            # Per-step means, so the trajectory averages are means too.
            v_sq: torch.Tensor = v.detach().square().mean()
            q_sq: torch.Tensor = q.square().mean()

        self._update_optimisation_metrics(
            metrics,
            loss,
            post_clip_norm,
            grad_rms,
            v_sq,
            q_sq,
        )

    def _semigradient_loss(
        self,
        v: torch.Tensor,
        minibatch: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Bellman residual δ = w·φ(s) − target, target evaluated
        # without a graph (μ and w held constant), so loss.backward()
        # yields the semigradient δ·φ(s): stochastic approximation
        # of the target Bellman operator's fixed point, whatever
        # value_target was injected.
        with torch.no_grad():
            q: torch.Tensor = self.value_target(self.q_function, minibatch)
        return self.loss_fn(v, q), q

    def _residual_gradient_loss(
        self,
        v: torch.Tensor,
        minibatch: TensorDictBase,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Residual δ = w·φ(s) − target, target kept fully
        # differentiable in w (μ still a constant, trained by the
        # CME): gradient descent on the squared model Bellman
        # residual. For the behaviour target
        # ∇½δ² = δ·(φ(s) − γ·(1−is_term)·μ(s,a)); for the max_a
        # target δ·(φ(s) − γ·μ(s,a*)), a* the argmax action.
        q: torch.Tensor = self.value_target(self.q_function, minibatch)
        return self.loss_fn(v, q), q

    def _update_optimisation_metrics(
        self,
        metrics: ValueFunctionOptimisationMetrics,
        loss: torch.Tensor,
        post_clip_norm: torch.Tensor,
        grad_rms: torch.Tensor,
        v_sq: torch.Tensor,
        q_sq: torch.Tensor,
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
            v_sq_sum=metrics.v_sq_sum + v_sq,
            q_sq_sum=metrics.q_sq_sum + q_sq,
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
        v_rms = (float(metrics.v_sq_sum) / step_count) ** 0.5
        q_rms = (float(metrics.q_sq_sum) / step_count) ** 0.5

        return base + (
            f" │ {prefix}v_rms: {v_rms:.4f} │ {prefix}q_rms: {q_rms:.4f}"
        )
