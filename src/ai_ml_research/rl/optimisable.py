from abc import ABC, abstractmethod
from dataclasses import fields, replace
from typing import Any, Generic, TypeVar

from ai_ml_research.rl.metrics import OptimisationMetrics

# OptimisationMetrics can be specialised to type T
# e.g. CME inherits Optimisable[CMEOptimisationMetrics].
T = TypeVar("T", bound=OptimisationMetrics)


class Optimisable(ABC, Generic[T]):
    """Contract for every component that takes gradient optimisation steps.

    A component owns its optimiser, its parameters and its own
    OptimisationMetrics accumulator; optimise() decides the cadence and
    the loss. nn.Module subclasses stay pure (forward only) and are
    wrapped by an Optimisable, so optimisers are always injected
    downwards: experiment -> agent -> component -> module.
    """

    def __init__(self, optimisation_metrics: T, name: str = "") -> None:
        self.optimisation_metrics: T = optimisation_metrics
        # Label for this component's metrics on the shared metric line.
        # Agents leave it empty; injected components name themselves.
        self._name: str = name
        # Elements in the gradient tensor, used for the grad_rms metric.
        # Set by the subclass as soon as its parameters exist.
        self._grad_element_count: int = 0

    @abstractmethod
    def optimise(self) -> None:
        """One cadence tick's worth of optimisation steps."""

    def on_trajectory_start(self) -> None:
        """Called before env.reset() of each trajectory."""
        # field.default is Any-typed; every metrics field declares one,
        # so annotating the dict keeps replace() happy.
        reset: dict[str, Any] = {
            field.name: field.default
            for field in fields(self.optimisation_metrics)
            if field.name != "optim_step_count"
        }
        self.optimisation_metrics = replace(self.optimisation_metrics, **reset)

    def on_trajectory_end(self) -> None:
        """Called when the trajectory is done, before metrics are printed."""

    def format_metrics(self) -> str:
        """This component's optimisation slice of the trajectory metric line.

        Subclasses append algorithm-specific metrics by chaining through
        super(), and must return "" early when this returns "" (their own
        accumulators are then empty too).
        """
        metrics = self.optimisation_metrics
        if metrics.trajectory_step_count == 0:
            return ""

        # Single device → host sync for the whole trajectory's metrics.
        step_count = metrics.trajectory_step_count
        avg_loss = float(metrics.loss_sum) / step_count
        avg_grad_norm = float(metrics.grad_norm_sum) / step_count
        avg_grad_rms = float(metrics.grad_rms_sum) / step_count
        prefix = self._metric_prefix()

        return (
            f" │ {prefix}avg_loss: {avg_loss:.4f} │"
            f" {prefix}avg_grad_norm: {avg_grad_norm:.4f} │"
            f" {prefix}avg_grad_rms: {avg_grad_rms:.6f}"
        )

    def _metric_prefix(self) -> str:
        return f"{self._name} " if self._name else ""
