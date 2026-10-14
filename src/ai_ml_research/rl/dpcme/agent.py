from dataclasses import replace

import torch

from torch.nn.functional import one_hot
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer

from ai_ml_research.rl.agent import Agent
from ai_ml_research.rl.dpcme.config import DPCMEConfig
from ai_ml_research.rl.dpcme.metrics import (
    DPCMEOptimisationMetrics,
)
from ai_ml_research.rl.dpcme.transition_model.CME import CME
from ai_ml_research.rl.optimisable import Optimisable


class DPCMEAgent(Agent[DPCMEOptimisationMetrics]):
    """Deep Parametic Conditional Mean Embedding (DPCME) agent.

    - Maintains a CME μ:SxA->Fϕ which is
      an estimate for the expected next-state feature ϕ(S').
    - μ(s,a) = Wψ(s,a) where ψ is the state-action feature.

    The transition model is an injected Optimisable component: it owns its
    parameters, optimiser and metrics. This agent selects actions, fills the
    replay buffer the component reads, and decides the cadence — one tick of
    the transition model per transition.
    """

    def __init__(
        self,
        action_count: int,
        state_dim: int,
        config: DPCMEConfig,
        optimisation_metrics: DPCMEOptimisationMetrics,
        replay_buffer: TensorDictReplayBuffer,
        cme: CME,
    ) -> None:
        super().__init__(optimisation_metrics)
        self.action_count = action_count
        self.state_dim = state_dim
        self.config = config
        self.replay_buffer = replay_buffer
        self.cme = cme

        self._transition_count: int = 0
        self._epsilon: float = config.epsilon_start

    def sub_optimisables(self) -> tuple[Optimisable, ...]:
        return (self.cme,)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        # Uniform random action: keep on env device, one-hot to match
        # the env's action spec.
        action_idx = torch.randint(
            0,
            self.action_count,
            (),
            device=observation.device,
        )
        return one_hot(action_idx, num_classes=self.action_count)

    def observe(self, transition: TensorDictBase) -> None:
        self.replay_buffer.add(transition)
        self._transition_count += 1

        metrics = self.optimisation_metrics
        reward: torch.Tensor = transition.get(("next", "reward"))

        self._update_optimisation_metrics(metrics, reward)

    def optimise(self) -> None:
        # Cadence only: the component gates on its own warm-up.
        self.cme.optimise()

    def _update_optimisation_metrics(
        self, metrics: DPCMEOptimisationMetrics, reward: torch.Tensor
    ):
        self.optimisation_metrics = replace(
            metrics,
            # [Performance] Device-side accumulation: no .item(),
            # thererfore variable stays on GPU device.
            last_reward=reward,
            reward_sum=metrics.reward_sum + reward.sum(),
            reward_sq_sum=metrics.reward_sq_sum + reward.square().sum(),
            reward_element_count=metrics.reward_element_count + 1,
            trajectory_step_count=metrics.trajectory_step_count + 1,
        )

    def format_metrics(self) -> str:
        base = super().format_metrics()
        if not base:
            return ""

        metrics = self.optimisation_metrics
        element_count = metrics.reward_element_count
        avg_reward = float(metrics.reward_sum) / element_count
        reward_rms = (float(metrics.reward_sq_sum) / element_count) ** 0.5
        last_reward = float(metrics.last_reward)

        return base + (
            f" │ avg_reward: {avg_reward:+.4f} │"
            f" reward_rms: {reward_rms:.4f} │"
            f" last_reward: {last_reward:+.4f}"
        )
