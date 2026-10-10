from dataclasses import replace

import torch
from torch.nn.functional import one_hot
from tensordict import TensorDictBase

from ai_ml_research.rl.agent import Agent
from ai_ml_research.rl.mbrl_pcme.config import MbrlPcmeConfig
from ai_ml_research.rl.mbrl_pcme.metrics import MbrlPcmeOptimisationMetrics


class MbrlPcmeAgent(Agent[MbrlPcmeOptimisationMetrics]):
    """Random-policy template exercising the Agent contract.

    Contains no algorithm-specific machinery: the policy draws uniform
    random actions and the only state kept is the optimisation metrics,
    which demonstrate the device-side and CPU-side metric idioms.
    """

    def __init__(
        self,
        action_count: int,
        config: MbrlPcmeConfig,
        optimisation_metrics: MbrlPcmeOptimisationMetrics,
    ) -> None:
        super().__init__(optimisation_metrics)
        self.action_count = action_count
        self.config = config

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
        metrics = self.optimisation_metrics
        reward: torch.Tensor = transition.get(("next", "reward"))

        self.optimisation_metrics = replace(
            metrics,
            # [Performance] Device-side accumulation: no .item(),
            # stays on the env device.
            reward_sum=metrics.reward_sum + reward.sum(),
            reward_sq_sum=metrics.reward_sq_sum + reward.square().sum(),
            # CPU-side data: plain host scalars. The float(reward) is a
            # deliberate per-step host sync — shown as the contrast to
            # the device-side accumulation above.
            reward_element_count=metrics.reward_element_count + 1,
            last_reward=float(reward),
            # Keeps print_metrics printing for a non-optimising agent.
            trajectory_step_count=metrics.trajectory_step_count + 1,
        )

    def optimise(self) -> None:
        """Called after every transition; nothing to do for a random policy."""

    def format_metrics(self) -> str:
        metrics = self.optimisation_metrics
        # Single device → host sync of the trajectory's metrics.
        element_count = metrics.reward_element_count
        avg_reward = float(metrics.reward_sum) / element_count
        reward_rms = (float(metrics.reward_sq_sum) / element_count) ** 0.5

        return (
            f" │ avg_reward: {avg_reward:+.4f} │"
            f" reward_rms: {reward_rms:.4f} │"
            f" last_reward: {metrics.last_reward:+.4f}"
        )
