from dataclasses import replace

import torch
from torch.nn.functional import one_hot
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer

from ai_ml_research.rl.agent import Agent
from ai_ml_research.rl.template.config import TemplateConfig
from ai_ml_research.rl.template.metrics import TemplateOptimisationMetrics


class TemplateAgent(Agent[TemplateOptimisationMetrics]):
    """Random-policy template exercising the Agent contract.

    Contains no algorithm-specific machinery: the policy draws uniform
    random actions and the only state kept is the optimisation metrics,
    which demonstrate the device-side and CPU-side metric idioms.
    """

    def __init__(
        self,
        action_count: int,
        state_dim: int,
        config: TemplateConfig,
        optimisation_metrics: TemplateOptimisationMetrics,
        replay_buffer: TensorDictReplayBuffer,
    ) -> None:
        super().__init__(optimisation_metrics)
        self.action_count = action_count
        self.state_dim = state_dim
        self.config = config
        self.replay_buffer = replay_buffer

        self._transition_count: int = 0
        self._epsilon: float = config.epsilon_start

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

        self._update_optimisation_metrics(metrics, reward)

    def _update_optimisation_metrics(
        self, metrics: TemplateOptimisationMetrics, reward: torch.Tensor
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

    def optimise(self) -> None:
        """Called after every transition; nothing to do for a random policy."""

    def format_metrics(self) -> str:
        # super() renders the shared loss/gradient slice; the agent's own
        # metrics are appended to it.
        base = super().format_metrics()
        if not base:
            return ""

        metrics = self.optimisation_metrics
        # Tensor->Float parsing syncs host device with GPU data.
        # This is a performance hit, so only do here.
        element_count = metrics.reward_element_count
        avg_reward = float(metrics.reward_sum) / element_count
        reward_rms = (float(metrics.reward_sq_sum) / element_count) ** 0.5
        last_reward = float(metrics.last_reward)

        return base + (
            f" │ avg_reward: {avg_reward:+.4f} │"
            f" reward_rms: {reward_rms:.4f} │"
            f" last_reward: {last_reward:+.4f}"
        )
