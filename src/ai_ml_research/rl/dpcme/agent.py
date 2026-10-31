import random
from dataclasses import replace

import torch

from torch.nn.functional import one_hot
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer

from ai_ml_research.rl.agent import Agent, get_epsilon_exp_decay
from ai_ml_research.rl.dpcme.config import DPCMEConfig
from ai_ml_research.rl.dpcme.functions import QFunctionModelBased
from ai_ml_research.rl.dpcme.metrics import (
    DPCMEOptimisationMetrics,
)
from ai_ml_research.rl.dpcme.transition_model.optimiser import CMEOptimiser
from ai_ml_research.rl.dpcme.value_function.optimiser import (
    ValueFunctionOptimiser,
)
from ai_ml_research.rl.optimisable import Optimisable


class DPCMEAgent(Agent[DPCMEOptimisationMetrics]):
    """Deep Parametic Conditional Mean Embedding (DPCME) agent.

    - Maintains a CME μ:SxA->Fϕ which is
      an estimate for the expected next-state feature ϕ(S').
    - μ(s,a) = Wψ(s,a) where ψ is the state-action feature.

    The transition model and the state-action value function are injected
    Optimisable components: each owns its parameters, optimiser and
    metrics. This agent selects actions with an ε-greedy policy over the
    model-based q function (evaluated as a constant), fills the replay
    buffer the components read, and decides the cadence — one tick of
    each component per transition.
    """

    def __init__(
        self,
        action_count: int,
        state_dim: int,
        config: DPCMEConfig,
        optimisation_metrics: DPCMEOptimisationMetrics,
        replay_buffer: TensorDictReplayBuffer,
        cme_optimiser: CMEOptimiser,
        value_function_optimiser: ValueFunctionOptimiser,
        q_function: QFunctionModelBased,
    ) -> None:
        super().__init__(optimisation_metrics)
        self.action_count = action_count
        self.state_dim = state_dim
        self.config = config
        self.replay_buffer = replay_buffer
        self.cme = cme_optimiser
        self.value_function = value_function_optimiser
        self.q_function = q_function  # evaluated as a constant

        self.model_device = next(q_function.parameters()).device
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

    def sub_optimisables(self) -> tuple[Optimisable, ...]:
        return (self.cme, self.value_function)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        # ε-greedy behaviour policy over the model-based q.
        # [Performance] The ε branch is Python control flow: the coin
        # comes from the Python RNG (seeded by setup). A device-tensor
        # `if` would be an implicit .item() — one sync per act on a
        # GPU env.
        if random.random() < self._epsilon:
            # [Performance] Random action: keep on env device.
            action_idx = torch.randint(
                0,
                self.action_count,
                (),
                device=observation.device,
            )
        else:
            action_idx = self._greedy_action_idx(observation)
        # One-hot to match the env's action spec.
        return one_hot(action_idx, num_classes=self.action_count)

    def _greedy_action_idx(self, observation: torch.Tensor) -> torch.Tensor:
        # Greedy: argmax_a q(s,a). At decision time r(s,a) is unknown
        # (no reward model), so the reward term of the backup is set to
        # zero and actions are ranked by the successor-value term
        # γ·μ(s,a)·w ≈ E[v(S′)|s,a].
        # [Performance] Forward pass on the model device; the final
        # action index returns to the env device.
        with torch.no_grad():
            state = observation.reshape(1, self.state_dim).to(
                self.model_device
            )
            states = state.expand(self.action_count, -1)
            actions = torch.eye(
                self.action_count,
                device=self.model_device,
                dtype=state.dtype,
            )  # one-hot action per candidate a
            rewards = torch.zeros(
                self.action_count,
                device=self.model_device,
                dtype=state.dtype,
            )
            # No realised transition at decision time: a zeros bool
            # flag leaves the survival mask fully open; the state is
            # non-terminal by construction (the env loop just stepped).
            is_terminated = torch.zeros(
                self.action_count,
                device=self.model_device,
                dtype=torch.bool,
            )
            q: torch.Tensor = self.q_function(
                torch.cat((states, actions), dim=-1),
                rewards,
                is_terminated,
            )  # [action_count]
        return q.argmax().to(observation.device)

    def observe(self, transition: TensorDictBase) -> None:
        self.replay_buffer.add(transition)
        self._transition_count += 1

        reward: torch.Tensor = transition.get(("next", "reward"))
        self._update_optimisation_metrics(self.optimisation_metrics, reward)

    def optimise(self) -> None:
        self.cme.optimise()
        self.value_function.optimise()

    def _update_optimisation_metrics(
        self, metrics: DPCMEOptimisationMetrics, reward: torch.Tensor
    ):
        self.optimisation_metrics = replace(
            metrics,
            # [Performance] Device-side accumulation: no .item();
            # format_metrics() converts once per trajectory.
            last_reward=reward,
            reward_element_count=metrics.reward_element_count + 1,
            optim_step_count=metrics.optim_step_count + 1,
            trajectory_step_count=metrics.trajectory_step_count + 1,
        )

    def format_metrics(self) -> str:
        base = super().format_metrics()
        if not base:
            return ""

        # NOT SURE WHAT METRICS SHOULD BE IN HERE!
        metrics = self.optimisation_metrics
        last_reward = float(metrics.last_reward)

        return base + (f" last_reward: {last_reward:+.4f}")
