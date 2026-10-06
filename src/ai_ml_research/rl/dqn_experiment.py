import copy
import random
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import optim
from torch.nn.functional import one_hot, smooth_l1_loss
from torchrl.envs import GymEnv
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from ai_ml_research.rl.agent import QFunction


def setup() -> tuple[int, torch.device]:
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.set_default_device(DEVICE)

    SEED = random.SystemRandom().randint(0, 2**32 - 1)
    torch.manual_seed(SEED)
    return SEED, DEVICE


SEED, DEVICE = setup()

print(f"SEED: {SEED}")

LEARNING_RATE = 1e-3
GAMMA = 0.99
MINI_BATCH_SIZE = 64
TARGET_Q_FUNCTION_UPDATE_FREQ = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000

TRAJECTORY_COUNT = 500
STORAGE_MEM_MAX = TRAJECTORY_COUNT * 100
MIN_BUFFER_SIZE = MINI_BATCH_SIZE * 3

HIDDEN_DIM = 256

env = GymEnv("CartPole-v1")
STATE_DIM = env.observation_spec["observation"].shape[-1]
ACTION_COUNT = env.action_spec.shape[-1]


print(f"Action count: {ACTION_COUNT}")
print(f"observation_spec: {env.observation_spec}")
print(f"action_spec: {env.action_spec}")
print(f"reward_spec: {env.reward_spec}")
print(f"done_spec: {env.done_spec}")

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(STORAGE_MEM_MAX),
    batch_size=MINI_BATCH_SIZE,
)

q_function = QFunction(STATE_DIM, ACTION_COUNT, HIDDEN_DIM)
target_q_function = copy.deepcopy(q_function).requires_grad_(False)
target_q_function.eval()

loss_fn = smooth_l1_loss
optimiser = optim.Adam(params=q_function.parameters(), lr=LEARNING_RATE)


def get_epsilon_greedy_action(
    observation: torch.Tensor, epsilon: float, action_count: int
):
    observation_count = observation.ndim
    isSingleObservation = observation_count == 1

    if isSingleObservation:
        observation = observation.unsqueeze(0)

    # What actions are greedy or random?
    random_mask = torch.rand(observation_count) < epsilon
    greedy_mask = ~random_mask
    action_idx = torch.randint(0, action_count, (observation_count,))

    if greedy_mask.any():
        with torch.inference_mode():
            q_values = q_function(observation[greedy_mask])

        action_idx[greedy_mask] = q_values.argmax(dim=-1)

    action_one_hot = one_hot(action_idx, num_classes=action_count).to(
        torch.int64
    )

    if isSingleObservation:
        action_one_hot = action_one_hot.squeeze(0)

    return action_one_hot


def get_epsilon(global_transition_idx: int) -> float:
    return EPS_END + (EPS_START - EPS_END) * np.exp(
        -global_transition_idx / EPS_DECAY
    )


def optimise(transition_batch: TensorDictBase):
    q_values: torch.Tensor = q_function(
        transition_batch.get("observation")
    )  # [batch, n_actions]

    # backprop is through max action only
    action_idx = transition_batch.get("action").argmax(dim=-1)
    # .gather adds to autograd graph
    current_q = q_values.gather(dim=-1, index=action_idx.unsqueeze(1)).squeeze(
        1
    )

    with torch.inference_mode():
        successor_q: torch.Tensor = (
            target_q_function(transition_batch.get(("next", "observation")))
            .max(dim=-1)
            .values
        )

    target_q: torch.Tensor = transition_batch.get(("next", "reward")).squeeze(
        -1
    ) + (
        GAMMA
        * successor_q
        * ~transition_batch.get(("next", "done")).squeeze(-1)
    )

    loss = smooth_l1_loss(input=current_q, target=target_q)

    optimiser.zero_grad()
    loss.backward()

    grad_norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(
        q_function.parameters(), max_norm=float("inf")
    )
    n_grad_elements = sum(
        p.numel() for p in q_function.parameters() if p.grad is not None
    )
    return loss, grad_norm, n_grad_elements


@dataclass(frozen=True)
class TrajectoryMetrics:
    transition_idx: int = 0
    discounted_reward: float = 0.0
    loss_sum: float = 0.0
    grad_norm_sum: float = 0.0
    grad_rms_sum: float = 0.0
    optim_step_count: int = 0


@dataclass(frozen=True)
class GlobalMetrics:
    total_transition_idx: int = 0
    trajectory_idx: int = 0
    epsilon: float = get_epsilon(0)


print("\n BEGIN EXPERIMENT \n")
metrics_global = GlobalMetrics()

for trajectory_idx in range(TRAJECTORY_COUNT):
    metrics = TrajectoryMetrics()
    proto_transition: TensorDictBase = env.reset()
    done = False

    while not done:

        one_hot_action = get_epsilon_greedy_action(
            observation=proto_transition.get("observation"),
            epsilon=metrics_global.epsilon,
            action_count=ACTION_COUNT,
        )

        # one_hot_action = env.action_spec.rand()
        proto_transition.set("action", one_hot_action)

        full_transition: TensorDictBase = env.step(proto_transition)
        replay_buffer.add(full_transition)

        reward: float = full_transition.get(("next", "reward")).item()
        metrics = replace(
            metrics,
            discounted_reward=metrics.discounted_reward
            + GAMMA**metrics.transition_idx * reward,
            transition_idx=metrics.transition_idx + 1,
        )

        # Optimise #
        if len(replay_buffer) > MIN_BUFFER_SIZE:
            transition_batch = replay_buffer.sample(
                batch_size=min(len(replay_buffer), MINI_BATCH_SIZE)
            ).to(DEVICE)

            loss, grad_norm, n_grad_elements = optimise(transition_batch)

            grad_rms = grad_norm / n_grad_elements**0.5
            metrics = replace(
                metrics,
                loss_sum=metrics.loss_sum + loss.item(),
                grad_norm_sum=metrics.grad_norm_sum + grad_norm.item(),
                grad_rms_sum=metrics.grad_rms_sum + grad_rms.item(),
                optim_step_count=metrics.optim_step_count + 1,
            )
            optimiser.step()

        proto_transition = full_transition.get("next")

        if proto_transition.get("done").item():
            if metrics.optim_step_count > 0:
                print(
                    f"Trajectory {metrics_global.trajectory_idx} complete │"
                    f" discounted reward: {metrics.discounted_reward:.4f} │"
                    f" transitions: {metrics.transition_idx} │"
                    f" global steps: {metrics_global.total_transition_idx + 1} │"
                    f" avg_loss: "
                    f"{metrics.loss_sum / metrics.optim_step_count:.4f} │"
                    f" avg_grad_norm: "
                    f"{metrics.grad_norm_sum / metrics.optim_step_count:.4f} │"
                    f" avg_grad_rms: "
                    f"{metrics.grad_rms_sum / metrics.optim_step_count:.6f} │"
                    f" epsilon: {metrics_global.epsilon:.4f}"
                )
            done = True

        if (
            metrics_global.total_transition_idx % TARGET_Q_FUNCTION_UPDATE_FREQ
            == 0
        ):
            target_q_function.load_state_dict(q_function.state_dict())

        metrics_global = replace(
            metrics_global,
            epsilon=get_epsilon(metrics_global.total_transition_idx),
            total_transition_idx=metrics_global.total_transition_idx + 1,
            trajectory_idx=trajectory_idx,
        )
