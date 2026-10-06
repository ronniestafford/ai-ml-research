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
    seed = random.SystemRandom().randint(0, 2**32 - 1)
    random.seed(seed)  # python random (epsilon etc.)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # seeds CPU; also CUDA/MPS generators

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.set_default_device(device)

    return seed, device


def get_epsilon_greedy_action(
    q_function: QFunction,
    observation: torch.Tensor,
    epsilon: float,
    action_count: int,
):
    is_single_observation = observation.ndim == 1
    if is_single_observation:
        observation = observation.unsqueeze(0)
    observation_count = observation.shape[0]

    # What actions are greedy or random?
    random_mask = torch.rand(observation_count) < epsilon
    greedy_mask = ~random_mask
    action_idx = torch.randint(0, action_count, (observation_count,))

    if greedy_mask.any():
        with torch.no_grad():
            q_values = q_function(observation[greedy_mask])

        action_idx[greedy_mask] = q_values.argmax(dim=-1)

    action_one_hot = one_hot(action_idx, num_classes=action_count)

    if is_single_observation:
        action_one_hot = action_one_hot.squeeze(0)

    return action_one_hot


def get_epsilon(
    global_transition_idx: int,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
) -> float:
    return eps_end + (eps_start - eps_end) * np.exp(
        -global_transition_idx / eps_decay
    )


def optimise(
    q_function: QFunction,
    target_q_function: QFunction,
    optimiser: optim.Optimizer,
    minibatch: TensorDictBase,
    metrics: OptimisationMetrics,
    loss_fn,
    gamma: float,
) -> OptimisationMetrics:

    # Begin autograd graph
    q_values: torch.Tensor = q_function(
        minibatch.get("observation")
    )  # [batch, n_actions]

    # backprop is through max action only
    action_idx = minibatch.get("action").argmax(dim=-1)
    # .gather adds this action to autograd graph
    current_q = q_values.gather(dim=-1, index=action_idx.unsqueeze(1)).squeeze(
        1
    )

    with torch.no_grad():
        successor_q: torch.Tensor = (
            target_q_function(minibatch.get(("next", "observation")))
            .max(dim=-1)
            .values
        )

    target_q: torch.Tensor = minibatch.get(("next", "reward")).squeeze(-1) + (
        gamma
        * successor_q
        * ~minibatch.get(("next", "terminated")).squeeze(-1)
    )

    loss = loss_fn(input=current_q, target=target_q)

    optimiser.zero_grad()
    loss.backward()

    grad_norm: float = torch.nn.utils.clip_grad_norm_(
        q_function.parameters(), max_norm=float(10)
    ).item()
    n_grad_elements = sum(
        p.numel() for p in q_function.parameters() if p.grad is not None
    )

    grad_rms = grad_norm / n_grad_elements**0.5
    return replace(
        metrics,
        loss_sum=metrics.loss_sum + loss.item(),
        grad_norm_sum=metrics.grad_norm_sum + grad_norm,
        grad_rms_sum=metrics.grad_rms_sum + grad_rms,
        optim_step_count=metrics.optim_step_count + 1,
    )


@dataclass(frozen=True)
class OptimisationMetrics:
    loss_sum: float = 0.0
    grad_norm_sum: float = 0.0
    grad_rms_sum: float = 0.0
    optim_step_count: int = 0


@dataclass(frozen=True)
class TrajectoryMetrics:
    transition_idx: int = 0
    discounted_reward: float = 0.0


@dataclass(frozen=True)
class GlobalMetrics:
    total_transition_idx: int = 0
    trajectory_idx: int = 0
    epsilon: float = 1
    seed: int = 0


def print_metrics(
    trajectory_idx: int,
    metrics_global: GlobalMetrics,
    metrics_trajectory: TrajectoryMetrics,
    metrics_optimisation: OptimisationMetrics,
):
    if metrics_optimisation.optim_step_count > 0:
        print(
            f"Trajectory {trajectory_idx} complete │"
            f" discounted reward: {metrics_trajectory.discounted_reward:.4f} │"
            f" transitions: {metrics_trajectory.transition_idx} │"
            f" global steps: {metrics_global.total_transition_idx + 1} │"
            f" avg_loss: "
            f"{metrics_optimisation.loss_sum / metrics_optimisation.optim_step_count:.4f} │"
            f" avg_grad_norm: "
            f"{metrics_optimisation.grad_norm_sum / metrics_optimisation.optim_step_count:.4f} │"
            f" avg_grad_rms: "
            f"{metrics_optimisation.grad_rms_sum / metrics_optimisation.optim_step_count:.6f} │"
            f" epsilon: {metrics_global.epsilon:.4f}"
        )


def main():
    SEED, DEVICE = setup()
    print(f"SEED: {SEED}")

    LEARNING_RATE: float = 1e-3
    GAMMA: float = 0.99
    MINI_BATCH_SIZE: int = 64
    TARGET_Q_FUNC_UPDATE_FREQ: int = 500
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY: float = 10000

    TRAJECTORY_COUNT: int = 500
    STORAGE_MEM_MAX: int = TRAJECTORY_COUNT * 100
    MIN_BUFFER_SIZE: int = MINI_BATCH_SIZE * 3

    HIDDEN_DIM: int = 256

    env = GymEnv("CartPole-v1")
    env.set_seed(SEED)

    STATE_DIM: int = env.observation_spec["observation"].shape[-1]
    ACTION_COUNT: int = env.action_spec.shape[-1]

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

    print("\n BEGIN EXPERIMENT \n")
    metrics_global = GlobalMetrics(
        seed=SEED, epsilon=get_epsilon(0, EPS_START, EPS_END, EPS_DECAY)
    )

    for trajectory_idx in range(TRAJECTORY_COUNT):
        metrics_trajectory = TrajectoryMetrics()
        metrics_optimisation = OptimisationMetrics()

        proto_transition: TensorDictBase = env.reset()
        done = False

        while not done:

            one_hot_action = get_epsilon_greedy_action(
                q_function=q_function,
                observation=proto_transition.get("observation"),
                epsilon=metrics_global.epsilon,
                action_count=ACTION_COUNT,
            )

            # one_hot_action = env.action_spec.rand()
            proto_transition.set("action", one_hot_action)
            full_transition: TensorDictBase = env.step(proto_transition)
            replay_buffer.add(full_transition)

            reward: float = full_transition.get(("next", "reward")).item()
            metrics_trajectory = replace(
                metrics_trajectory,
                discounted_reward=metrics_trajectory.discounted_reward
                + GAMMA**metrics_trajectory.transition_idx * reward,
                transition_idx=metrics_trajectory.transition_idx + 1,
            )

            # Optimise #
            if len(replay_buffer) > MIN_BUFFER_SIZE:
                minibatch = replay_buffer.sample(
                    batch_size=min(len(replay_buffer), MINI_BATCH_SIZE)
                ).to(DEVICE)

                metrics_optimisation = optimise(
                    q_function=q_function,
                    target_q_function=target_q_function,
                    optimiser=optimiser,
                    minibatch=minibatch,
                    metrics=metrics_optimisation,
                    loss_fn=loss_fn,
                    gamma=GAMMA,
                )

                optimiser.step()

            proto_transition = full_transition.get("next")

            if proto_transition.get("done").item():
                print_metrics(
                    trajectory_idx,
                    metrics_global,
                    metrics_trajectory,
                    metrics_optimisation,
                )
                done = True

            if (
                metrics_global.total_transition_idx % TARGET_Q_FUNC_UPDATE_FREQ
                == 0
            ):
                target_q_function.load_state_dict(q_function.state_dict())

            metrics_global = replace(
                metrics_global,
                epsilon=get_epsilon(
                    metrics_global.total_transition_idx,
                    EPS_START,
                    EPS_END,
                    EPS_DECAY,
                ),
                total_transition_idx=metrics_global.total_transition_idx + 1,
                trajectory_idx=trajectory_idx,
            )


if __name__ == "__main__":
    main()
