import copy
import random
from dataclasses import replace

import numpy as np
import torch
from torch import optim
from torch.nn.functional import smooth_l1_loss
from torchrl.envs import GymEnv
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from ai_ml_research.rl.dqn.agent import (
    QFunction,
    get_epsilon_greedy_action,
    optimise,
    soft_update_target,
)
from ai_ml_research.rl.dqn.metrics import (
    GlobalMetrics,
    OptimisationMetrics,
    TrajectoryMetrics,
    print_metrics,
)


def setup() -> tuple[int, torch.device]:
    seed = random.SystemRandom().randint(0, 2**32 - 1)
    random.seed(seed)  # python random (epsilon etc.)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # seeds CPU; also CUDA/MPS generators

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # device = torch.device("cpu")
    torch.set_default_device(device)

    return seed, device


def get_epsilon(
    global_transition_idx: int,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
) -> float:
    return eps_end + (eps_start - eps_end) * np.exp(
        -global_transition_idx / eps_decay
    )


def main():
    SEED, DEVICE = setup()

    LEARNING_RATE: float = 1e-3
    GAMMA: float = 0.99
    MINI_BATCH_SIZE: int = 64
    OPTIM_STEPS_PER_TRANSITION: int = 3
    TARGET_Q_FUNC_UPDATE_FREQ: int = 2000  # was 2000
    TAU: float = 0.5
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY: float = 10000

    HIDDEN_DIM: int = 128
    MAX_NORM: float = 10.0
    HUBER_BETA: float = 1.0

    env = GymEnv("CartPole-v1")
    env.set_seed(SEED)
    MAX_TRANSITION_COUNT: int = env.spec.max_episode_steps
    STATE_DIM: int = env.observation_spec["observation"].shape[-1]
    ACTION_COUNT: int = env.action_spec.shape[-1]

    TRAJECTORY_COUNT: int = 1500
    STORAGE_MEM_MAX: int = TRAJECTORY_COUNT * 50
    MIN_BUFFER_SIZE: int = MAX_TRANSITION_COUNT * 5

    print(f"MAX_TRANSITION_COUNT: {MAX_TRANSITION_COUNT}")
    print(f"Action count: {ACTION_COUNT}")
    print(f"observation_spec: {env.observation_spec}")
    print(f"action_spec: {env.action_spec}")
    print(f"reward_spec: {env.reward_spec}")
    print(f"done_spec: {env.done_spec}")

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(STORAGE_MEM_MAX, device=DEVICE),
        batch_size=MINI_BATCH_SIZE,
    )

    q_function = QFunction(STATE_DIM, ACTION_COUNT, HIDDEN_DIM)
    target_q_function = copy.deepcopy(q_function).requires_grad_(False)
    target_q_function.eval()

    n_grad_elements: int = q_function.get_param_count()
    loss_fn = smooth_l1_loss
    optimiser = optim.Adam(params=q_function.parameters(), lr=LEARNING_RATE)

    print("\n BEGIN EXPERIMENT \n")
    metrics_global = GlobalMetrics(
        seed=SEED, epsilon=get_epsilon(0, EPS_START, EPS_END, EPS_DECAY)
    )

    for trajectory_idx in range(TRAJECTORY_COUNT):
        metrics_trajectory = TrajectoryMetrics()
        metrics_optimisation = OptimisationMetrics()
        metrics_global = replace(
            metrics_global,
            epsilon=get_epsilon(
                metrics_global.transition_count,
                EPS_START,
                EPS_END,
                EPS_DECAY,
            ),
        )

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
                + GAMMA**metrics_trajectory.transition_count * reward,
                transition_count=metrics_trajectory.transition_count + 1,
            )

            # Optimise #
            if len(replay_buffer) > MIN_BUFFER_SIZE:
                for _ in range(OPTIM_STEPS_PER_TRANSITION):
                    minibatch = replay_buffer.sample(
                        batch_size=min(len(replay_buffer), MINI_BATCH_SIZE)
                    )

                    metrics_optimisation = optimise(
                        q_function=q_function,
                        target_q_function=target_q_function,
                        optimiser=optimiser,
                        minibatch=minibatch,
                        metrics=metrics_optimisation,
                        loss_fn=loss_fn,
                        gamma=GAMMA,
                        n_grad_elements=n_grad_elements,
                        max_norm=MAX_NORM,
                        huber_beta=HUBER_BETA,
                    )

                    metrics_global = replace(
                        metrics_global,
                        optim_step_count=metrics_global.optim_step_count + 1,
                    )

                    if (
                        metrics_global.optim_step_count
                        % TARGET_Q_FUNC_UPDATE_FREQ
                        == 0
                    ):
                        soft_update_target(
                            q_function=q_function,
                            target_q_function=target_q_function,
                            tau=TAU,
                        )

            proto_transition = full_transition.get("next")

            # "done" == "truncated" (trajectory exceeded max transition count,
            # a successor state exists) OR "terminated" (trajectory cut short
            # early because of an even e.g. dropped the pole in cart-pole).
            if proto_transition.get("done").item():
                done = True
                print_metrics(
                    trajectory_idx,
                    metrics_global,
                    metrics_trajectory,
                    metrics_optimisation,
                )
                metrics_global = replace(
                    metrics_global,
                    trajectory_count=metrics_global.trajectory_count + 1,
                )

            metrics_global = replace(
                metrics_global,
                transition_count=metrics_global.transition_count + 1,
            )


if __name__ == "__main__":
    main()
