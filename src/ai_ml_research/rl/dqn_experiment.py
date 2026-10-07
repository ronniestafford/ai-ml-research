import copy
import random
from dataclasses import replace

import numpy as np
import torch
from torch import optim
from torch.nn.functional import one_hot, smooth_l1_loss
from torchrl.envs import GymEnv
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from ai_ml_research.rl.agent import QFunction
from ai_ml_research.rl.metrics import (
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
    n_grad_elements: int,
    max_norm: float,
    huber_beta: float,
) -> OptimisationMetrics:

    # Begin autograd graph
    q_values: torch.Tensor = q_function(
        minibatch.get("observation")
    )  # [batch, n_actions]

    # backprop is through max action only
    action_idx = minibatch.get("action").argmax(dim=-1)
    # .gather adds all the actions to autograd graph
    current_q = q_values.gather(dim=-1, index=action_idx.unsqueeze(1)).squeeze(
        1
    )

    with torch.no_grad():
        successor_q: torch.Tensor = (
            target_q_function(minibatch.get(("next", "observation")))
            .max(dim=-1)
            .values
        )

    # "Terminated": There is no successor state if MDP trajectory is terminated
    # e.g. when cart-pole pole is dropped.
    target_q: torch.Tensor = minibatch.get(("next", "reward")).squeeze(-1) + (
        gamma
        * successor_q
        * ~minibatch.get(("next", "terminated")).squeeze(-1)
    )

    # TD error δ = r + γ·max_a' Q_target(s',a') − Q(s,a)
    with torch.no_grad():
        td_error: torch.Tensor = target_q - current_q
        td_error_abs = td_error.abs()
        td_error_element_count: int = td_error.numel()
        td_error_in_huber_count: int = int(
            (td_error_abs < huber_beta).sum().item()
        )

    loss = loss_fn(input=current_q, target=target_q, beta=huber_beta)

    optimiser.zero_grad(set_to_none=True)
    loss.backward()

    # Not really needed has TD error rarely exceeds
    # quadratic regime of Huber loss anyway.
    grad_norm: float = torch.nn.utils.clip_grad_norm_(
        q_function.parameters(), max_norm=max_norm
    ).item()
    post_clip_norm = min(grad_norm, max_norm)
    grad_rms: float = post_clip_norm / n_grad_elements**0.5

    optimiser.step()

    return replace(
        metrics,
        loss_sum=metrics.loss_sum + loss.item(),
        grad_norm_sum=metrics.grad_norm_sum + post_clip_norm,
        grad_rms_sum=metrics.grad_rms_sum + grad_rms,
        optim_step_count=metrics.optim_step_count + 1,
        td_error_sum=metrics.td_error_sum + td_error.sum().item(),
        td_error_abs_sum=metrics.td_error_abs_sum + td_error_abs.sum().item(),
        td_error_sq_sum=metrics.td_error_sq_sum
        + td_error.square().sum().item(),
        td_error_in_huber_count=metrics.td_error_in_huber_count
        + td_error_in_huber_count,
        td_error_element_count=metrics.td_error_element_count
        + td_error_element_count,
    )


def main():
    SEED, DEVICE = setup()
    print(f"SEED: {SEED}")

    LEARNING_RATE: float = 1e-3
    GAMMA: float = 0.99
    MINI_BATCH_SIZE: int = 32
    TARGET_Q_FUNC_UPDATE_FREQ: int = 2000
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
    STORAGE_MEM_MAX: int = TRAJECTORY_COUNT * MAX_TRANSITION_COUNT
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

            proto_transition = full_transition.get("next")

            if metrics_global.optim_step_count > 0 and (
                metrics_global.optim_step_count % TARGET_Q_FUNC_UPDATE_FREQ
                == 0
            ):
                target_q_function.load_state_dict(q_function.state_dict())

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
