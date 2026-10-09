import random
from dataclasses import replace

import numpy as np
import torch
from torchrl.envs import GymEnv

from ai_ml_research.rl.agent import Agent
from ai_ml_research.rl.config import ExperimentConfig
from ai_ml_research.rl.metrics import GlobalMetrics, TrajectoryMetrics


def setup() -> tuple[int, torch.device, torch.device]:
    # [Performance] Intra-op parallelism overhead is
    # disproportionate when using small neural nets.
    torch.set_num_threads(1)

    seed = random.SystemRandom().randint(0, 2**32 - 1)
    random.seed(seed)  # python random (epsilon etc.)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # seeds CPU; also CUDA/MPS generators

    # [Performance] Don't use torch.set_default_device; env,
    # replay buffer and trajectory bookkeeping stay on the CPU;
    # Restrict tensors crossing to the compute device only at
    # minibatch's .to(device) call.
    model_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    env_device = torch.device("cpu")

    return seed, model_device, env_device


def run_experiment(
    env: GymEnv,
    agent: Agent,
    config: ExperimentConfig,
) -> GlobalMetrics:
    print("\n BEGIN EXPERIMENT \n")
    metrics_global = GlobalMetrics(seed=config.seed)

    for trajectory_idx in range(config.trajectory_count):
        metrics_trajectory = TrajectoryMetrics()
        agent.on_trajectory_start()

        proto_transition = env.reset()
        done = False

        while not done:
            proto_transition.set(
                "action", agent.act(proto_transition.get("observation"))
            )
            full_transition = env.step(proto_transition)
            agent.observe(full_transition)

            reward: float = full_transition.get(("next", "reward")).item()
            metrics_trajectory = replace(
                metrics_trajectory,
                discounted_reward=metrics_trajectory.discounted_reward
                + config.gamma**metrics_trajectory.transition_count * reward,
                transition_count=metrics_trajectory.transition_count + 1,
            )

            agent.optimise()

            proto_transition = full_transition.get("next")

            # "done" == "truncated" (trajectory exceeded max transition count,
            # a successor state exists) OR "terminated" (trajectory cut short
            # early because of an event e.g. dropped the pole in cart-pole).
            if proto_transition.get("done").item():
                done = True
                agent.on_trajectory_end()
                agent.print_metrics(
                    trajectory_idx,
                    metrics_global,
                    metrics_trajectory,
                )
                metrics_global = replace(
                    metrics_global,
                    trajectory_count=metrics_global.trajectory_count + 1,
                )

            metrics_global = replace(
                metrics_global,
                transition_count=metrics_global.transition_count + 1,
            )

    return metrics_global
