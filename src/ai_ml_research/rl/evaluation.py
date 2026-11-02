from dataclasses import dataclass
from functools import partial
from pickle import dumps, loads
from statistics import fmean, stdev
from typing import Protocol

import torch

from torchrl.envs import GymEnv, ParallelEnv


class GreedyPolicy(Protocol):
    """Batched, inference-only action selection.

    The evaluation counterpart of the runner's act()/observe()
    loop: one forward pass over the whole worker batch, no
    exploration, no replay-buffer writes, no metric updates —
    an agent satisfies this through its act_batch().
    """

    def act_batch(self, observations: torch.Tensor) -> torch.Tensor:
        """[batch, ...state] -> actions as env.step() expects."""


@dataclass(frozen=True)
class EvaluationConfig:
    # Number of evaluation episodes per evaluation, run once and
    # concurrently: one parallel env worker process per episode
    # (the envs step asynchronously in the workers while the
    # main process does batched inference). The workers persist
    # across evaluations; one evaluation runs per training
    # trajectory.
    episode_count: int

    # Successive evaluations advance the worker seed by one:
    # workers are independent draws within an evaluation, and
    # each evaluation samples fresh episodes.
    base_seed: int = 0


@dataclass(frozen=True)
class EvaluationResult:
    """Raw per-episode outputs of one evaluation."""

    discounted_returns: tuple[float, ...]
    step_counts: tuple[int, ...]

    @property
    def episode_count(self) -> int:
        return len(self.discounted_returns)

    @property
    def mean_return(self) -> float:
        return fmean(self.discounted_returns)

    @property
    def sem_return(self) -> float:
        return _sem(self.discounted_returns)

    @property
    def mean_steps(self) -> float:
        return fmean(self.step_counts)

    @property
    def sem_steps(self) -> float:
        return _sem(self.step_counts)


def _sem(samples: tuple[float, ...]) -> float:
    # Standard error of the mean over independent episodes.
    if len(samples) < 2:
        return float("nan")
    return stdev(samples) / len(samples) ** 0.5


def make_eval_envs(
    mdp: GymEnv,
    config: EvaluationConfig,
) -> ParallelEnv:
    """The persistent evaluation worker pool: spawned once per
    experiment, evaluated once per training trajectory.

    ParallelEnv needs a *constructor* it can run in each worker
    (env instances may not travel between processes), so the
    template is pickled once and reconstructed per worker: every
    worker owns an independent clone of the training env.
    """
    return ParallelEnv(
        num_workers=config.episode_count,
        create_env_fn=partial(loads, dumps(mdp)),
    )


def evaluate_greedy(
    policy: GreedyPolicy,
    envs: ParallelEnv,
    gamma: float,
    base_seed: int = 0,
    evaluation_idx: int = 0,
) -> EvaluationResult:
    """Evaluate the current greedy policy on one concurrent batch
    of episodes (one per worker) and return the raw per-episode
    results.

    The workers were cloned from the training env (their own
    processes, their own episodes); the policy is only read.
    Episode returns use the runner's discounting, gamma**t * r_t,
    so the numbers are comparable with the training curve. The
    worker seed advances with evaluation_idx: each evaluation
    samples fresh episodes.
    """
    episode_count = envs.num_workers
    envs.set_seed(base_seed + evaluation_idx)

    returns = torch.zeros(episode_count)
    steps = torch.zeros(episode_count, dtype=torch.long)
    # A worker's episode ends at its first done (termination or
    # truncation); later steps belong to the worker's auto-reset
    # episode and are not counted.
    active = torch.ones(episode_count, dtype=torch.bool)

    with torch.no_grad():
        td = envs.reset()
        while active.any():
            actions = policy.act_batch(td.get("observation"))
            full_transition = envs.step(td.set("action", actions))

            reward = full_transition.get(("next", "reward")).squeeze(-1)
            done = full_transition.get(("next", "done")).squeeze(-1)

            # Discount by the steps taken so far, count the
            # terminal transition in the step total.
            returns[active] += gamma ** steps[active] * reward[active]
            steps[active] += 1
            active &= done.logical_not()

            td = full_transition.get("next")

    return EvaluationResult(
        discounted_returns=returns.tolist(),
        step_counts=steps.tolist(),
    )


def print_evaluation_metrics(
    result: EvaluationResult,
    total_transitions: int,
) -> EvaluationResult:
    """One line per evaluation, kept separate from the training
    trajectory metrics (own line, own dataclass, no share with
    the optimisation accumulators)."""
    print(
        f"Evaluation @ {total_transitions} transitions │"
        f" episodes: {result.episode_count} │"
        f" eval return (mean ± sem):"
        f" {result.mean_return:.4f} ± {result.sem_return:.4f} │"
        f" eval steps (mean ± sem):"
        f" {result.mean_steps:.1f} ± {result.sem_steps:.1f}"
    )
    return result
