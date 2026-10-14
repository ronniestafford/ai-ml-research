from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import GymEnv

from ai_ml_research.rl.config import ExperimentConfig
from ai_ml_research.rl.experiment import run_experiment, setup
from ai_ml_research.rl.template.agent import TemplateAgent
from ai_ml_research.rl.template.config import TemplateConfig
from ai_ml_research.rl.template.metrics import (
    TemplateOptimisationMetrics,
)


def main() -> None:
    # Hardware and seed
    seed, model_device, env_device = setup()

    # Environment MDP
    env = GymEnv("CartPole-v1")
    env.set_seed(seed)
    max_transition_count: int = env.spec.max_episode_steps
    state_dim: int = env.observation_spec["observation"].shape[-1]
    action_count: int = env.action_spec.shape[-1]
    # Discount rate for the experiment's evaluated returns metric
    evaluation_gamma: float = 0.99

    experiment_config = ExperimentConfig(
        seed=seed,
        model_device=model_device,
        env_device=env_device,
        gamma=evaluation_gamma,
        trajectory_count=5,
    )
    config = TemplateConfig(
        # Discount factor
        gamma=0.99,
        # Exploration
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=10000,
        # Optimisation
        learning_rate=3e-4,
        mini_batch_size=64,
        optim_steps_per_transition=1,
        # Replay buffer
        replay_buffer_max_size=20000,
        warm_up_transition_count=max_transition_count * 2,
    )

    agent = TemplateAgent(
        action_count=action_count,
        state_dim=state_dim,
        config=config,
        optimisation_metrics=TemplateOptimisationMetrics(),
        # [Performance] The environment is usually on the CPU,
        # so keep replay buffer CPU-side.
        replay_buffer=TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                config.replay_buffer_max_size, device=env_device
            ),
            batch_size=config.mini_batch_size,
        ),
    )

    run_experiment(env=env, agent=agent, config=experiment_config)


if __name__ == "__main__":
    main()
