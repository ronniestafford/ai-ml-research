from torch import optim
from torch.nn.functional import smooth_l1_loss
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import GymEnv

from ai_ml_research.rl.config import ExperimentConfig
from ai_ml_research.rl.dqn.agent import (
    DQNAgent,
    QFunction,
)
from ai_ml_research.rl.dqn.config import DQNConfig
from ai_ml_research.rl.dqn.metrics import DQNOptimisationMetrics
from ai_ml_research.rl.experiment import run_experiment, setup


def main() -> None:
    """
    Learns optimal policy in ~50,000-70,000 samples.
    """

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
        trajectory_count=1000,
    )
    config = DQNConfig(
        # Discount rate for the Bellman target
        gamma=0.99,
        # Optimisation
        learning_rate=3e-4,
        mini_batch_size=64,
        optim_steps_per_transition=1,
        # Q Functions
        hidden_dim=128,
        hidden_layer_count=2,
        # Target Q function
        target_soft_update=0.005,
        optim_steps_per_target_update=3,
        # Loss and gradient clipping
        huber_beta=1.0,
        max_grad_norm=10.0,
        # Exploration
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=10000,
        # Replay buffer
        replay_buffer_max_size=20000,
        warm_up_transition_count=max_transition_count * 2,
    )

    # [Performance] The environment is usually on the CPU,
    # so keep replay buffer CPU-side.
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            config.replay_buffer_max_size, device=env_device
        ),
        batch_size=config.mini_batch_size,
    )

    # [Performance] Deep networks on GPU.
    q_function = QFunction(
        state_dim=state_dim,
        action_count=action_count,
        hidden_dim=config.hidden_dim,
        hidden_layer_count=config.hidden_layer_count,
    ).to(model_device)

    agent = DQNAgent(
        q_function=q_function,
        optimiser=optim.Adam(
            params=q_function.parameters(), lr=config.learning_rate
        ),
        replay_buffer=replay_buffer,
        loss_fn=smooth_l1_loss,
        action_count=action_count,
        config=config,
        optimisation_metrics=DQNOptimisationMetrics(),
    )

    run_experiment(env=env, agent=agent, config=experiment_config)


if __name__ == "__main__":
    main()
