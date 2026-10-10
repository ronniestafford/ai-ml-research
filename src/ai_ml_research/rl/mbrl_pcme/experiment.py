from torchrl.envs import GymEnv

from ai_ml_research.rl.config import ExperimentConfig
from ai_ml_research.rl.experiment import run_experiment, setup
from ai_ml_research.rl.mbrl_pcme.agent import MbrlPcmeAgent
from ai_ml_research.rl.mbrl_pcme.config import MbrlPcmeConfig
from ai_ml_research.rl.mbrl_pcme.metrics import (
    MbrlPcmeOptimisationMetrics,
)


def main() -> None:
    # Hardware and seed
    seed, model_device, env_device = setup()

    # Environment MDP
    env = GymEnv("CartPole-v1")
    env.set_seed(seed)
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
    config = MbrlPcmeConfig(
        # Discount rate for the agent's Bellman target
        gamma=0.99,
        # Unused by the random policy; placeholder values to be replaced
        # by a real agent's hyperparameters.
        learning_rate=3e-4,
        mini_batch_size=64,
        optim_steps_per_transition=1,
    )

    agent = MbrlPcmeAgent(
        action_count=action_count,
        config=config,
        optimisation_metrics=MbrlPcmeOptimisationMetrics(),
    )

    run_experiment(env=env, agent=agent, config=experiment_config)


if __name__ == "__main__":
    main()
