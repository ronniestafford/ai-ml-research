from torch import optim
from torch.nn.functional import mse_loss
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import GymEnv

from ai_ml_research.rl.config import ExperimentConfig
from ai_ml_research.rl.dpcme.functions import (
    MuFunction,
    PhiFunction,
    QFunctionModelBased,
    ValueFunction,
)
from ai_ml_research.rl.dpcme.value_function.optimiser import (
    ValueFunctionOptimiser,
)
from ai_ml_research.rl.dpcme.value_function.config import (
    ValueFunctionConfig,
    ValueFunctionOptimisationConfig,
)
from ai_ml_research.rl.dpcme.value_function.metrics import (
    ValueFunctionOptimisationMetrics,
)
from ai_ml_research.rl.experiment import run_experiment, setup
from ai_ml_research.rl.dpcme.agent import DPCMEAgent
from ai_ml_research.rl.dpcme.config import DPCMEConfig
from ai_ml_research.rl.dpcme.metrics import (
    DPCMEOptimisationMetrics,
)
from ai_ml_research.rl.dpcme.transition_model.optimiser import (
    CMEOptimiser,
)
from ai_ml_research.rl.dpcme.transition_model.config import (
    CMEConfig,
    CMEOptimisationConfig,
)
from ai_ml_research.rl.dpcme.transition_model.metrics import (
    CMEOptimisationMetrics,
)


def main() -> None:
    # Hardware and seed
    seed, model_device, env_device = setup()

    # Environment MDP
    # env = GymEnv("CartPole-v1", sutton_barto_reward=True)
    mdp = GymEnv("CartPole-v1")
    mdp.set_seed(seed)
    max_transition_count: int = mdp.spec.max_episode_steps
    state_dim: int = mdp.observation_spec["observation"].shape[-1]
    action_count: int = mdp.action_spec.shape[-1]
    # Discount rate for the experiment's evaluated returns metric
    evaluation_gamma: float = 0.99

    experiment_config = ExperimentConfig(
        seed=seed,
        model_device=model_device,
        env_device=env_device,
        gamma=evaluation_gamma,
        trajectory_count=100,
    )
    config = DPCMEConfig(
        # Discount factor
        gamma=0.99,
        # Exploration
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=10000,
        # Replay buffer
        replay_buffer_max_size=20000,
    )
    cme_config = CMEConfig(
        # CME function: ψ (state-action features) then W
        state_action_dim=state_dim + action_count,
        psi_space_dim=64,
        phi_space_dim=64,
        hidden_dim=128,
        hidden_layer_count=2,
        # Random Fourier feature map φ: μ regresses onto φ's output space,
        # so phi_space_dim is shared between the two.
        phi_bandwidth=1.0,
        optimisation=CMEOptimisationConfig(
            learning_rate=3e-4,
            mini_batch_size=64,
            optim_steps_per_transition=10,
            max_grad_norm=10.0,
            warm_up_transition_count=max_transition_count * 2,
        ),
    )

    value_function_config = ValueFunctionConfig(
        # Bellman-target discount, shared with the agent's
        gamma=config.gamma,
        optimisation=ValueFunctionOptimisationConfig(
            learning_rate=1e-3,
            mini_batch_size=64,
            optim_steps_per_transition=10,
            max_grad_norm=10.0,
            warm_up_transition_count=max_transition_count * 2,
        ),
    )

    # [Performance] The environment is usually on the CPU,
    # so keep replay buffer CPU-side. The agent fills it, the
    # transition model samples it.
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            config.replay_buffer_max_size, device=env_device
        ),
        batch_size=cme_config.optimisation.mini_batch_size,
    )

    # [Performance] Deep networks on GPU. φ is frozen but lives on the
    # model device, so the regression target shares μ's device.
    mu_function = MuFunction(
        state_action_dim=cme_config.state_action_dim,
        psi_space_dim=cme_config.psi_space_dim,
        phi_space_dim=cme_config.phi_space_dim,
        hidden_dim=cme_config.hidden_dim,
        hidden_layer_count=cme_config.hidden_layer_count,
    ).to(model_device)

    phi_function = PhiFunction(
        state_dim=state_dim,
        phi_space_dim=cme_config.phi_space_dim,
        bandwidth=cme_config.phi_bandwidth,
    ).to(model_device)

    # v(s) = w·φ(s) in the frozen φ space: only the value weights w
    # are learnable.
    value_function = ValueFunction(
        phi_function=phi_function,
        phi_space_dim=cme_config.phi_space_dim,
    ).to(model_device)

    # q(s,a) = r(s,a) + γ·μ(s,a)·w is defined through the CME: it shares
    # the CME's μ and the value function by reference, both used as
    # constants here, so it owns no parameters of its own.
    q_function = QFunctionModelBased(
        value_function=value_function,
        cme_function=mu_function,
        gamma=value_function_config.gamma,
    )

    value_function_optimiser = ValueFunctionOptimiser(
        value_function=value_function,
        q_function=q_function,
        optimiser=optim.Adam(
            params=value_function.weights.parameters(),
            lr=value_function_config.optimisation.learning_rate,
        ),
        replay_buffer=replay_buffer,
        loss_fn=mse_loss,
        config=value_function_config,
        optimisation_metrics=ValueFunctionOptimisationMetrics(),
    )

    cme_optimiser = CMEOptimiser(
        cme_function=mu_function,
        phi_function=phi_function,
        optimiser=optim.Adam(
            params=mu_function.parameters(),
            lr=cme_config.optimisation.learning_rate,
        ),
        replay_buffer=replay_buffer,
        loss_fn=mse_loss,
        config=cme_config,
        optimisation_metrics=CMEOptimisationMetrics(),
    )

    agent = DPCMEAgent(
        action_count=action_count,
        state_dim=state_dim,
        config=config,
        optimisation_metrics=DPCMEOptimisationMetrics(),
        replay_buffer=replay_buffer,
        cme_optimiser=cme_optimiser,
        value_function_optimiser=value_function_optimiser,
        q_function=q_function,
    )

    run_experiment(mdp=mdp, agent=agent, config=experiment_config)


if __name__ == "__main__":
    main()
