from functools import partial

from torch import optim
from torch.nn.functional import mse_loss
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import GymEnv

from ai_ml_research.rl.config import ExperimentConfig
from ai_ml_research.rl.dpcme.bellman import model_based_expected_sarsa_target
from ai_ml_research.rl.dpcme.functions import (
    MuFunction,
    PhiFunction,
    QFunctionModelBased,
    ValueFunction,
)
from ai_ml_research.rl.dpcme.phi.optimiser import PhiOptimiser
from ai_ml_research.rl.dpcme.phi.config import (
    PhiConfig,
    PhiOptimisationConfig,
)
from ai_ml_research.rl.dpcme.phi.metrics import (
    PhiOptimisationMetrics,
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
    """
    Learns optimal policy in ~? samples.
    """

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
        trajectory_count=1000,
        # Typical CartPole magnitude per state dimension:
        # positions/angles at their failure thresholds, velocities
        # at typical pre-failure magnitudes.
        state_scale=(2.4, 6.0, 0.42, 6.0),
    )
    config = DPCMEConfig(
        # Discount factor
        gamma=evaluation_gamma,
        # Exploration
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=25000,
        # Replay buffer
        replay_buffer_max_size=20000,
        # Warm-up: no component takes a gradient step until the
        # replay buffer holds more transitions than this.
        warm_up_transition_count=max_transition_count * 2,
    )
    cme_config = CMEConfig(
        # CME function: ψ (state-action features) then W
        state_action_dim=state_dim + action_count,
        psi_space_dim=512,
        # Random Fourier feature map φ: μ regresses onto φ's
        # output space, so phi_space_dim is shared between the
        # two.
        phi_space_dim=256,
        hidden_dim=200,
        hidden_layer_count=2,
        optimisation=CMEOptimisationConfig(
            learning_rate=3e-4,
            # μ is the regression target of every downstream
            # fit: halve its minibatch noise relative to the
            # consumers'.
            mini_batch_size=128,
            max_grad_norm=10.0,
            sgd_step_count=4,
        ),
    )

    value_function_config = ValueFunctionConfig(
        # Dimensionless RBF bandwidth for a StaticPhiFunction;
        # unused by the parametrised PhiFunction built below.
        phi_bandwidth=1.0,
        phi_hidden_dim=200,
        phi_hidden_layer_count=2,
        optimisation=ValueFunctionOptimisationConfig(
            learning_rate=3e-4,
            mini_batch_size=64,
            max_grad_norm=10.0,
            # Plan budget vs model staleness: re-converging w to
            # the fixed point of a stale μ every transition just
            # fits μ's noise (model error is amplified by
            # 1/(1−backup_gamma)). Two damped iterations instead;
            # the blend is per iteration, so the effective
            # commitment per transition is 1−(1−0.05)² ≈ 0.10.
            policy_iteration_count=2,
            sgd_step_count=10,
            # Backup discount used only when fitting v; distinct
            # from the agent's gamma. Long enough a horizon for
            # the optimal policy, short enough to keep the
            # model-error amplification 1/(1−γ) at 50.
            backup_gamma=0.98,
            conservative_update_rate=0.05,
            # Semigradient: the target (a max over actions) is held
            # constant per step, so the fit tracks the Bellman
            # operator's fixed point. The residual_gradient rule
            # instead descends the squared optimal Bellman
            # residual, whose gradient flow is not a contraction.
            update_rule="semigradient",
        ),
    )

    phi_config = PhiConfig(
        # False: φ stays a frozen random map wired directly to
        # its consumers (today's behaviour). True: PhiOptimiser
        # trains φ and the consumers read its EMA target copy.
        train_phi=True,
        optimisation=PhiOptimisationConfig(
            # A tenth of μ's rate: the space must move slower
            # than anything fitting itself to it.
            learning_rate=3e-5,
            mini_batch_size=64,
            sgd_step_count=1,
            max_grad_norm=10.0,
            whitening_coefficient=1e-3,
            # ~1000 optimise() calls (transitions) of memory: an
            # order of magnitude slower than the value fit's
            # ~10-transition blend.
            target_ema_rate=0.999,
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

    # [Performance] Deep networks on GPU. φ enters the CME
    # regression target and the value fit as a constant (both
    # evaluate it through its inference), so like μ it lives on
    # the model device.
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
        hidden_dim=value_function_config.phi_hidden_dim,
        hidden_layer_count=value_function_config.phi_hidden_layer_count,
    ).to(model_device)

    # When trained, consumers (the CME regression target and the
    # value fit) read the optimiser's EMA target copy φ̄, never the
    # fast φ: they see a near-static space and every loop into φ
    # closes through a detach. When not trained, they read φ
    # itself — no copy, no extra tick, the frozen-feature path.
    phi_optimiser: PhiOptimiser | None = None
    if phi_config.train_phi:
        phi_optimiser = PhiOptimiser(
            phi_function=phi_function,
            cme_function=mu_function,
            optimiser=optim.Adam(
                params=phi_function.parameters(),
                lr=phi_config.optimisation.learning_rate,
            ),
            replay_buffer=replay_buffer,
            loss_fn=mse_loss,
            config=phi_config,
            optimisation_metrics=PhiOptimisationMetrics(),
        )
    consumers_phi = (
        phi_optimiser.target_phi if phi_optimiser is not None else phi_function
    )

    # v(s) = w·φ(s): φ enters as a constant, so only the value
    # weights w are trained.
    value_function = ValueFunction(
        phi_function=consumers_phi,
        phi_space_dim=cme_config.phi_space_dim,
    ).to(model_device)

    # q(s,a) = r(s,a) + γ·μ(s,a)·w is defined through the CME: it shares
    # the CME's μ and the value function by reference, both used as
    # constants here, so it owns no parameters of its own.
    q_function = QFunctionModelBased(
        value_function=value_function,
        cme_function=mu_function,
        action_count=action_count,
    )

    value_function_optimiser = ValueFunctionOptimiser(
        value_function=value_function,
        q_function=q_function,
        optimiser=optim.Adam(
            params=value_function.linear_layer.parameters(),
            lr=value_function_config.optimisation.learning_rate,
        ),
        replay_buffer=replay_buffer,
        loss_fn=mse_loss,
        config=value_function_config,
        optimisation_metrics=ValueFunctionOptimisationMetrics(),
        value_target=partial(
            model_based_expected_sarsa_target,
            epsilon=0.0,
        ),
    )

    cme_optimiser = CMEOptimiser(
        cme_function=mu_function,
        phi_function=consumers_phi,
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
        phi_optimiser=phi_optimiser,
    )

    run_experiment(mdp=mdp, agent=agent, config=experiment_config)


if __name__ == "__main__":
    main()
