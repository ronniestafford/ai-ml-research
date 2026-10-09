# Spec: Agent-agnostic RL experiment runner

## Goal

`experiment.py` currently hard-codes DQN: it builds the q-networks, replay
buffer and optimiser, runs the epsilon schedule and calls DQN-specific
`optimise`/`soft_update_target` functions from inside the environment loop.
Refactor so the loop is generic over any agent, with DQN as the first
implementation behind a small interface. No behaviour change for the DQN run
(same loop semantics, same printed metric line, same seeding).

## Package layout

```
rl/
├── agent.py        # Agent ABC (the contract)
├── config.py       # ExperimentConfig, AgentConfig (frozen dataclasses)
├── metrics.py      # OptimisationMetrics, TrajectoryMetrics, GlobalMetrics
│                   #   (data types only; printing lives on the Agent)
├── experiment.py   # setup() and run_experiment()  (the generic loop)
└── dqn/
    ├── agent.py    # DQNAgent(Agent), DQNOptimisationMetrics,
    │               #   get_epsilon_greedy_action, soft_update_target
    ├── config.py   # DQNConfig(AgentConfig)
    └── experiment.py  # thin main(): env + collaborators + configs -> run_experiment()
```

## Agent contract (`rl/agent.py`)

```python
class Agent(ABC):
    def __init__(self, optimisation_metrics: OptimisationMetrics):
        self.optimisation_metrics = optimisation_metrics  # injected

    @abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def observe(self, transition: TensorDictBase) -> None: ...
    @abstractmethod
    def optimise(self) -> None: ...                       # accumulates into self.optimisation_metrics

    def on_trajectory_start(self) -> None:                # resets this trajectory's metrics
        ...
    def on_trajectory_end(self) -> None: ...              # default no-op

    def print_metrics(self, trajectory_idx, metrics_global,
                      metrics_trajectory) -> None: ...    # template method
    def format_metrics(self) -> str:                      # agent suffix line
        return ""
```

Semantics:
- `act` returns an action in env format (e.g. one-hot); exploration strategy
  (epsilon schedule) lives entirely in the agent.
- `observe` is called once per environment step; the agent decides whether to
  store the transition (off-policy buffer) or accumulate it (on-policy).
- `optimise` is called once per environment step. The agent owns the warm-up
  check and `optim_steps_per_transition`, and accumulates into
  `self.optimisation_metrics`. On-policy agents no-op here and update in
  `on_trajectory_end` instead.
- `on_trajectory_start` resets the per-trajectory metric accumulators (base
  implementation zeros every field except the cumulative `optim_step_count`,
  generically via `dataclasses.fields`); DQN overrides it to also recompute
  epsilon from its own transition count.
- The runner never inspects agent internals and never sees an
  `OptimisationMetrics` — the metric *type* is agent-owned, so a PPO agent
  can inject a completely different `OptimisationMetrics` subclass without
  touching `rl/`.

## Dependency injection

`DQNAgent.__init__` receives its collaborators from the call site
(`dqn/experiment.py`), which already knows the env specs and devices:

```python
DQNAgent(
    q_function=QFunction(...).to(MODEL_DEVICE),
    optimiser=optim.Adam(q_function.parameters(), lr=...),
    replay_buffer=TensorDictReplayBuffer(...),
    loss_fn=smooth_l1_loss,
    action_count=ACTION_COUNT,
    config=DQNConfig(...),
    optimisation_metrics=DQNOptimisationMetrics(),
)
```

The target network is derived internally
(`copy.deepcopy(q_function)`) — an implementation detail of DQN, not a
collaborator. Future agents take whatever collaborators they need
(e.g. PPO: actor, critic, GAE lambda); the base class is not polluted.

## Metrics

- Optimisation metrics are **agent-owned**: the concrete instance is
  injected into `Agent.__init__` and accumulated inside `optimise()`; the
  runner never sees it. A PPO agent injects its own `OptimisationMetrics`
  subclass with zero changes in `rl/`.
- `OptimisationMetrics` (generic, frozen): `optim_step_count` (cumulative),
  `trajectory_step_count` (reset per trajectory), `loss_sum`,
  `grad_norm_sum`, `grad_rms_sum`. Sums stay device-side; the single host
  sync still happens once per trajectory, in `Agent.print_metrics`.
- `DQNOptimisationMetrics(OptimisationMetrics)` adds the TD-error fields.
- `GlobalMetrics` carries only `trajectory_count`, `transition_count`,
  `seed` — no optimisation state (query it from the agent, e.g.
  `agent.optimisation_metrics`).
- `Agent.print_metrics(trajectory_idx, metrics_global, metrics_trajectory)`
  is a template method: prints the generic line from
  `self.optimisation_metrics`, then appends `self.format_metrics()` — the
  DQN suffix carries `avg_td_error`, `td_error_rms`, `|td_error|<beta` and
  `epsilon`, preserving the old line layout. Printing is gated on
  `trajectory_step_count > 0` (same as the old `optim_step_count > 0`).

## Config

- `ExperimentConfig(seed, model_device, env_device, gamma, trajectory_count)`
  — consumed by the runner.
- `AgentConfig(gamma, learning_rate, mini_batch_size,
  optim_steps_per_transition)` — base hyperparameters.
- `DQNConfig(AgentConfig)` adds hidden dims, target-update cadence/tau,
  warm-up, Huber beta, grad-norm clip, epsilon schedule.

## Runner (`run_experiment(env, agent, config) -> GlobalMetrics`)

Per trajectory: `on_trajectory_start` (resets per-trajectory metrics) ->
step loop {`act`, `env.step`, `observe`, discounted-reward bookkeeping,
`optimise`} -> on `done`: `on_trajectory_end` -> `agent.print_metrics` ->
update `GlobalMetrics`. Returns final `GlobalMetrics` (trajectories/
transitions only) for programmatic checks; optimisation state is queried
from the agent, never returned by the runner.

## Non-goals

- No second agent implementation (abstraction validated against DQN only).
- No hyperparameter-sweep plumbing beyond dataclass configs.
- `setup()` (seeding/devices) moves to `rl/experiment.py` unchanged.
