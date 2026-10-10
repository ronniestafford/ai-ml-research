---
name: scaffold-rl-agent
description: Scaffold a new RL agent package by cloning src/ai_ml_research/rl/template and renaming the agent, config, metrics and experiment to a new name. Use when the user asks to create, add or scaffold a new RL agent/algorithm (e.g. "create a ppo agent", "new agent from the template").
---

# Scaffold a new RL agent

Creates `src/ai_ml_research/rl/<name>/` from `src/ai_ml_research/rl/template/`
with every `Template`/`rl.template` reference renamed, ready for a real
algorithm implementation.

## Inputs

- `<name>` (required): snake_case package name, e.g. `monte_carlo`.
  Ask the user for it if not given. If the user supplies a name that is not
  snake_case (e.g. "MonteCarlo" or "monte-carlo"), convert it to snake_case
  and confirm.
- `<ClassName>` (optional): CamelCase prefix for the generated classes.
  Defaults to the snake_case name CamelCased (`monte_carlo` →
  `MonteCarlo`, giving `MonteCarloAgent`, `MonteCarloConfig`,
  `MonteCarloOptimisationMetrics`).

## Steps

1. Run the scaffold script from the repo root:

   ```bash
   skills/scaffold-rl-agent/scaffold.sh <name> [ClassName]
   ```

   It copies the template (excluding `__pycache__`), then replaces
   `rl.template` → `rl.<name>` in imports and `Template` → `<ClassName>`
   in identifiers and docstrings. It fails without touching anything if the
   target directory already exists or the name is invalid.

2. Verify (read-only, never install anything):

   ```bash
   python3 -m py_compile src/ai_ml_research/rl/<name>/*.py
   grep -rn "Template\|rl\.template" src/ai_ml_research/rl/<name> || echo clean
   ```

3. Report the created files and tell the user the entry point is
   `python -m ai_ml_research.rl.<name>.experiment`.

## After scaffolding — implementing the algorithm

The scaffold is a random-policy template that exercises the `Agent`
contract. When implementing the real algorithm:

- `agent.py`: keep the `Agent` subclass contract (`act`, `observe`,
  `optimise`, `format_metrics`); rewrite the class docstring to describe
  the actual algorithm. Keep the device-side vs CPU-side metric idioms.
- `config.py`: add algorithm hyperparameters to `<ClassName>Config`
  (it extends `AgentConfig`; `gamma` there is the Bellman-target discount,
  NOT the evaluation discount which lives in `ExperimentConfig`).
- `metrics.py`: new fields MUST declare a default —
  `Agent.on_trajectory_start` resets every field to `field.default`.
- `experiment.py`: hyperparameters live here; note it carries two distinct
  discounts: `evaluation_gamma` (→ `ExperimentConfig`, returns metric) and
  the agent-config `gamma` (→ Bellman target). Keep them separate.

Never run `uv`/`pip` to verify — use `python3 -m py_compile` only
(see AGENTS.md).
