---
name: scaffold-rl-agent
description: Scaffold a new RL agent package by cloning src/ai_ml_research/rl/template (code/ + paper/) and renaming the agent, config, metrics and experiment to a new name. Use when the user asks to create, add or scaffold a new RL agent/algorithm (e.g. "create a ppo agent", "new agent from the template").
---

# Scaffold a new RL agent

Every algorithm under `src/ai_ml_research/rl/<name>/` follows a two-folder
layout:

- `code/` — the implementation (package; entry point
  `python -m ai_ml_research.rl.<name>.code.experiment`, imports under
  `ai_ml_research.rl.<name>.code.*`).
- `paper/` — the algorithm's associated paper, scaffolded from one of the
  LaTeX templates under `template/paper/` (default `arxiv-style`; ask the
  user which if unspecified):
  - `arxiv-style` — George Kour's arxiv-style preprint style
    (https://github.com/kourgeorge/arxiv-style, MIT); credits upstream via
    the `main.tex` header comment and a title-page footnote citing
    `kour_arxiv_style`.
  - `finai-arxiv` — The Fin AI's arXiv template
    (https://github.com/The-FinAI/arxiv-template, MIT); credits upstream
    via a title-page footnote (`\footnotemark` in `\title` +
    `\footnotetext` after `\maketitle`) citing `finai_arxiv_template`.
    The Fin AI name and logo are excluded from the MIT grant and must not
    be used, so the logo has been removed from the bundled style and it uses generic
    placeholders.

  License notices (`License.txt`/`LICENSE`), `preview.png` and `README.md`
  stay in `template/paper/` and are never copied to new papers; both
  templates are also mirrored in `CITATION.bib`.
  Replace/add the PDF and
  record the paper's origin and attribution in the document per the
  repository citation policy (AGENTS.md).
- `citation.bib` — per-algorithm citations at the algorithm root
  (copied from `template/citation.bib`, kept distinct from the paper's
  `references.bib`): at minimum `ai_ml_research` (the repository) and
  `ronnie_stafford_thesis_2020` (the PhD thesis), wired into the paper
  via `\bibliography{references,../citation}`, and MUST be extended
  with the algorithm's original publication where one is documented.

This skill creates `src/ai_ml_research/rl/<name>/` from
`src/ai_ml_research/rl/template/` with every `Template`/`rl.template`
reference renamed, ready for a real algorithm implementation.

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
   skills/scaffold-rl-agent/scaffold.sh <name> [ClassName] [paper-template]
   ```

   It copies `template/code/` → `<name>/code/`,
   `template/paper/<paper-template>/` → `<name>/paper/` (default template
   `arxiv-style`, excluding `__pycache__`, license notices, previews and
   readmes) and `template/citation.bib` → `<name>/citation.bib`, then replaces
   `rl.template`/`rl.template.code` → `rl.<name>.code` in imports,
   `Template` → `<ClassName>` in identifiers and docstrings, and the
   `<AlgorithmName>`/`<name>` placeholders in the paper's `.tex` and
   `.bib` files (including `citation.bib`). It
   fails without touching anything if the target directory already exists
   or the name or paper template is invalid.

2. Verify (read-only, never install anything):

   ```bash
   python3 -m py_compile src/ai_ml_research/rl/<name>/code/*.py
   grep -rn "Template\|rl\.template" src/ai_ml_research/rl/<name> || echo clean
   ```

3. Report the created files and tell the user the entry point is
   `python -m ai_ml_research.rl.<name>.code.experiment`.

## After scaffolding — implementing the algorithm

The scaffold is a random-policy template that exercises the `Agent`
contract. When implementing the real algorithm:

- `code/agent.py`: keep the `Agent` subclass contract (`act`, `observe`,
  `optimise`, `format_metrics`); rewrite the class docstring to describe
  the actual algorithm. Keep the device-side vs CPU-side metric idioms.
  `format_metrics` MUST chain `super().format_metrics()` and return `""`
  when the base is empty — the base prints `avg_loss`/`avg_grad_norm`/
  `avg_grad_rms`, so overriding without chaining silently drops them.
- `code/agent.py` (trainable sub-components): if the algorithm owns a second
  thing with parameters, add an `Optimisable` component for it (see
  "Adding an Optimisable component" below), inject it in `__init__` and
  list it in `sub_optimisables()`. Trajectory resets, `optimise()` cadence
  and metric printing then cover it automatically.
- `code/config.py`: add algorithm hyperparameters to `<ClassName>Config`
  (it extends `AgentConfig`; `gamma` there is the Bellman-target discount,
  NOT the evaluation discount which lives in `ExperimentConfig`).
  Optimiser hyperparameters live in `OptimisationConfig`
  (`learning_rate`, `mini_batch_size`, `optim_steps_per_transition`),
  embedded as an `optimisation` field on whichever config actually takes
  the gradient step — the agent's, or its component's.
- `code/metrics.py`: new fields MUST declare a default —
  `Optimisable.on_trajectory_start` resets every field to `field.default`.
  Subclass `OptimisationMetrics` to inherit the loss/grad accumulators.
- `code/experiment.py`: hyperparameters live here, including the nested
  `optimisation=OptimisationConfig(...)`; note it carries two distinct
  discounts: `evaluation_gamma` (→ `ExperimentConfig`, returns metric) and
  the agent-config `gamma` (→ Bellman target). Keep them separate.
- `paper/`: holds the algorithm's paper; develop it from the copied LaTeX
  skeleton (keep the upstream attribution header and footnote or
  acknowledgments credit) and record the algorithm's origin and citations
  there per the repository citation policy (AGENTS.md).

## Adding an Optimisable component

For any trainable sub-component (transition model, value function,
encoder, ...), follow the two-module split used by
`rl/dpcme/code/transition_model/` (μ / CME) and `rl/dpcme/code/value_function/`
(v / ValueFunctionOptimiser):

1. Subpackage: `rl/<name>/code/<component>/` with `__init__.py`, the
   component module, `config.py` and `metrics.py`.
2. Pure `nn.Module`s stay dumb — `forward` plus `get_param_count`. The
   optimiser, loss, replay buffer, config and metrics are injected into
   the `Optimisable` wrapper, never built by the module. Frozen
   collaborators (e.g. φ) are injected by reference and shared, not
   re-created; a collaborator trained elsewhere but used as a constant
   here (e.g. the CME's μ, shared with the value function) is likewise
   shared and evaluated under `torch.no_grad()`, and its parameters are
   excluded from this component's optimiser and grad counts.
3. `<Component>(Optimisable[<Component>OptimisationMetrics])`: pass the
   metrics to `super().__init__(metrics, name=...)` — `name` labels the
   component on the shared metric line. Set `model_device` and
   `_grad_element_count` from the learnable parameters (a frozen
   submodule's parameters have `requires_grad=False`; count/filter them
   out when building the optimiser and the grad RMS).
4. `optimise()` gates on `warm_up_transition_count` and samples the
   buffer with `.to(self.model_device)` — the batch is committed to the
   GPU there and nowhere else. `_optimise_step` runs graph → `loss.backward()`
   → `clip_grad_norm_` → `step()`, then accumulates metrics as device-side
   scalars via `dataclasses.replace()`; the only host sync happens in
   `format_metrics`, which MUST chain `super().format_metrics()` and
   return `""` when the base is empty.
5. `config.py`: embed `optimisation=<Component>OptimisationConfig(...)`
   (`learning_rate`, `mini_batch_size`, `optim_steps_per_transition`,
   `max_grad_norm`, `warm_up_transition_count`) — the `OptimisationConfig`
   lives with whichever Optimisable takes the gradient step.
   `metrics.py`: every new field MUST declare a default
   (`Optimisable.on_trajectory_start` resets fields to `field.default`).
6. Wiring: the agent takes the component in `__init__`, returns it from
   `sub_optimisables()` and drives its cadence in `optimise()`.
   `experiment.py` builds the modules, moves them `.to(model_device)`,
   then constructs the Optimisable wrapper with an injected optimiser over
   the learnable parameters only.

Never run `uv`/`pip` to verify — use `python3 -m py_compile` only
(see AGENTS.md).
