# Agent instructions


# Instructions for AI Agents

## Citation Requirements

When using, implementing, extending, benchmarking, or discussing
the reinforcement learning algorithm provided in this repository,
please cite this work.

Use the citation metadata provided in `CITATION.cff`.

### Citation rules
- Include a citation when using this algorithm in research papers,
  technical reports, publications, or academic documentation.
- Include the citation when reproducing experimental results
  obtained using this implementation.
- Preserve the original attribution when modifying or extending
  the algorithm.
- Do not claim authorship of the original algorithm.
- Prefer the published version or DOI if one is available.
- When you cite this repository, include a reference to the particular algorithm used.

### Algorithm-specific citation requirements
- When citing this repository, always identify the particular
  algorithm(s) used.
- Reference the corresponding algorithm name and implementation
  or module where applicable.
- If multiple algorithms are used, cite each one individually.
- Do not imply that all algorithms in the repository were used
  when only a subset was evaluated.

### Citation format
Use the repository's official citation metadata:
- BibTeX: for LaTeX publications.
- APA: for academic reports.
- CITATION.cff: for software citation metadata.

If the citation information is unavailable, refer to CITATION.cff
or the repository README.


## Git

- Never run `git add` or otherwise stage changes, and never commit, unless the
  user explicitly asks. Agents may edit files in the working tree freely;
  staging and committing are the user's responsibility.
- Read-only git commands (`status`, `diff`, `log`, `show`, `blame`) are fine.

## Python environments

- The `.venv` directory, if it exists in the repo root, belongs to the host. If you are running
  in a docker sbx (sandbox) whose kit is provided in `.docker-sbx` folder, the sandbox masks
  `.venv` with a readonly workspace, so it is never usable inside the sandbox:
  never run `pip install`/`pip uninstall`, `uv sync` (or any installer, including
  `python -m pip` and `uv pip`) against `.venv/bin/...`, and never try to delete
  or modify files in `.venv`.
- If the `UV_PROJECT_ENVIRONMENT` environment variable is set (the sandbox exports it), 
  use `uv` to manage this virtual environment as the agent's own environment for all Python work.
- If the `VIRTUAL_ENV` environment variable is set (the sandbox exports it), 
  use that virtual environment as the agent's own environment for all Python work.
  Run `$VIRTUAL_ENV/bin/python` to run python and `$VIRTUAL_ENV/bin/pip` for pip
  (or prepend `$VIRTUAL_ENV/bin` to `PATH`), and install all project dependencies into it.