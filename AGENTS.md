# Agent instructions

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