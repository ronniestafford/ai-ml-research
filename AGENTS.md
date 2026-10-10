# Agent instructions

## Python environments — never modify them

- NEVER run `uv run`, `uv sync`, `uv add`, `uv pip`, or `pip install`.
  `uv run` deletes and recreates `.venv` when its interpreter link is
  stale, destroying the user's environment.
- To check code changes, use the venv's interpreter directly and read-only:
  - `.venv/bin/python -m py_compile <file>` for syntax checks.
  - `.venv/bin/python <script>` only if running the script is required.
- If `.venv/bin/python` is broken or a package is missing, DO NOT fix it
  yourself. Report it and let the user run `uv sync`.
- Dependencies are managed by the user; only suggest changes to
  `pyproject.toml`, never apply environment changes.
