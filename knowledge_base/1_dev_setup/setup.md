# Dev Environment Setup

## Project Structure
Experiments and algorithms are contained in the [src](/src/) folder.

## Native (non DevContainer) Development
To exploit Apple Metal GPU acceleration, develop on the host. Requires installation [uv](https://docs.astral.sh/uv/getting-started/installation/) directly on your dev machine to manage python and project dependencies,
```bash
$ brew install uv
$ uv sync
```

## DevContainer Development
[Devcontainers](/.devcontainer) are available for both [python](/.devcontainer/ai-ml-research-python/) and [LaTeX](/.devcontainer/ai-ml-research-python/). The LaTeX container ships with the full LaTeX-Workshop (large container) to compile and render [theory_snippets](/knowledge_base/theory_snippets.tex).

## Isolated Coding Agents
Even if you develop on the host, you can still run a coding agent a [Docker sandbox](/.docker-sbxs/sbx-agents.md).