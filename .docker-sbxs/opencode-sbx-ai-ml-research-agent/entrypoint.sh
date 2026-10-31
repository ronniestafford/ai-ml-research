#!/bin/bash
set -Eeuo pipefail

# Respect the sandbox runtime directory
WORKSPACE_DIR="${WORKSPACE_DIR:?WORKSPACE_DIR not set by sandbox runtime}"
cd "$WORKSPACE_DIR" || { echo "ERROR: cannot enter workspace at $WORKSPACE_DIR" >&2; exit 1; }

# Ensure configuration files exist in the bind-mounted workspace
if [ ! -f .python-version ]; then
    echo "Error: .python-version file not found in $WORKSPACE_DIR!" >&2
    exit 1
fi

if [ ! -f pyproject.toml ]; then
    echo "Error: pyproject.toml file not found in $WORKSPACE_DIR!" >&2
    exit 1
fi

TARGET_VERSION=$(cat .python-version)
echo "=== Target Python Version Detected: $TARGET_VERSION ==="

echo "=== 1. Installing Pinned Python Version ==="
# Installs the version specified in the host's .python-version inside the sandbox
uv python install

echo "=== 2. Syncing Dependencies Silently to Sandbox ==="
# --project . reads the host pyproject.toml
# --frozen prevents uv from modifying or creating a uv.lock file on the host mount
uv sync --project . --frozen

echo "=== 3. Verifying Isolated Environment ==="
uv run python --version

echo "=== 4. Handing over to OpenCode Agent ==="
# Inject the sandbox venv binary path directly to system PATH.
# When the agent runs 'python' or global tools, it is strictly forced 
# to use the container's environment, protecting your host.
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

exec opencode "$@"
