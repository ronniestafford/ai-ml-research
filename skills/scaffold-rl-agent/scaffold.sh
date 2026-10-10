#!/usr/bin/env bash
# Scaffold a new RL agent package from src/ai_ml_research/rl/template.
# Usage: scaffold.sh <snake_case_name> [CamelCasePrefix]
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <snake_case_name> [CamelCasePrefix]" >&2
    exit 2
fi

NAME="$1"

if [[ ! "$NAME" =~ ^[a-z][a-z0-9_]*$ ]]; then
    echo "error: name must be snake_case ([a-z][a-z0-9_]*), got: $NAME" >&2
    exit 2
fi

if [ $# -eq 2 ]; then
    CLASS="$2"
else
    CLASS="$(printf '%s' "$NAME" | awk -F_ '{ for (i = 1; i <= NF; i++) \
        printf "%s%s", toupper(substr($i, 1, 1)), substr($i, 2) }')"
fi

if [[ ! "$CLASS" =~ ^[A-Z][A-Za-z0-9]*$ ]]; then
    echo "error: class prefix must be CamelCase, got: $CLASS" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/src/ai_ml_research/rl/template"
TARGET_DIR="$REPO_ROOT/src/ai_ml_research/rl/$NAME"

if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "error: template not found at $TEMPLATE_DIR" >&2
    exit 1
fi

if [ -e "$TARGET_DIR" ]; then
    echo "error: target already exists: $TARGET_DIR" >&2
    exit 1
fi

if ! mkdir "$TARGET_DIR" 2>/dev/null; then
    echo "error: cannot create $TARGET_DIR" >&2
    exit 1
fi
trap 'rm -rf "$TARGET_DIR"' EXIT

find "$TEMPLATE_DIR" -maxdepth 1 -type f -name '*.py' -exec cp {} "$TARGET_DIR/" \;

SCAFFOLD_NAME="$NAME" SCAFFOLD_CLASS="$CLASS" find "$TARGET_DIR" \
    -maxdepth 1 -type f -name '*.py' -exec \
    perl -pi -e 'BEGIN { ($n, $c) = (quotemeta $ENV{SCAFFOLD_NAME}, quotemeta $ENV{SCAFFOLD_CLASS}) }
                 s/rl\.template/rl.$n/g; s/Template/$c/g' {} \;

if grep -rn 'Template\|rl\.template' "$TARGET_DIR"; then
    echo "error: stale template references remain in $TARGET_DIR" >&2
    exit 1
fi

trap - EXIT

echo "created $TARGET_DIR:"
for f in "$TARGET_DIR"/*.py; do
    echo "  ${f#"$REPO_ROOT"/}"
done
echo
echo "classes: ${CLASS}Agent, ${CLASS}Config, ${CLASS}OptimisationMetrics"
echo "entry point: python -m ai_ml_research.rl.$NAME.experiment"
