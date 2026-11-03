#!/usr/bin/env bash
# Scaffold a new RL agent package from src/ai_ml_research/rl/template.
# Usage: scaffold.sh <snake_case_name> [CamelCasePrefix] [paper-template]
#   paper-template: folder under template/paper to use (default: arxiv-style)
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "usage: $0 <snake_case_name> [CamelCasePrefix] [paper-template]" >&2
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

PAPER_TEMPLATE="arxiv-style"
if [ $# -eq 3 ]; then
    PAPER_TEMPLATE="$3"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/src/ai_ml_research/rl/template"
TARGET_DIR="$REPO_ROOT/src/ai_ml_research/rl/$NAME"

if [ ! -d "$TEMPLATE_DIR/code" ] || [ ! -d "$TEMPLATE_DIR/paper" ]; then
    echo "error: template (code/ and paper/) not found at $TEMPLATE_DIR" >&2
    exit 1
fi

PAPER_SRC="$TEMPLATE_DIR/paper/$PAPER_TEMPLATE"
if [ ! -d "$PAPER_SRC" ]; then
    echo "error: paper template not found: $PAPER_SRC" >&2
    echo "available paper templates:" >&2
    for d in "$TEMPLATE_DIR/paper"/*/; do
        [ -d "$d" ] && echo "  $(basename "$d")" >&2
    done
    exit 2
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

cp -R "$TEMPLATE_DIR/code" "$TARGET_DIR/code"
cp -R "$PAPER_SRC" "$TARGET_DIR/paper"
cp "$TEMPLATE_DIR/citation.bib" "$TARGET_DIR/citation.bib"
# license notices, previews and readmes stay in template/paper; new papers
# credit upstream via the main.tex header comment, footnote/acknowledgments
rm -f "$TARGET_DIR/paper/License.txt" "$TARGET_DIR/paper/LICENSE"
rm -f "$TARGET_DIR/paper/README.md" "$TARGET_DIR/paper/preview.png"
find "$TARGET_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +

SCAFFOLD_NAME="$NAME" SCAFFOLD_CLASS="$CLASS" find "$TARGET_DIR" \
    -type f \( -name '*.py' -o -name '*.tex' -o -name '*.bib' \) -exec \
    perl -pi -e 'BEGIN { ($n, $c) = (quotemeta $ENV{SCAFFOLD_NAME}, quotemeta $ENV{SCAFFOLD_CLASS}) }
                 s/rl\.template(?:\.code)?/rl.$n.code/g;
                 s/Template/$c/g;
                 s/<AlgorithmName>/$c/g; s/<name>/$n/g' {} \;

if grep -rn 'Template\|rl\.template' "$TARGET_DIR"; then
    echo "error: stale template references remain in $TARGET_DIR" >&2
    exit 1
fi

trap - EXIT

echo "created $TARGET_DIR:"
find "$TARGET_DIR" -type f | sort | while read -r f; do
    echo "  ${f#"$REPO_ROOT"/}"
done
echo
echo "classes: ${CLASS}Agent, ${CLASS}Config, ${CLASS}OptimisationMetrics"
echo "paper template: $PAPER_TEMPLATE"
echo "entry point: python -m ai_ml_research.rl.$NAME.code.experiment"
