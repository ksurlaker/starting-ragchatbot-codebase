#!/bin/bash

# Code quality check script
# Usage: ./quality.sh [command]
# Commands:
#   check   - Run all checks without modifying files (default)
#   fix     - Auto-fix formatting and fixable lint issues
#   format  - Run black formatter only
#   lint    - Run ruff linter only

set -e

COMMAND="${1:-check}"

# Ensure dev dependencies are installed
ensure_deps() {
    if ! uv run python -c "import black" 2>/dev/null; then
        echo "Installing dev dependencies..."
        uv sync --extra dev
    fi
}

run_black_check() {
    echo "Checking code formatting with black..."
    uv run black --check --diff .
}

run_black_fix() {
    echo "Formatting code with black..."
    uv run black .
}

run_ruff_check() {
    echo "Checking code with ruff..."
    uv run ruff check .
}

run_ruff_fix() {
    echo "Fixing lint issues with ruff..."
    uv run ruff check --fix .
}

case "$COMMAND" in
    check)
        ensure_deps
        echo "Running code quality checks..."
        echo ""
        run_black_check
        echo ""
        run_ruff_check
        echo ""
        echo "All checks passed!"
        ;;
    fix)
        ensure_deps
        echo "Fixing code quality issues..."
        echo ""
        run_black_fix
        echo ""
        run_ruff_fix
        echo ""
        echo "Done!"
        ;;
    format)
        ensure_deps
        run_black_fix
        ;;
    lint)
        ensure_deps
        run_ruff_check
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: ./quality.sh [check|fix|format|lint]"
        exit 1
        ;;
esac
