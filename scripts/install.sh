#!/usr/bin/env bash
set -euo pipefail

# Cross-platform install script for fiber-tracer (macOS/Linux).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD=""

for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        version="$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        major="$(echo "$version" | cut -d. -f1)"
        minor="$(echo "$version" | cut -d. -f2)"
        if [ "$major" -ge 3 ] && { [ "$major" -gt 3 ] || [ "$minor" -ge 10 ]; }; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python >=3.10 is required but was not found." >&2
    echo "Please install Python 3.10, 3.11, or 3.12 and try again." >&2
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

if ! command -v bun >/dev/null 2>&1; then
    echo "Bun not found. Installing Bun..."
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
    export PATH="$BUN_INSTALL/bin:$PATH"
else
    echo "Bun is already installed: $(command -v bun)"
fi

cd "$REPO_ROOT"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    "$PYTHON_CMD" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev,ml]"

cd tui
bun install

echo ""
echo "Installation complete."
echo "Activate the virtual environment with: source .venv/bin/activate"
