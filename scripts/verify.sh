#!/usr/bin/env bash
set -euo pipefail

# Cross-platform verification script for fiber-tracer (macOS/Linux).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Please run scripts/install.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# Ensure freshly-installed Bun is discoverable in this session.
if ! command -v bun >/dev/null 2>&1 && [ -d "$HOME/.bun/bin" ]; then
    export PATH="$HOME/.bun/bin:$PATH"
fi

echo "Running fiber-tracer --version..."
fiber-tracer --version

echo ""
echo "Running pytest..."
pytest

echo ""
echo "Running TUI typecheck and tests..."
cd tui
bun run typecheck
bun test

echo ""
echo "All verification checks passed."
