#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

export PATH=$PATH:$HOME/.local/bin

exec uv run --frozen --project "$ROOT" "$ROOT/src/main.py" "$@"
