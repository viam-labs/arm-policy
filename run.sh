#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

export HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
export PATH=$PATH:$HOME/.local/bin

# temp workaround to set up container paths
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

exec uv run --frozen --project "$ROOT" "$ROOT/src/main.py" "$@"
