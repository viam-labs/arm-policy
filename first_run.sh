#!/usr/bin/env bash
# viam-server runs this once per module version, before the entrypoint, with its
# own long timeout. Everything here is also done lazily by the module, so this
# script only moves slow work out of the module-startup and readiness budgets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# viam-server may not set HOME; the launcher caches under ~/.cache and uv installs to ~/.local.
export HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
export PATH="$PATH:$HOME/.local/bin"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# The module's own environment, so run.sh does not resolve it inside the
# module-startup timeout.
uv sync --frozen --project "$ROOT"

# The pinned Cosmos policy-server runtime (git checkout + CUDA torch, ~10 GB);
# otherwise the first `policy_server start` does this under its readiness timeout.
# exec so viam-server's kill reaches the sync.
exec uv run --frozen --project "$ROOT" python "$ROOT/policy_server_runtime/launcher.py" --provision-only
