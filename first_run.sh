#!/usr/bin/env bash
# viam-server runs this once per module version, before the entrypoint, with its
# own long timeout (1 h default). A non-zero exit aborts the machine's whole
# reconfigure, so only the module's own environment is allowed to fail here; the
# Cosmos runtime is best-effort and the first `policy_server start` installs it
# if this did not.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# viam-server may not set HOME; the launcher caches under ~/.cache and uv installs to ~/.local.
export HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
export PATH="$PATH:$HOME/.local/bin"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/0.11.8/install.sh | UV_NO_MODIFY_PATH=1 sh
fi

# The module's own environment, so run.sh does not resolve it inside the
# module-startup timeout.
uv sync --frozen --project "$ROOT"

# The pinned Cosmos policy-server runtime (git checkout + CUDA torch, ~10 GB).
uv run --frozen --project "$ROOT" python "$ROOT/policy_server_runtime/launcher.py" --provision-only \
  || echo "arm-policy: Cosmos runtime not provisioned; the first policy_server start will install it" >&2
