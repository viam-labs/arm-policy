# arm-policy

Viam module for safely executing contract-validated arm policies. The module gathers observations from typed arm, gripper, and camera dependencies, calls an OpenPI-compatible WebSocket policy server, validates and conditions returned trajectories, and dispatches only a bounded safe prefix.

The inference server is optional but owned by the module: `models/download` atomically stores a configured immutable Hugging Face portable export only after its `.cosmos-policy-complete.json` publication marker validates, and `policy_server` starts its isolated, pinned-Cosmos runtime only on demand. Normal module startup and offline `models/list` use neither Cosmos nor Torch/CUDA. See the [model documentation](awinter_arm-policy_policy.md#model-and-server-management) for configuration, persistent-store layout, publication marker contract, and lifecycle commands.

## Model

- [`awinter:arm-policy:policy`](awinter_arm-policy_policy.md) — generic component implementing policy execution, status, cancellation, trajectory conditioning, and simulator or physical Lite6 dispatch.

## Development

```sh
uv sync
uv run pytest tests/
```

Build the registry archive with:

```sh
make module.tar.gz
```
