# arm-policy

Viam module for safely executing contract-validated arm policies. The module gathers observations from typed arm, gripper, and camera dependencies, calls an external OpenPI-compatible WebSocket policy server, validates and conditions returned trajectories, and dispatches only a bounded safe prefix.

The inference server remains a separate process. This keeps the robot-control module independent from Cosmos/PyTorch/CUDA startup and GPU failure modes.

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
