# arm-policy

Viam module for safely executing contract-validated arm policies. The module gathers observations from typed arm, gripper, and camera dependencies, calls an OpenPI-compatible WebSocket policy server, validates and conditions returned trajectories, and dispatches only a bounded safe prefix.

Policy exports are deployed by viam-server as Viam registry `ml_model` packages and referenced with `model_path`; the module does not download models itself. A component can start in `wait_contract` before the package is deployed; execution stays fail-closed until its validated contract makes status `ready`. The inference server is optional but owned by the module: `policy_server` provisions a pinned Cosmos checkout with its upstream CUDA 13 policy-server dependency groups only on first start. Normal module startup uses neither Cosmos nor Torch/CUDA. See the [model documentation](viam-labs_arm-policy_policy.md#viam-deployed-model) for configuration and lifecycle commands.

## Model

- [`viam-labs:arm-policy:policy`](viam-labs_arm-policy_policy.md) — generic component implementing policy execution, status, cancellation, trajectory conditioning, and simulator or physical Lite6 dispatch.

## Development

```sh
uv sync
uv run pytest tests/
```

Build the registry archive with:

```sh
make module.tar.gz
```

The archive's entrypoint, `dist/arm-policy`, is a launcher for `run.sh` that the
build generates. It is deliberately not committed: Viam cloud build treats a
module whose entrypoint already exists in source as pre-built and skips `make`.

Hot reload the working tree onto a machine with a cloud build:

```sh
viam module reload --part-id <part-id>
```

`viam module reload-local --part-id <part-id>` builds on your workstation
instead; on macOS prefix it with `COPYFILE_DISABLE=1` to keep `._*` metadata
files out of the archive.
