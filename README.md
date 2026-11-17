# arm-policy

Viam module for safely executing contract-validated arm policies. It ships two resources: `viam-labs:arm-policy:cosmos`, an `rdk:service:mlmodel` that owns a Cosmos/OpenPI export deployed as a Viam `ml_model` package and serves it through the standard MLModel `Infer`/`Metadata` API; and `viam-labs:arm-policy:policy`, a generic controller component that gathers observations from typed arm, gripper, and camera dependencies, calls the MLModel, validates and conditions returned trajectories, and dispatches only a bounded safe prefix.

The MLModel service never downloads models itself; viam-server deploys the package referenced by `model_path`. The service provisions a pinned Cosmos checkout with its upstream CUDA 13 policy-server dependency groups on first start and supervises the resulting server. The controller starts in `wait_contract` until the MLModel publishes a valid contract and stays fail-closed until then; it uses neither Cosmos nor Torch/CUDA, so it can run on the robot while inference runs on a GPU part.

## Models

- [`viam-labs:arm-policy:cosmos`](viam-labs_arm-policy_cosmos.md) — MLModel service configuration, tensor contract, and managed-server lifecycle.
- [`viam-labs:arm-policy:policy`](viam-labs_arm-policy_policy.md) — controller configuration, execution commands, safety behavior, and workflow client.

## Migrating from the single-resource configuration

- Add a `viam-labs:arm-policy:cosmos` MLModel service. Move `model_path`, `model_data_root`, `server_uri`, and the `policy_server_*` host/port/readiness attributes from the controller to that service.
- Set the controller's `policy_model` to the MLModel service name. Keep arm, gripper, camera, prompt, and safety/execution settings on the controller.
- Remove `policy_server_autoclose` and calls to the `policy_server` start/stop/status commands. The service starts its server automatically and always stops it on close; service configuration edits restart it. Controller edits do not restart the server.



## Development

```sh
uv sync
uv run pytest tests/
```

Build the registry archive with:

```sh
make module.tar.gz
```

## Install for the policy runner command

If you just want the policy runner command:

```sh
uv add git+https://github.com/viam-labs/arm-policy.git
uv run arm-policy-workflow --help
```
