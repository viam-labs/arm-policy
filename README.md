# arm-policy

Viam module for safely executing contract-validated arm policies. It ships two resources: `viam-labs:arm-policy:cosmos`, an `rdk:service:mlmodel` that owns a Cosmos/OpenPI export deployed as a Viam `ml_model` package and serves it through the standard MLModel `Infer`/`Metadata` API; and `viam-labs:arm-policy:policy`, a generic controller component that gathers observations from typed arm, gripper, and camera dependencies, calls the MLModel, validates and conditions returned trajectories, and dispatches only a bounded safe prefix.

The MLModel service never downloads models itself; viam-server deploys the package referenced by `model_path`. The service provisions a pinned Cosmos checkout with its upstream CUDA 13 policy-server dependency groups on first start and supervises the resulting server. The controller starts in `wait_contract` until the MLModel publishes a valid contract and stays fail-closed until then; it uses neither Cosmos nor Torch/CUDA, so it can run on the robot while inference runs on a GPU part. See the [model documentation](viam-labs_arm-policy_policy.md) for configuration, the tensor contract, and lifecycle details.

## Models

- [`viam-labs:arm-policy:cosmos`](viam-labs_arm-policy_policy.md#viam-labsarm-policycosmos-mlmodel-service) — MLModel service that supervises the pinned Cosmos policy server for a deployed export and serves `Infer`/`Metadata`.
- [`viam-labs:arm-policy:policy`](viam-labs_arm-policy_policy.md#viam-labsarm-policypolicy-controller-component) — generic controller component implementing policy execution, status, cancellation, trajectory conditioning, and simulator or physical Lite6 dispatch.

## Development

```sh
uv sync
uv run pytest tests/
```

Build the registry archive with:

```sh
make module.tar.gz
```
