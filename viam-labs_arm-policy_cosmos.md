# Model `viam-labs:arm-policy:cosmos`

An `rdk:service:mlmodel` that owns a portable Cosmos/OpenPI export deployed as a Viam `ml_model` package. It provisions and supervises the pinned Cosmos policy server and exposes inference through `Infer` and `Metadata`. The [policy controller](viam-labs_arm-policy_policy.md) captures observations, validates and conditions trajectories, and dispatches motion; this service never commands actuators.

## Configuration

```json
{
  "name": "cosmos",
  "model": "viam-labs:arm-policy:cosmos",
  "api": "rdk:service:mlmodel",
  "attributes": {
    "model_path": "${packages.ml_model.cosmos-xarm-pick}"
  }
}
```

Add the portable export to the machine's `packages` as an `ml_model` package named `cosmos-xarm-pick`. The package must contain `policy_contract.json`, `runtime_manifest.json`, the training configuration, weight index and shards, and the VAE. viam-server downloads and versions the package; the service never downloads model weights itself.

The managed server requires an NVIDIA GPU and the pinned upstream CUDA 13 environment. On first start it checks out the pinned Cosmos revision and runs that checkout's `uv sync --all-extras --group=cu130-torch213-train --group=policy-server`. Configure `server_uri` to use an externally managed OpenPI server instead; `model_path` must still point to the matching export so `Metadata` can publish its contract. For inference on another machine part, expose this service through a Viam remote and use its remote-qualified name as the controller's `policy_model`.

## Attributes

| Name | Default | Description |
|---|---:|---|
| `model_path` | required | Absolute directory of the deployed portable export, normally `"${packages.ml_model.<name>}"`. A legacy `/runtime/vae/Wan2.2_VAE.pth` suffix is stripped. |
| `model_data_root` | `/opt/viam/arm-policy` | Holds the managed policy server's log. Set it to another absolute persistent directory when the deployment uses a different writable data location; `null` falls back to `$VIAM_MODULE_DATA/arm-policy`, then `$XDG_DATA_HOME/arm-policy`, then `~/.local/share/viam/arm-policy`. |
| `policy_server_host` / `policy_server_port` | `127.0.0.1` / `8000` | Bind address of the managed server. |
| `policy_server_readiness_timeout_sec` | `900` | How long a start waits for `GET /healthz`; includes first-run dependency installation. |
| `package_reference` | — | Written by the Viam app when a model is deployed to the service; informational only. |
| `server_uri` | — | `ws://` or `wss://` endpoint of a policy server this service does not manage. When set, no subprocess is started. |
| `connect_timeout_sec` | `10` | WebSocket connect timeout. |
| `inference_timeout_sec` | `120` | Per-request WebSocket timeout. |

### Tensor contract

`Metadata` describes the inputs and outputs; `Infer` takes exactly these tensors:

| Direction | Name | dtype | shape |
|---|---|---|---|
| in | `observation/wrist_image_left` | uint8 | `[H, W, 3]` RGB |
| in | `observation/exterior_image_1_left` | uint8 | `[H, W, 3]` RGB |
| in | `observation/exterior_image_2_left` | uint8 | `[H, W, 3]` RGB |
| in | `observation/joint_position` | float32 | `[6]` radians |
| in | `observation/gripper_position` | float32 | `[1]` in `[0, 1]`, 0=open |
| in | `prompt` | uint8 | `[N]` UTF-8 bytes of the task prompt |
| out | `action` | float32 | `[32, 7]` absolute joint targets (rad) plus gripper |
| out | `server_timing/<stage>` | float32 | `[1]` per stage the server reports |

The prompt travels as bytes because tensors cannot carry strings and viam-server does not forward `Infer`'s `extra`. The verbatim `policy_contract.json` is published in the `extra` of the `action` output `TensorInfo` under `policy_contract_json`; the controller reads it from there because the MLModel API has no `DoCommand`.

Gripper values on the MLModel API follow the contract: `0=open`, `1=closed`. The service compensates for the pinned Cosmos server's DROID gripper inversion on both input and output; controllers and direct MLModel clients must not apply another inversion.

### Server lifecycle

The service starts the managed server in the background from `reconfigure` whenever `server_uri` is unset; there is no command to start or stop it. The server lives exactly as long as the service instance: closing the service (module or `viam-server` restart, resource removal) stops it. Because the Python module SDK rebuilds a resource on every configuration edit (close the old instance, construct the new one), every edit to the service's attributes, including a `model_path` change, stops the server and starts a fresh one, which pays the full startup cost again (typically 60–90 s once the environment is provisioned). `Metadata` works as soon as `<model_path>/policy_contract.json` is readable; `Infer` fails with the supervisor's state and the last start error until the server answers.

Closing also cancels active inference, closes its WebSocket, and rejects queued or later calls. Supervisor shutdown prevents a delayed startup worker from launching a child after the service closes. An external server configured with `server_uri` is never stopped by the service.

### Managed policy server (MLModel service)

The MLModel service starts the isolated pinned-Cosmos server for its `model_path` package from `reconfigure`; there are no lifecycle DoCommands because the MLModel API has none. The first start requires network access and substantial disk space while the pinned upstream CUDA 13 policy-server environment is provisioned; later starts reuse it. Start waits for HTTP `GET /healthz` on the configured host and port to return status 200 with body `OK\n`, so `policy_server_readiness_timeout_sec` must cover first-time provisioning. A port already answering `/healthz` is refused rather than adopted, because the service cannot know which weights that server loaded. Readiness failures are logged with the log path and a credential-redacted log tail, and are reported by the next `Infer` call together with the supervisor state (`absent`, `starting`, `healthy`, `unhealthy`, or `exited`).

Server startup checks that the package contains the required export files and the VAE, and that the training config and policy contract match their `runtime_manifest.json` SHA-256 digests. It does not traverse the package or hash the VAE or weight shards; payload loading errors remain the policy server's responsibility. A missing file or digest mismatch prevents startup.

The supervisor is intentionally a module-owned subprocess rather than s6: it needs no host package and keeps no on-disk state. The server is stopped whenever the service instance closes, because the MLModel API offers no command to stop an orphaned server afterwards.

Each start emits a Viam module info log with its host and port. A successful readiness check emits a second info log with the PID, host, port, child-log path, and whether the start was idempotent. After startup, a background watcher checks for unexpected process exits and logs each failed process once, with its PID, exit status, child-log path, and a bounded, credential-redacted log tail. Deliberate stops do not emit failure notifications. The child log remains the full diagnostic record; these notifications do not replace it or constitute an inference health check.

At startup the runtime quarantines corrupt Hugging Face Hub cache metadata (tree JSON with an asterisk in `xet_hash`) left by upstream Cosmos downloads, logging both paths in the policy-server log.
