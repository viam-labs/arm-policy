# Models `viam-labs:arm-policy:cosmos` and `viam-labs:arm-policy:policy`

The module ships two resources that are configured together:

- `viam-labs:arm-policy:cosmos` (`rdk:service:mlmodel`) owns a portable Cosmos/OpenPI export deployed as a Viam `ml_model` package: it provisions and supervises the pinned Cosmos policy server and exposes inference through the standard MLModel `Infer`/`Metadata` API.
- `viam-labs:arm-policy:policy` (`rdk:component:generic`) is the controller. It captures camera and joint observations, validates the export contract published by the MLModel service, conditions each 32-step trajectory, and executes only a validated receding-horizon prefix. Malformed output, unsafe kinematics, timeouts, cancellation, or missed dispatch deadlines fail closed and stop the arm and gripper.

The policy server is a subprocess owned by the MLModel service. On its first start, the service checks out the pinned Cosmos revision into its cache and uses that checkout's own `uv sync --all-extras --group=cu130-torch213-train --group=policy-server` dependency definitions. The controller never imports or installs Cosmos, Torch, or CUDA dependencies, so it can run on the robot while the MLModel service runs on a GPU part reached through a remote.

## Configuration

```json
{
  "services": [
    {
      "name": "cosmos",
      "model": "viam-labs:arm-policy:cosmos",
      "api": "rdk:service:mlmodel",
      "attributes": {
        "model_path": "${packages.ml_model.cosmos-xarm-pick}"
      }
    }
  ],
  "components": [
    {
      "name": "policy",
      "model": "viam-labs:arm-policy:policy",
      "api": "rdk:component:generic",
      "attributes": {
        "world": "sim-world",
        "arm": "pick-arm",
        "gripper": "pick-gripper",
        "fixed_camera": "scene-cam",
        "fixed_camera_2": "third-cam",
        "wrist_camera": "wrist-cam",
        "policy_model": "cosmos",
        "prompt": "Pick up the block",
        "action_prefix_duration_sec": 1.0,
        "max_chunks": 1
      }
    }
  ]
}
```

## `viam-labs:arm-policy:cosmos` (MLModel service)

| Name | Default | Description |
|---|---:|---|
| `model_path` | required | Absolute directory of the deployed portable export, normally `"${packages.ml_model.<name>}"`. A legacy `/runtime/vae/Wan2.2_VAE.pth` suffix is stripped. |
| `model_data_root` | `/opt/viam/arm-policy` | Holds the managed policy server's state file and log. Set it to another absolute persistent directory when the deployment uses a different writable data location; `null` falls back to `$VIAM_MODULE_DATA/arm-policy`, then `$XDG_DATA_HOME/arm-policy`, then `~/.local/share/viam/arm-policy`. |
| `policy_server_host` / `policy_server_port` | `127.0.0.1` / `8000` | Bind address of the managed server. |
| `policy_server_readiness_timeout_sec` | `900` | How long a start waits for `GET /healthz`; includes first-run dependency installation. |
| `policy_server_autoclose` | `false` | Stop the managed server on every reconfigure and on close. Leave disabled to keep it alive across module and `viam-server` restarts. |
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

### Server lifecycle

The service starts the managed server from `reconfigure` in the background once `model_path` is configured and `server_uri` is unset. A `model_path` change (a new package version) stops the running server before restarting it so it cannot keep serving the previous weights; any other reconfigure leaves a healthy server alone (start is idempotent). `Metadata` works as soon as `<model_path>/policy_contract.json` is readable; `Infer` fails with the supervisor's state and the last start error until the server answers.

## `viam-labs:arm-policy:policy` (controller component)

### Resource attributes

| Name | Default | Description |
|---|---:|---|
| `world` | `sim-world` | Generic world resource used when an execute command sets `progress_prop`. Set to `null` when no world resource is available. |
| `arm` | `pick-arm` | Arm resource receiving validated targets. |
| `gripper` | `pick-gripper` | Gripper resource receiving commands selected by the policy contract's embodiment. |
| `fixed_camera` | `scene-cam` | Camera returning exactly one 1280x720 or 640x360 image. |
| `fixed_camera_2` | `third-cam` | Second fixed camera, using the same supported sizes as `fixed_camera`. Set to `""` to reuse `fixed_camera` for both exterior views. |
| `wrist_camera` | `wrist-cam` | Camera returning exactly one 640x480 image. |
| `policy_model` | — | Name of the `viam-labs:arm-policy:cosmos` MLModel service (or any MLModel implementing the tensor contract above). Required unless `policy_contract_path` is set and only `replay` executions are used. |

`policy_contract_path` is an optional local path to a schema-version-3
`policy_contract.json`. It is required unless `policy_model` is set. When omitted,
the contract is read from the policy model's `Metadata`.

### Optional attributes

| Name | Default | Description |
|---|---:|---|
| `fixed_camera_width` / `fixed_camera_height` | unset | Unset accepts 1280x720 or 640x360 fixed images. Set both (`1280`/`720` or `640`/`360`) to require that size. |
| `prompt` | — | Optional non-empty default task prompt. Each `execute` must provide `prompt` when this is unset. |
| `conditioner_algorithm` | `toppra` | `quintic` or `toppra`. |
| `max_chunks` | `1` | Receding-horizon chunks per execution, from 1 through 100. |
| `action_prefix_steps` | `8` | Steps executed from each conditioned chunk, from 1 through 32. Mutually exclusive with duration. |
| `action_prefix_duration_sec` | — | Prefix duration from 0.05 through 1.6 seconds in 50 ms increments. |
| `connect_timeout_sec` | `10` | Timeout for metadata fetches and arm sampling. |
| `inference_timeout_sec` | `120` | Observation/inference timeout per chunk. |
| `deadline_tolerance_sec` | `0.5` | Maximum local target-dispatch lateness. |
| `workspace_task_min_m` / `workspace_task_max_m` | — | Optional three-value task-space bounds; set together. |
| `binary_gripper_open_threshold` | `0.4` | Binary-open hysteresis threshold. |
| `binary_gripper_close_threshold` | `0.6` | Binary-close hysteresis threshold. |
| `binary_gripper_initial_state` | `0` | Initial commanded binary state: `0` open or `1` closed. Setup should establish this state physically. |
| `binary_gripper_settle_sec` | `3` | Cancellable wait after each nonblocking `Open` or `Grab` command. |


### Viam-deployed model

Upload the portable export (the directory containing `policy_contract.json`,
`runtime_manifest.json`, `model.safetensors.index.json`, and its shards) to the
Viam registry as an `ml_model` package, add it to the machine's `packages`, and
point the MLModel service's `model_path` at it. viam-server downloads, versions,
and syncs the package; the module never downloads model weights itself.

```json
{
  "packages": [
    {
      "name": "cosmos-xarm-pick",
      "package": "<org-id>/cosmos-xarm-pick",
      "type": "ml_model",
      "version": "latest"
    }
  ]
}
```

On startup and every reconfiguration the controller resets readiness and, on
the next `status` or `execute`, fetches `Metadata` from `policy_model` and
applies contract and safety-profile validation. It reports `wait_contract`
until the package is deployed and the contract is valid; `execute` (including
observation, inference, safety validation, and motion) fails closed with the
missing or invalid contract reason. An explicit `policy_contract_path` takes
precedence and is parsed and safety-validated during reconfigure.

The policy server runs with Cosmos guardrails disabled. They are prompt-safety
models fetched from a gated Hugging Face repository; arm safety is enforced by
the controller's contract, safety profile, and trajectory validation.

Arm joint observations always come from `GetJointPositions`, and validated arm
prefixes always use `MoveThroughJointPositionsStreamed`. The command path
selects binary or continuous gripper behavior exclusively from
`embodiment.gripper_mode` in the policy contract, not from the execution
environment. `position` grippers receive normalized targets at the contract's
`action_schema.fps` cadence.

`binary` grippers expose last-commanded `0`/`1` state. The runner applies
hysteresis to policy output and truncates the current arm prefix at the first
open/close transition. After reaching that target it issues `Open` or `Grab`,
waits `binary_gripper_settle_sec`, discards the unexecuted suffix, and captures a
fresh observation before the next inference chunk.

Finite policy gripper outputs are saturated to `[0, 1]` before validation and
dispatch, with the maximum projection recorded in conditioning diagnostics.
Trajectory conditioning interpolates the gripper signal over retimed samples;
TOPPRA constraints apply only to the six arm joints.
Measured joint-history velocity uses monotonic timestamps from the two completed
`GetJointPositions` calls; camera capture latency is not treated as a fixed-rate
joint displacement.


Current contracts require a 32x7 action chunk. `action_schema.fps` must equal
`conditioning_fps`; the runner derives trajectory conditioning, safety
derivatives, prefix durations, dispatch deadlines, and policy-server
conditioning from that rate. Supported embodiments are Lite6 with
`lite6_gripper` and UR20 with `robotiq_2f_85`. Safety profiles pin joint order,
units, limits, FK, reach, floor, and TCP offset; configuration cannot weaken
them.

## Commands (controller component)

Execute one or more policy chunks:

```json
{
  "command": "execute",
  "prompt": "optional override",
  "max_chunks": 1,
  "action_prefix_duration_sec": 1.0,
  "deadline_tolerance_sec": 0.15,
  "progress_prop": "pick_block"
}
```
`prompt` overrides the component default. It is required when the component has
no configured default prompt.

`progress_prop` is optional. When set, the policy queries the configured `world`
for that prop's pose before and after each executed prefix. Omit both fields when
ground-truth world poses are unavailable, including normal physical execution.

Stop inference/execution and hold both actuators:

```json
{"command": "stop"}
```

Read active and previous-run diagnostics:

```json
{"command": "status"}
```

Only one execution owns the robot. Status and results include raw and conditioned chunks, conditioning metrics, validation scope, timing, safety profile, progress records, and rejection reasons.

Command payloads are validated as a Pydantic discriminated union keyed by
`command`. Unknown commands, actions, fields, and invalid field types are
rejected before dispatch.

### Managed policy server (MLModel service)

The MLModel service starts the isolated pinned-Cosmos server for its `model_path` package from `reconfigure`; there are no lifecycle DoCommands because the MLModel API has none. The first start requires network access and substantial disk space while the pinned upstream CUDA 13 policy-server environment is provisioned; later starts reuse it. Start is idempotent and waits for HTTP `GET /healthz` on the configured host and port to return status 200 with body `OK\n`, so `policy_server_readiness_timeout_sec` must cover first-time provisioning. Readiness failures are logged with the log path and a credential-redacted log tail, and are reported by the next `Infer` call together with the supervisor state (`absent`, `starting`, `healthy`, `unhealthy`, or `exited`).

Server startup checks that the package contains the required export files and the VAE, and that the training config and policy contract match their `runtime_manifest.json` SHA-256 digests. It does not traverse the package or hash the VAE or weight shards; payload loading errors remain the policy server's responsibility. A missing file or digest mismatch prevents startup.

The supervisor is intentionally a module-owned subprocess rather than s6: it needs no host package, records Linux start ticks to avoid PID-reuse kills, and recovers the persistent process state when the service restarts with the same name and data root. By default the server survives reconfigures, module restarts, and `viam-server` restarts; only a `model_path` change, `policy_server_autoclose`, or removing the service stops it.

Each start emits a Viam module info log with its host and port. A successful readiness check emits a second info log with the PID, host, port, child-log path, and whether the start was idempotent. After startup, a background watcher checks for unexpected process exits and logs each failed process once, with its PID, exit status when available, child-log path, and a bounded, credential-redacted log tail. Recovered processes are monitored after service reconfiguration or restart; their exit status may be unavailable. Deliberate stops do not emit failure notifications.

At startup the runtime quarantines corrupt Hugging Face Hub cache metadata (tree JSON with an asterisk in `xet_hash`) left by upstream Cosmos downloads, logging both paths in the policy-server log.

## Workflow client

With the policy server and machine running:

```sh
uv run examples/policy_workflow.py <machine-address> \
  --policy policy --prompt "Pick up the block" --max-chunks 3 \
  --conditioner-algorithm toppra --action-prefix-duration-sec 1.0 \
  --deadline-tolerance-sec 0.15
```

The client reads SDK credentials from `VIAM_API_KEY` and `VIAM_API_KEY_ID` when
both are set. Otherwise, it uses the machine part credentials in
`--viam-config` (default `/etc/viam.json`) when that file exists; if it does not,
the client retains its insecure local-connection behavior.

The workflow homes the Lite6 to `[0, 0, 90, 0, 90, 0]` degrees and opens the
gripper unless `--no-setup` is set. Pass `--world sim-world` to additionally
sample a deterministic block pose from `--seed` and apply it with the world's
`set_prop_pose` command; omit it for physical execution or an already-prepared
scene. `--progress-prop` independently enables world-backed prop pose diagnostics
and therefore requires the policy component to configure a world.

Console output is a bounded human-readable summary of setup, effective control
parameters, terminal outcome, aggregate timing/execution/safety counts, final
TCP and gripper state, and the output path. `--output` retains the complete JSON
evidence, including raw and conditioned trajectories and per-chunk diagnostics.
Remote errors returned before the first inference response are printed after the
summary, and the client exits normally without a traceback.

