# Model `viam-labs:arm-policy:policy`

A generic component that executes portable Cosmos/OpenPI arm-policy exports through normal Viam resources. It captures camera and joint observations, validates the export contract, conditions each 32-step trajectory, and executes only a validated receding-horizon prefix. Malformed output, unsafe kinematics, timeouts, cancellation, or missed dispatch deadlines fail closed and stop the arm and gripper.

The inference server is an optional module-owned subprocess. On its first `policy_server start`, the module checks out the pinned Cosmos revision into its cache and uses that checkout's own `uv sync --all-extras --group=cu130-torch213-train --group=policy-server` dependency definitions. Normal robot-control startup never imports or installs Cosmos, Torch, or CUDA dependencies.

## Configuration

```json
{
  "world": "sim-world",
  "arm": "pick-arm",
  "gripper": "pick-gripper",
  "fixed_camera": "scene-cam",
  "fixed_camera_2": "third-cam",
  "wrist_camera": "wrist-cam",
  "prompt": "Pick up the block",
  "policy_contract_path": "/opt/models/policy_contract.json",
  "action_prefix_duration_sec": 1.0,
  "max_chunks": 1
}
```

### Resource attributes

| Name | Default | Description |
|---|---:|---|
| `world` | `sim-world` | Generic world resource used when an execute command sets `progress_prop`. Set to `null` when no world resource is available. |
| `arm` | `pick-arm` | Arm resource receiving validated targets. |
| `gripper` | `pick-gripper` | Gripper resource receiving commands selected by the policy contract's embodiment. |
| `fixed_camera` | `scene-cam` | Camera returning exactly one 1280x720 or 640x360 image. |
| `fixed_camera_2` | `third-cam` | Second fixed camera, using the same supported sizes as `fixed_camera`. Set to `""` to reuse `fixed_camera` for both exterior views. |
| `wrist_camera` | `wrist-cam` | Camera returning exactly one 640x480 image. |

`policy_contract_path` is an optional local path to a schema-version-3
`policy_contract.json`. It is required unless `model_path` is set. When omitted,
the component uses `policy_contract.json` from the deployed `model_path` package.

### Optional attributes

| Name | Default | Description |
|---|---:|---|
| `fixed_camera_width` / `fixed_camera_height` | unset | Unset accepts 1280x720 or 640x360 fixed images. Set both (`1280`/`720` or `640`/`360`) to require that size. |
| `prompt` | — | Optional non-empty default task prompt. Each `execute` must provide `prompt` when this is unset. |
| `conditioner_algorithm` | `toppra` | `quintic` or `toppra`. |
| `max_chunks` | `1` | Receding-horizon chunks per execution, from 1 through 100. |
| `action_prefix_steps` | `8` | Steps executed from each conditioned chunk, from 1 through 32. Mutually exclusive with duration. |
| `action_prefix_duration_sec` | — | Prefix duration from 0.05 through 1.6 seconds in 50 ms increments. |
| `connect_timeout_sec` | `10` | Policy-server connection timeout. |
| `inference_timeout_sec` | `120` | Observation/inference timeout. |
| `deadline_tolerance_sec` | `0.5` | Maximum local target-dispatch lateness. |
| `workspace_task_min_m` / `workspace_task_max_m` | — | Optional three-value task-space bounds; set together. |
| `binary_gripper_open_threshold` | `0.4` | Binary-open hysteresis threshold. |
| `binary_gripper_close_threshold` | `0.6` | Binary-close hysteresis threshold. |
| `binary_gripper_initial_state` | `0` | Initial commanded binary state: `0` open or `1` closed. Setup should establish this state physically. |
| `binary_gripper_settle_sec` | `3` | Cancellable wait after each nonblocking `Open` or `Grab` command. |
| `server_uri` | derived | OpenPI policy-server `ws://` or `wss://` endpoint; when omitted, uses `ws://{policy_server_host}:{policy_server_port}`. |
| `policy_server_host` | `127.0.0.1` | Bind host for the module-owned inference process. |
| `policy_server_port` | `8000` | Bind port for the module-owned inference process. |
| `policy_server_readiness_timeout_sec` | `900` | Startup readiness timeout, including first-run environment provisioning. |

### Viam-deployed model

Upload the portable export (the directory containing `policy_contract.json`,
`runtime_manifest.json`, `model.safetensors.index.json`, and its shards) to the
Viam registry as an `ml_model` package, add it to the machine's `packages`, and
point `model_path` at it. viam-server downloads, versions, and syncs the
package.

```json
{
  "packages": [
    {
      "name": "cosmos-xarm-pick",
      "package": "<org-id>/cosmos-xarm-pick",
      "type": "ml_model",
      "version": "latest"
    }
  ],
  "components": [
    {
      "name": "policy",
      "model": "viam-labs:arm-policy:policy",
      "attributes": {
        "model_path": "${packages.ml_model.cosmos-xarm-pick}",
        "...": "..."
      }
    }
  ]
}
```

`model_path` must resolve to an absolute directory. The module never
downloads model weights itself.

On startup and every reconfiguration the component reads only
`<model_path>/policy_contract.json` and applies contract and safety-profile
validation; it does not traverse the package or read weight shards. It reports
`wait_contract` until viam-server has finished deploying the package and the
contract is valid. `policy_server` lifecycle commands remain available while
waiting. `execute` (including observation, inference, safety validation, and
motion) fails closed with the missing or invalid contract reason. An explicit
`policy_contract_path` takes precedence and is parsed and safety-validated
directly.

### Data root and server attributes

`model_data_root` defaults to `/opt/viam/arm-policy` and holds the managed policy
server's log. Set it to another absolute persistent directory when the
deployment uses a different writable data location.

`policy_server_host` and `policy_server_port` configure the module-owned inference
process and provide the default `server_uri`. The 900-second
`policy_server_readiness_timeout_sec` includes first-run dependency installation.

The policy server runs with Cosmos guardrails disabled. They are prompt-safety
models fetched from a gated Hugging Face repository; arm safety is enforced by
the module's contract, safety profile, and trajectory validation.

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

## Commands

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

### Model and server management

Start the isolated pinned-Cosmos server for the deployed `model_path` package. The first start requires network access and substantial disk space while the pinned upstream CUDA 13 policy-server environment is provisioned; later starts reuse it. Start is idempotent and waits for HTTP `GET /healthz` on the configured host and port to return status 200 with body `OK\n`, so its readiness timeout must cover first-time provisioning. The bounded health request checks server responsiveness without opening a WebSocket or running inference. Readiness failures include the log path and log tail. `status` reports `absent`, `starting`, `healthy`, `unhealthy`, or `exited`; a running server also reports `model`, the directory it was launched with. That can differ from the `model_path` shown in the app while `viam-server` is still downloading a package for a pending configuration, so compare the two before trusting a `healthy` server after a model change. `stop` sends a bounded graceful process-group termination followed by forced cleanup when needed.

Server startup checks that the package contains the required export files and the VAE, and that the training config and policy contract match their `runtime_manifest.json` SHA-256 digests. It does not traverse the package or hash the VAE or weight shards; payload loading errors remain the policy server's responsibility. A missing file or digest mismatch prevents startup.

```json
{"command": "policy_server", "action": "start"}
{"command": "policy_server", "action": "status"}
{"command": "policy_server", "action": "stop"}
```

The supervisor is intentionally a module-owned subprocess rather than s6: it needs no host package and keeps no on-disk state. The server lives exactly as long as the component instance: closing the component stops it, and an explicit `policy_server stop` stops it earlier. Because the Python module SDK rebuilds a resource on every configuration edit (close the old instance, construct the new one), each config edit or `viam-server` restart stops the server and requires a new `policy_server start`, which pays the full startup cost again (typically 60–90 s once the environment is provisioned).

Each start request emits a Viam module info log with its host and port. A successful readiness check emits a second info log with the PID, host, port, child-log path, and whether the start was idempotent. Startup failures also emit an error in the Viam module log before the start command fails. After startup, a background watcher checks for unexpected process exits and logs each failed process once, with its PID, exit status, child-log path, and a bounded, credential-redacted log tail. Deliberate stops do not emit failure notifications. The child log remains the full diagnostic record; these notifications do not replace it or constitute an inference health check.

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

