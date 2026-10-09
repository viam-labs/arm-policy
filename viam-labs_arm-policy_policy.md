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
  "wrist_camera": "wrist-cam",
  "server_uri": "ws://127.0.0.1:8000",
  "prompt": "Pick up the block",
  "policy_contract_path": "/opt/models/policy_contract.json",
  "conditioner_algorithm": "toppra",
  "action_prefix_duration_sec": 1.0,
  "max_chunks": 1
}
```

### Required attributes

| Name | Type | Description |
|---|---|---|
| `arm` | string | Arm resource receiving validated targets. |
| `gripper` | string | Gripper resource receiving commands selected by the policy contract's embodiment. |
| `fixed_camera` | string | Camera returning exactly one 1280x720 image. |
| `wrist_camera` | string | Camera returning exactly one 640x480 image. |
| `server_uri` | string | `ws://` or `wss://` OpenPI policy-server endpoint. |
| `prompt` | string | Non-empty default task prompt. |
| `policy_contract_path` | string | Optional local path to a schema-version-3 `policy_contract.json`. Required unless `model_source` is set. When omitted, the component uses `policy_contract.json` from the matching validated download. |

### Optional attributes

| Name | Default | Description |
|---|---:|---|
| `world` | — | Optional Generic world resource. Required only when an execute command sets `progress_prop`. |
| `conditioner_algorithm` | `quintic` | `quintic` or `toppra`. |
| `max_chunks` | `1` | Receding-horizon chunks per execution, from 1 through 100. |
| `action_prefix_steps` | `8` | Steps executed from each conditioned chunk, from 1 through 32. Mutually exclusive with duration. |
| `action_prefix_duration_sec` | — | Prefix duration from 0.05 through 1.6 seconds in 50 ms increments. |
| `connect_timeout_sec` | `10` | Policy-server connection timeout. |
| `inference_timeout_sec` | `120` | Observation/inference timeout. |
| `deadline_tolerance_sec` | `0.025` | Maximum local target-dispatch lateness. |
| `workspace_task_min_m` / `workspace_task_max_m` | — | Optional three-value task-space bounds; set together. |
| `binary_gripper_open_threshold` | `0.4` | Binary-open hysteresis threshold. |
| `binary_gripper_close_threshold` | `0.6` | Binary-close hysteresis threshold. |
| `binary_gripper_initial_state` | `0` | Initial commanded binary state: `0` open or `1` closed. Setup should establish this state physically. |
| `binary_gripper_settle_sec` | `3` | Cancellable wait after each nonblocking `Open` or `Grab` command. |
| `policy_server_autoclose` | `false` | Stop the managed policy-server process during component reconfiguration or close. Leave disabled to preserve the server across arm-policy and `viam-server` restarts. |

### Downloaded model and server attributes

Set `model_source` to either an HF model source (`owner/name` or `model:owner/name`) or a bucket source (`bucket:owner/name:path/to/export`). Model sources require `model_revision`, an immutable lowercase 40-character commit; bucket sources forbid `model_revision`. `model_data_root` is optional: when omitted the module uses `$VIAM_MODULE_DATA/arm-policy`, then `$XDG_DATA_HOME/arm-policy`, then `~/.local/share/viam/arm-policy`; set it to an absolute persistent directory to override that convention. `hf_token` is optional; `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` are also supported. A configured `hf_token` is exported as `HF_TOKEN` to the policy-server process for its gated upstream downloads, and is never returned in command responses. `policy_server_host`, `policy_server_port`, and `policy_server_readiness_timeout_sec` select the module-owned server endpoint.

For example, a bucket-backed component that discovers its contract from the downloaded export is configured as:

```json
{
  "model_source": "bucket:owner/policy:exports/production",
  "model_data_root": "/var/lib/viam/arm-policy"
}
```

When an explicit contract path is absent, startup and reconfiguration look up only the configured source/revision at its deterministic local store path. They validate the completion metadata's canonical source, revisions, publication state, and hash entries (including the remote publication marker for model sources), then SHA-256-check `policy_contract.json` before parsing it and applying safety-profile validation. This reads only the selected completion metadata and contract: it does not enumerate other exports or read model shards. Each reconfiguration rechecks these files; no discovery result is cached across reconfigurations.

The component reports `wait_contract` until a matching published download supplies a valid contract, or until a configured explicit contract becomes valid. `models/download`, `models/list`, and `policy_server` lifecycle commands remain available while waiting. `execute` (including observation, inference, safety validation, and motion) fails closed with the missing or invalid contract reason. A successful download automatically rechecks readiness and changes status to `ready` only after contract and safety-profile validation succeeds.

The persistent store is `<data-root>/arm-policy-models/...`. A directory is discoverable only after validation and atomic publication of its local `.complete.json`; interrupted or invalid staging directories are never listed.

Readiness verifies the selected safety contract, not the current integrity of model weights or other payload files; it trusts their validation at publication. Full offline bundle hashing remains in `models/list`, download cache/publication validation, and managed `policy_server` startup. Use `models/list` to audit local payload integrity. An explicit `policy_contract_path` still bypasses store discovery and is parsed and safety-validated directly.

Arm joint observations always come from `GetJointPositions`, and validated arm
prefixes always use `MoveThroughJointPositionsStreamed`. Gripper behavior comes
from `embodiment.gripper_mode` in the policy contract, not from the execution
environment. `position` grippers receive normalized targets at 20 Hz.

`binary` grippers expose last-commanded `0`/`1` state. The runner applies
hysteresis to policy output and truncates the current arm prefix at the first
open/close transition. After reaching that target it issues `Open` or `Grab`,
waits `binary_gripper_settle_sec`, discards the unexecuted suffix, and captures a
fresh observation before the next inference chunk.

Finite policy gripper outputs are saturated to `[0, 1]` before validation and
dispatch, with the maximum projection recorded in conditioning diagnostics.
Trajectory conditioning interpolates the gripper signal over retimed samples;
TOPPRA constraints apply only to the six arm joints.

The current contracts require a 32x7 action chunk at 20 Hz. Supported embodiments are Lite6 with `lite6_gripper` and UR20 with `robotiq_2f_85`. Safety profiles pin joint order, units, limits, FK, reach, floor, and TCP offset; configuration cannot weaken them.

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

Download the configured immutable portable export (the response is stable JSON containing repository, requested/resolved revisions, path, and validation state):

```json
{"command": "models/download"}
```

For an uncached bucket source, download first lists the exact configured bucket prefix using the same Hugging Face CLI credentials as the transfer. Access denial, a missing or empty prefix, or a connection failure is returned to the caller before staging or transferring the export. A configured `hf_token` takes precedence; otherwise the CLI retains its inherited environment and saved-login behavior. The preflight is a non-recursive listing; sync performs its own transfer enumeration. Already validated cached downloads remain offline and skip this check.

List only locally complete, hash-validated downloads. This command is offline and makes no Hugging Face request:

```json
{"command": "models/list"}
```

Start the isolated pinned-Cosmos server for a validated model. The first start requires network access and substantial disk space while the pinned upstream CUDA 13 policy-server environment is provisioned; later starts reuse it. Start is idempotent and waits for HTTP `GET /healthz` on the configured host and port to return status 200 with body `OK\n`, so its readiness timeout must cover first-time provisioning. The bounded health request checks server responsiveness without opening a WebSocket or running inference. Readiness failures include the log path and log tail. `status` reports `absent`, `starting`, `healthy`, `unhealthy`, or `exited`; `stop` sends a bounded graceful process-group termination followed by forced cleanup when needed.

Server startup resolves the configured `model_source` and `model_revision` directly, validates their completion metadata, and fully validates and hashes only that export before launching. It neither enumerates nor hashes other cached exports. Unlike contract readiness, this check includes model weights and runtime payloads; a missing or corrupt selected export prevents startup. `models/list` retains exhaustive store-wide validation.

```json
{"command": "policy_server", "action": "start"}
{"command": "policy_server", "action": "status"}
{"command": "policy_server", "action": "stop"}
```

The supervisor is intentionally a module-owned subprocess rather than s6: it needs no host package, records Linux start ticks to avoid PID-reuse kills, and recovers the persistent process state when the component restarts with the same name and data root. By default, only an explicit `policy_server stop` terminates the server. Set `policy_server_autoclose` to `true` to stop it automatically during component reconfiguration or close.

Each start request emits a Viam module info log with its host and port. A successful readiness check emits a second info log with the PID, host, port, child-log path, and whether the start was idempotent. Startup failures also emit an error in the Viam module log before the start command fails. After startup, a background watcher checks for unexpected process exits and logs each failed process once, with its PID, exit status when available, child-log path, and a bounded, credential-redacted log tail. Recovered processes are monitored after component reconfiguration or restart; their exit status may be unavailable. Deliberate stops do not emit failure notifications, and closing or reconfiguring the component detaches its old watcher without changing the `policy_server_autoclose` behavior. The child log remains the full diagnostic record; these notifications do not replace it or constitute an inference health check.

At process startup, before importing Cosmos or downloading models, the runtime audits `trees/*.json` under every standard model, dataset, and space repository in its effective Hugging Face Hub cache. If format-version-1 metadata contains an asterisk in `xet_hash`, it renames the entire JSON to a unique `.json.poisoned.<timestamp>.<pid>.<uuid>` sibling and emits a warning in the policy-server log with both paths. Original bytes are preserved for diagnosis; model weights, healthy metadata, malformed or unknown-format metadata, and unrelated JSON files are untouched. Previously quarantined files are not scanned again. Cache selection follows `HF_HUB_CACHE`, legacy `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, and the default cache location. Subsequent downloads fetch fresh tree metadata normally; the audit does not bypass authentication or introduce automatic download retries.

## Workflow client

With the policy server and machine running:

```sh
python3 examples/policy_workflow.py <machine-address> \
  --policy policy --prompt "Pick up the block" --max-chunks 3 \
  --conditioner-algorithm toppra --action-prefix-duration-sec 1.0 \
  --deadline-tolerance-sec 0.15
```

The workflow homes the Lite6 to `[0, 0, 90, 0, 90, 0]` degrees and opens the
gripper unless `--no-setup` is set. Pass `--world sim-world` to additionally
reset the simulated prop; omit it for physical execution or an already-prepared
scene. `--progress-prop` independently enables world-backed prop pose diagnostics
and therefore requires the policy component to configure a world.
