# Model `awinter:arm-policy:policy`

A generic component that executes portable Cosmos/OpenPI arm-policy exports through normal Viam resources. It captures camera and joint observations, validates the export contract, conditions each 32-step trajectory, and executes only a validated receding-horizon prefix. Malformed output, unsafe kinematics, timeouts, cancellation, or missed dispatch deadlines fail closed and stop the arm and gripper.

The model does not host or download the inference server. Start an OpenPI-compatible WebSocket server separately and configure its URI.

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
| `gripper` | string | Gripper resource receiving normalized commands. |
| `fixed_camera` | string | Camera returning exactly one 1280x720 image. |
| `wrist_camera` | string | Camera returning exactly one 640x480 image. |
| `server_uri` | string | `ws://` or `wss://` OpenPI policy-server endpoint. |
| `prompt` | string | Non-empty default task prompt. |
| `policy_contract_path` | string | Local path to a schema-version-2 `policy_contract.json`. |

### Optional attributes

| Name | Default | Description |
|---|---:|---|
| `world` | — | Optional Generic world resource. Required only when an execute command sets `progress_prop`. |
| `execution_backend` | `simulation` | Selects gripper handling only: normalized simulator targets or binary physical Lite6 thresholds. Arm observation and execution always use the standard Viam arm APIs. |
| `conditioner_algorithm` | `quintic` | `quintic` or `toppra`. |
| `max_chunks` | `1` | Receding-horizon chunks per execution, from 1 through 100. |
| `action_prefix_steps` | `8` | Steps executed from each conditioned chunk, from 1 through 32. Mutually exclusive with duration. |
| `action_prefix_duration_sec` | — | Prefix duration from 0.05 through 1.6 seconds in 50 ms increments. |
| `connect_timeout_sec` | `10` | Policy-server connection timeout. |
| `inference_timeout_sec` | `120` | Observation/inference timeout. |
| `deadline_tolerance_sec` | `0.025` | Maximum local target-dispatch lateness. |
| `workspace_task_min_m` / `workspace_task_max_m` | — | Optional three-value task-space bounds; set together. |
| `physical_gripper_open_threshold` | `0.4` | Physical Lite6 binary-open threshold. |
| `physical_gripper_close_threshold` | `0.6` | Physical Lite6 binary-close threshold. |
| `physical_gripper_initial_state` | `0` | Initial physical gripper state: `0` open or `1` closed. |

For every backend, arm joint observations come from `GetJointPositions` and each
validated prefix is sent through `MoveThroughJointPositionsStreamed`. The backend
adapter remains only because the simulated gripper accepts normalized inputs while
the physical Lite6 gripper uses binary `Open`/`Grab` commands without normalized
position feedback.

Finite policy gripper outputs are saturated to `[0, 1]` before validation and
dispatch, with the maximum projection recorded in conditioning diagnostics.
Trajectory conditioning interpolates the gripper command over retimed samples;
TOPPRA constraints apply only to the six arm joints. The simulation backend uses
continuous normalized targets, while `physical_lite6` converts them to binary
open/close commands with hysteresis thresholds.

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
