# arm-policy code flow

Two resources in one module: `viam-labs:arm-policy:cosmos` (`rdk:service:mlmodel`, `src/models/cosmos_policy.py`) serves a Cosmos action policy; `viam-labs:arm-policy:policy` (`rdk:component:generic`, `src/models/policy.py`) is the controller that drives the arm and gripper from it.

```mermaid
flowchart TB
    subgraph boot["Module startup"]
        VS[viam-server] --> RUN[run.sh<br/>uv run src/main.py]
        RUN --> MAIN[main.py<br/>import models → Module.run_from_registry]
        MAIN --> CP[CosmosPolicy<br/>rdk:service:mlmodel]
        MAIN --> POL[Policy<br/>rdk:component:generic]
    end

    subgraph mlmodel["CosmosPolicy reconfigure()"]
        CP --> CCFG[CosmosConfig<br/>model_path, server_uri?, host/port, timeouts]
        CCFG --> CONTRACT[read model_path/policy_contract.json<br/>kept verbatim for metadata]
        CCFG -->|model_path changed or autoclose| STOPOLD[PolicyServerSupervisor.stop]
        CCFG -->|server_uri unset| START[background _start_policy_server<br/>validate_runtime_manifest →<br/>supervisor.start launcher.py --model]
        START --> LAUNCH[launcher.py<br/>_validate → _model_view →<br/>_provision_framework cosmos-framework@pin →<br/>_prepare_overlay]
        LAUNCH -->|os.execvpe| ADAPT[service_adapter.py<br/>ContractRobolabPolicyService<br/>OpenPI WebSocket server + /healthz]
        START -.->|poll /healthz| ADAPT
    end

    subgraph ctrl["Policy reconfigure() + readiness"]
        POL --> PCFG[Config<br/>arm, gripper, cameras, policy_model, prompt, limits]
        PCFG --> DEPS[typed deps<br/>Arm / Gripper / Camera×2-3 / MLModel / world]
        PCFG -->|policy_contract_path| LOCAL[_read_policy_contract]
        PCFG -->|policy_model| META[policy_model.metadata<br/>action TensorInfo.extra.policy_contract_json]
        LOCAL --> READY[validate_contract → SafetyProfile<br/>state = ready]
        META --> READY
    end

    subgraph exec["DoCommand execute → _execute_run loop"]
        X1[sample joints, sleep 1/fps] --> X2[_observation<br/>cameras.get_images + joints + gripper]
        X2 --> X3[_observation_tensors<br/>images uint8, joints f32, gripper f32,<br/>prompt as uint8 bytes]
        X3 -->|MLModel.infer over gRPC| INF[CosmosPolicy.infer<br/>validate tensors → msgpack → ws send/recv<br/>→ action f32 32×7 + server_timing/*]
        INF -->|ws| ADAPT
        INF --> X4[_parse_chunk + _condition_chunk<br/>quintic/toppra, FK, SafetyProfile]
        X4 --> X5[_dispatch_prefix<br/>move_through_joint_positions_streamed<br/>gripper set_input / binary barrier]
        X5 -->|chunk < max_chunks| X1
        X5 -->|done / stop / error| X6[_finish_result, _safe_stop]
    end

    READY --> X1
    DC{Policy DoCommand} -->|execute| X1
    DC -->|stop| STOP[stop event; cancels in-flight infer<br/>→ CosmosPolicy closes its socket]
    DC -->|status| META

    CLIENT[examples/policy_workflow.py] --> DC
```

## Stages

| Stage | Files | Notes |
|---|---|---|
| Startup | `run.sh`, `src/main.py`, `src/models/__init__.py` | Importing both `EasyResource` classes registers them; `Module.run_from_registry()` serves them. |
| MLModel config / lifecycle | `src/models/cosmos_policy.py`, `src/models/model_export.py`, `src/models/policy_server.py` | `reconfigure` validates `model_path`, reads the contract, and (unless `server_uri` is set) starts the managed server in the background. A `model_path` change stops the old server first. The MLModel API has no DoCommand. |
| Policy server | `policy_server_runtime/launcher.py`, `policy_server_runtime/service_adapter.py` | Launcher provisions the pinned `cosmos-framework`, builds a symlinked model view (keyed by package path + config), applies the joint-width patch, then `execvpe`s the adapter, which serves OpenPI WebSocket (msgpack) with guardrails disabled. |
| Tensor contract | `src/models/wire.py` | Tensor names, msgpack framing, `prompt` ↔ uint8 bytes, `CONTRACT_EXTRA` key for the contract in `metadata()`. |
| Controller readiness | `src/models/policy.py` (`_advance_readiness`) | Contract from `policy_contract_path` (sync, at reconfigure) or from `policy_model.metadata()` (on first `status`/`execute`). Cross-checked against config limits via `validate_contract`. |
| Execute loop | `src/models/policy.py` (`_execute_run`, `_infer`), `src/models/trajectory_conditioner.py` | Per chunk: observation → tensors → `MLModel.infer` → `{action, server_timing}` → conditioning/validation → bounded prefix to hardware. `stop` cancels the in-flight infer; the service closes its WebSocket on any torn exchange. |
| Replay | `src/models/replay_source.py` | Recorded chunks stand in for the model; needs only `policy_contract_path`. |

## DoCommand surface

- `viam-labs:arm-policy:policy`: `execute`, `stop`, `status`.
- `viam-labs:arm-policy:cosmos`: none (MLModel API: `Infer`, `Metadata`).
