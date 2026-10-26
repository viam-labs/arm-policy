import asyncio
import json

import numpy as np
import pytest
import websockets.exceptions
from test_policy import FakeSocket, _write_contract
from viam.proto.app.robot import ServiceConfig
from viam.utils import dict_to_struct, struct_to_dict

from viam_arm_policy.cosmos_policy import CosmosConfig, CosmosPolicy
from viam_arm_policy.wire import (
    ACTION_TENSOR,
    CONTRACT_EXTRA,
    GRIPPER_TENSOR,
    IMAGE_TENSORS,
    JOINT_TENSOR,
    PROMPT_TENSOR,
    pack_message,
    prompt_from_tensor,
    prompt_to_tensor,
    unpack_message,
)


class SocketCosmos(CosmosPolicy, register=False):
    def __init__(self, socket, model_path="/tmp/policy-model"):
        super().__init__("cosmos")
        self.socket = socket
        self.connections = 0
        self._config = CosmosConfig(model_path=model_path, inference_timeout_sec=0.5)

    async def _connect(self):
        self.connections += 1
        return self.socket


def _inputs(prompt="pick the block"):
    return {
        IMAGE_TENSORS[0]: np.zeros((480, 640, 3), dtype=np.uint8),
        IMAGE_TENSORS[1]: np.zeros((720, 1280, 3), dtype=np.uint8),
        IMAGE_TENSORS[2]: np.zeros((720, 1280, 3), dtype=np.uint8),
        JOINT_TENSOR: np.zeros(6, dtype=np.float32),
        GRIPPER_TENSOR: np.zeros(1, dtype=np.float32),
        PROMPT_TENSOR: prompt_to_tensor(prompt),
    }


def _config(attrs):
    return ServiceConfig(name="cosmos", attributes=dict_to_struct(attrs))


def test_prompt_tensor_round_trips_utf8():
    assert prompt_from_tensor(prompt_to_tensor("pick the blöck")) == "pick the blöck"
    with pytest.raises(ValueError, match="uint8"):
        prompt_from_tensor(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty"):
        prompt_from_tensor(np.zeros(0, dtype=np.uint8))
    with pytest.raises(ValueError, match="UTF-8"):
        prompt_from_tensor(np.asarray([0xFF, 0xFE], dtype=np.uint8))


def test_infer_round_trips_tensors_over_one_socket():
    async def scenario():
        actions = np.full((32, 7), 0.25, dtype=np.float64)
        socket = FakeSocket(pack_message({"action": actions, "server_timing": {"infer_ms": 3.5}}))
        service = SocketCosmos(socket)

        outputs = await service.infer(_inputs())
        await service.infer(_inputs("again"))

        assert service.connections == 1
        assert outputs[ACTION_TENSOR].dtype == np.float32
        assert outputs[ACTION_TENSOR].shape == (32, 7)
        assert outputs["server_timing/infer_ms"].tolist() == pytest.approx([3.5])
        sent = unpack_message(socket.sent[0])
        assert sent["prompt"] == "pick the block"
        assert sent[JOINT_TENSOR].dtype == np.float32 and sent[JOINT_TENSOR].shape == (6,)
        assert sent[GRIPPER_TENSOR].shape == (1,)
        assert sent[IMAGE_TENSORS[0]].shape == (480, 640, 3)
        assert set(sent) == {PROMPT_TENSOR, JOINT_TENSOR, GRIPPER_TENSOR, *IMAGE_TENSORS}
        assert unpack_message(socket.sent[1])["prompt"] == "again"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda inputs: inputs.pop(PROMPT_TENSOR), "missing input tensors: prompt"),
        (lambda inputs: inputs.update({JOINT_TENSOR: np.zeros(7, dtype=np.float32)}), "six finite"),
        (lambda inputs: inputs.update({IMAGE_TENSORS[0]: np.zeros((480, 640, 3), dtype=np.float32)}), "uint8"),
        (lambda inputs: inputs.update({GRIPPER_TENSOR: np.asarray([np.nan], dtype=np.float32)}), "finite"),
    ],
)
def test_infer_rejects_malformed_inputs_before_contacting_the_server(mutate, error):
    async def scenario():
        socket = FakeSocket(pack_message({"action": np.zeros((32, 7))}))
        service = SocketCosmos(socket)
        inputs = _inputs()
        mutate(inputs)
        with pytest.raises(ValueError, match=error):
            await service.infer(inputs)
        assert socket.sent == []

    asyncio.run(scenario())


def test_infer_drops_the_socket_after_a_bad_exchange():
    async def scenario():
        socket = FakeSocket("traceback from the server")
        service = SocketCosmos(socket)
        with pytest.raises(RuntimeError, match="text error"):
            await service.infer(_inputs())
        assert service._socket is None
        assert socket.closed.is_set()

        service.socket = FakeSocket(pack_message({"action": np.zeros((31, 7))}))
        with pytest.raises(ValueError, match="shape"):
            await service.infer(_inputs())
        assert service.connections == 2

    asyncio.run(scenario())


def test_infer_reconnects_once_when_the_server_closed_the_idle_socket():
    class ClosedSocket(FakeSocket):
        async def send(self, data):
            raise websockets.exceptions.ConnectionClosedError(None, None)

    async def scenario():
        service = SocketCosmos(FakeSocket(pack_message({"action": np.zeros((32, 7))})))
        service._socket = ClosedSocket()

        outputs = await service.infer(_inputs())

        assert service.connections == 1
        assert outputs[ACTION_TENSOR].shape == (32, 7)

    asyncio.run(scenario())



def test_infer_timeout_closes_the_socket():
    async def scenario():
        socket = FakeSocket(None, block=True)
        service = SocketCosmos(socket)
        with pytest.raises(asyncio.TimeoutError):
            await service.infer(_inputs())
        assert socket.closed.is_set()
        assert service._socket is None

    asyncio.run(scenario())


def test_contract_and_metadata_wait_for_the_deployed_package(tmp_path):
    _, contract, _ = _write_contract(tmp_path)
    package = tmp_path / "package"
    service = CosmosPolicy("cosmos")
    service.reconfigure(_config({"model_path": str(package)}), {})

    async def scenario():
        with pytest.raises(RuntimeError, match="not available yet"):
            await service.metadata()

        package.mkdir()
        (package / "policy_contract.json").write_text(json.dumps(contract))
        metadata = await service.metadata()
        action_info = next(info for info in metadata.output_info if info.name == ACTION_TENSOR)
        assert json.loads(struct_to_dict(action_info.extra)[CONTRACT_EXTRA]) == contract
        assert metadata.description.endswith("ws://127.0.0.1:8000")
        inputs = {info.name: (info.data_type, list(info.shape)) for info in metadata.input_info}
        assert inputs[PROMPT_TENSOR] == ("uint8", [-1])
        assert inputs[JOINT_TENSOR] == ("float32", [6])
        assert inputs[IMAGE_TENSORS[1]] == ("uint8", [-1, -1, 3])
        outputs = {info.name: list(info.shape) for info in metadata.output_info}
        assert outputs[ACTION_TENSOR] == [32, 7]
        await service.close()

    asyncio.run(scenario())


def test_config_ignores_controller_attributes_and_derives_server_defaults():
    CosmosPolicy.validate_config(_config({"model_path": "/models/x"}))
    with pytest.raises(ValueError, match="absolute"):
        CosmosPolicy.validate_config(_config({"model_path": "relative"}))
    with pytest.raises(ValueError, match="model_path"):
        CosmosPolicy.validate_config(_config({}))
    for ignored in ("model_source", "hf_token", "prompt", "arm"):
        parsed = CosmosConfig.model_validate({"model_path": "/models/x", ignored: "x"})
        assert not hasattr(parsed, ignored)
    parsed = CosmosConfig(model_path="/models/x", server_uri="ws://gpu-box:9000")
    assert parsed.uri == "ws://gpu-box:9000"
    parsed = CosmosConfig(
        model_path="/models/x/runtime/vae/Wan2.2_VAE.pth",
        policy_server_host="::1",
        policy_server_port=9001,
    )
    assert parsed.model_path == "/models/x"
    assert parsed.uri == "ws://[::1]:9001"
    assert parsed.model_data_root == "/opt/viam/arm-policy"
    assert parsed.policy_server_readiness_timeout_sec == 900.0


@pytest.mark.parametrize("autoclose", [False, True])
def test_reconfigure_stops_server_only_with_autoclose_but_close_always_stops(tmp_path, autoclose, monkeypatch):
    async def fake_start(self):
        return {"state": "healthy"}

    monkeypatch.setattr(CosmosPolicy, "_start_policy_server", fake_start)

    class ExistingServer:
        stopped = 0

        def stop(self):
            self.stopped += 1

    async def scenario():
        existing = ExistingServer()
        service = CosmosPolicy("cosmos")
        service.reconfigure(_config({"model_path": str(tmp_path)}), {})
        service._policy_server = existing
        service.reconfigure(
            _config({"model_path": str(tmp_path), "policy_server_autoclose": autoclose}), {}
        )
        await service._server_task
        assert existing.stopped == (1 if autoclose else 0)
        service._policy_server = existing
        await service.close()
        # Removal of the service also goes through close(); a retained server would be orphaned.
        assert existing.stopped == (2 if autoclose else 1)

    asyncio.run(scenario())


def test_reconfigure_starts_server_once_and_retains_it_across_reconfigures(tmp_path, monkeypatch):
    class RecordingServer:
        def __init__(self):
            self.stops = 0

        def stop(self):
            self.stops += 1

    starts = []

    async def fake_start(self):
        starts.append(self._config.model_path)
        return {"state": "healthy"}

    monkeypatch.setattr(CosmosPolicy, "_start_policy_server", fake_start)

    async def scenario():
        service = CosmosPolicy("cosmos")
        service.reconfigure(_config({"model_path": str(tmp_path / "v1")}), {})
        await service._server_task
        assert starts == [str(tmp_path / "v1")]
        running = RecordingServer()

        # Any reconfigure, including a model_path change, leaves a running server alone
        # (start is idempotent on a healthy state file); see APP-18431.
        for attrs in (
            {"model_path": str(tmp_path / "v1"), "policy_server_port": 8001},
            {"model_path": str(tmp_path / "v2")},
        ):
            service._policy_server = running
            service.reconfigure(_config(attrs), {})
            await service._server_task
        assert running.stops == 0 and len(starts) == 3

        # External server: nothing to manage.
        service._policy_server = RecordingServer()
        service.reconfigure(_config({"model_path": str(tmp_path / "v2"), "server_uri": "ws://gpu:9000"}), {})
        await service._server_task
        assert len(starts) == 3
        await service.close()

    asyncio.run(scenario())


def test_reconfigure_waits_for_the_in_flight_start_before_stopping_or_starting(tmp_path, monkeypatch):
    """Two starts racing on one state file orphan a server; the second must queue."""
    release = asyncio.Event()
    starts = []

    async def slow_start(self):
        starts.append(self._config.model_path)
        await release.wait()
        return {"state": "healthy"}

    monkeypatch.setattr(CosmosPolicy, "_start_policy_server", slow_start)

    class RecordingServer:
        stops = 0

        def stop(self):
            self.stops += 1

    async def scenario():
        service = CosmosPolicy("cosmos")
        service.reconfigure(_config({"model_path": str(tmp_path / "v1")}), {})
        await asyncio.sleep(0)
        running = RecordingServer()
        service._policy_server = running
        service.reconfigure(
            _config({"model_path": str(tmp_path / "v2"), "policy_server_autoclose": True}), {}
        )
        await asyncio.sleep(0.05)
        assert starts == [str(tmp_path / "v1")] and running.stops == 0
        release.set()
        await service._server_task
        assert starts == [str(tmp_path / "v1"), str(tmp_path / "v2")] and running.stops == 1
        await service.close()

    asyncio.run(scenario())


def test_connect_failure_reports_managed_server_state(monkeypatch):
    async def scenario():
        service = CosmosPolicy("cosmos")
        service._config = CosmosConfig(model_path="/tmp/policy-model", connect_timeout_sec=0.2)
        service._server_error = "RuntimeError: did not become ready"
        monkeypatch.setattr(service._policy_server, "status", lambda: {"state": "absent"})
        with pytest.raises(RuntimeError, match="managed policy server is absent; last start error: RuntimeError: did not become ready"):
            await service.infer(_inputs())

    asyncio.run(scenario())
