import asyncio
import json
import threading

import numpy as np
import pytest
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


class RecordingServer:
    def __init__(self):
        self.starts = []
        self.stops = 0

    def start(self, command, host, port, readiness_timeout, environment=None):
        self.starts.append(command)
        return {"state": "healthy", "pid": 1}

    def stop(self):
        self.stops += 1
        return {"state": "absent", "stopped": True}

    def close(self):
        return self.stop()

    def running(self):
        return False

    def status(self):
        return {"state": "absent"}


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


def test_infer_round_trips_tensors_connecting_per_call():
    async def scenario():
        actions = np.full((32, 7), 0.25, dtype=np.float64)
        socket = FakeSocket(pack_message({"action": actions, "server_timing": {"infer_ms": 3.5}}))
        service = SocketCosmos(socket)

        outputs = await service.infer(_inputs())
        assert socket.closed.is_set()
        await service.infer(_inputs("again"))

        assert service.connections == 2
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


def test_mlmodel_gripper_convention_is_independent_of_cosmos_wire_convention():
    async def scenario():
        actions = np.zeros((32, 7), dtype=np.float32)
        actions[:, 6] = 0.25
        service = SocketCosmos(FakeSocket(pack_message({"action": actions})))
        inputs = _inputs()
        inputs[GRIPPER_TENSOR][0] = 0.2
        outputs = await service.infer(inputs)
        sent = unpack_message(service.socket.sent[0])
        assert sent[GRIPPER_TENSOR].tolist() == pytest.approx([0.8])
        assert inputs[GRIPPER_TENSOR].tolist() == pytest.approx([0.2])
        np.testing.assert_allclose(outputs[ACTION_TENSOR][:, 6], 0.75)
        np.testing.assert_allclose(outputs[ACTION_TENSOR][:, :6], 0.0)

    asyncio.run(scenario())


def test_close_cancels_active_inference_and_rejects_queued_calls():
    async def scenario():
        connected = asyncio.Event()
        socket = FakeSocket(None, block=True)
        service = SocketCosmos(socket)

        async def connect():
            service.connections += 1
            connected.set()
            return socket

        service._connect = connect
        active = asyncio.create_task(service.infer(_inputs()))
        await asyncio.wait_for(connected.wait(), 1)
        queued = asyncio.create_task(service.infer(_inputs()))
        await asyncio.sleep(0)
        await service.close()
        try:
            assert socket.closed.is_set()
            with pytest.raises(asyncio.CancelledError):
                await active
            with pytest.raises(RuntimeError, match="closed"):
                await queued
            with pytest.raises(RuntimeError, match="closed"):
                await service.infer(_inputs())
            assert service.connections == 1
        finally:
            active.cancel()
            queued.cancel()
            await asyncio.gather(active, queued, return_exceptions=True)

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
def test_infer_rejects_malformed_inputs_before_connecting(mutate, error):
    async def scenario():
        socket = FakeSocket(pack_message({"action": np.zeros((32, 7))}))
        service = SocketCosmos(socket)
        inputs = _inputs()
        mutate(inputs)
        with pytest.raises(ValueError, match=error):
            await service.infer(inputs)
        assert service.connections == 0 and socket.sent == []

    asyncio.run(scenario())


def test_infer_rejects_text_errors_and_bad_shapes():
    async def scenario():
        service = SocketCosmos(FakeSocket("traceback from the server"))
        with pytest.raises(RuntimeError, match="text error"):
            await service.infer(_inputs())
        service.socket = FakeSocket(pack_message({"action": np.zeros((31, 7))}))
        with pytest.raises(ValueError, match="shape"):
            await service.infer(_inputs())

    asyncio.run(scenario())


def test_infer_timeout_closes_the_socket():
    async def scenario():
        socket = FakeSocket(None, block=True)
        service = SocketCosmos(socket)
        with pytest.raises(asyncio.TimeoutError):
            await service.infer(_inputs())
        assert socket.closed.is_set()

    asyncio.run(scenario())


def test_contract_and_metadata_wait_for_the_deployed_package(tmp_path):
    _, contract, _ = _write_contract(tmp_path)
    package = tmp_path / "package"

    async def scenario():
        service = CosmosPolicy("cosmos")
        service.reconfigure(_config({"model_path": str(package), "server_uri": "ws://gpu:9000"}), {})
        with pytest.raises(RuntimeError, match="not available yet"):
            await service.metadata()

        package.mkdir()
        (package / "policy_contract.json").write_text(json.dumps(contract))
        metadata = await service.metadata()
        action_info = next(info for info in metadata.output_info if info.name == ACTION_TENSOR)
        assert json.loads(struct_to_dict(action_info.extra)[CONTRACT_EXTRA]) == contract
        assert metadata.description.endswith("ws://gpu:9000")
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
        package_reference="org/pkg:latest",
    )
    assert parsed.model_path == "/models/x"
    assert parsed.uri == "ws://[::1]:9001"
    assert parsed.model_data_root == "/opt/viam/arm-policy"
    assert parsed.policy_server_readiness_timeout_sec == 900.0


def test_new_starts_the_server_once_and_close_stops_it(tmp_path, monkeypatch):
    server = RecordingServer()
    monkeypatch.setattr(
        "viam_arm_policy.cosmos_policy.PolicyServerSupervisor", lambda root, name: server
    )
    monkeypatch.setattr("viam_arm_policy.cosmos_policy.validate_runtime_manifest", lambda model: {})

    async def scenario():
        service = CosmosPolicy.new(_config({"model_path": str(tmp_path / "model")}), {})
        await service._server_task
        assert len(server.starts) == 1
        assert server.starts[0][server.starts[0].index("--model") + 1] == str(tmp_path / "model")
        with pytest.raises(RuntimeError, match="already configured"):
            service.reconfigure(_config({"model_path": str(tmp_path / "model")}), {})
        await service.close()
        await service.close()
        assert server.stops == 1

    asyncio.run(scenario())


def test_external_server_uri_starts_nothing(tmp_path, monkeypatch):
    server = RecordingServer()
    monkeypatch.setattr(
        "viam_arm_policy.cosmos_policy.PolicyServerSupervisor", lambda root, name: server
    )

    async def scenario():
        service = CosmosPolicy.new(
            _config({"model_path": str(tmp_path), "server_uri": "ws://gpu:9000"}), {}
        )
        await asyncio.sleep(0)
        assert service._server_task is None and server.starts == []
        await service.close()

    asyncio.run(scenario())


def test_close_before_start_worker_launches_prevents_child_creation(tmp_path, monkeypatch):
    from test_policy_server import _free_port, _server_command

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    marker = tmp_path / "spawned"
    port = _free_port()
    command = _server_command(port, 60, 0)
    command[2] = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + command[2]
    monkeypatch.setattr("viam_arm_policy.cosmos_policy.runtime_command", lambda *args: command)
    monkeypatch.setattr("viam_arm_policy.cosmos_policy.validate_runtime_manifest", lambda model: {})

    async def scenario():
        service = CosmosPolicy.new(
            _config({
                "model_path": str(tmp_path),
                "model_data_root": str(tmp_path),
                "policy_server_port": port,
                "policy_server_readiness_timeout_sec": 2,
            }), {}
        )
        supervisor = service._policy_server
        start = supervisor.start

        def delayed_start(*args, **kwargs):
            entered.set()
            try:
                release.wait(5)
                return start(*args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(supervisor, "start", delayed_start)
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await service.close()
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            assert not marker.exists()
        finally:
            release.set()
            await asyncio.to_thread(supervisor.stop)
            await service.close()

    asyncio.run(scenario())


def test_connect_failure_reports_managed_server_state(monkeypatch):
    async def scenario():
        service = CosmosPolicy("cosmos")
        service._config = CosmosConfig(model_path="/tmp/policy-model", connect_timeout_sec=0.2)
        service._server_error = "RuntimeError: did not become ready"
        monkeypatch.setattr(service._policy_server, "status", lambda: {"state": "absent"})
        with pytest.raises(
            RuntimeError,
            match="managed policy server is absent; last error: RuntimeError: did not become ready",
        ):
            await service.infer(_inputs())

    asyncio.run(scenario())
