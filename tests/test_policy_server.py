import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_model_store import _published_export
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

import models.policy as policy_module
import models.policy_server as policy_server_module
from models import model_store
from models.policy import Policy
from models.policy_server import PolicyServerSupervisor

_SERVER_PROGRAM = """
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import threading
import time

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK\\n")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

server = HTTPServer(("127.0.0.1", int(sys.argv[1])), HealthHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[2]))
server.server_close()
raise SystemExit(int(sys.argv[3]))
"""


class CapturingLogger:
    def __init__(self):
        self.errors = []
        self.infos = []

    def error(self, message, *args):
        self.errors.append(message % args)

    def info(self, message, *args):
        self.infos.append(message % args)


class CapturingSupervisor:
    def __init__(self):
        self.commands = []

    def start(self, command, host, port, readiness_timeout, environment):
        self.commands.append(command)
        return {"state": "healthy", "pid": 123}

    def poll_failure(self):
        return None

    def _has_state(self):
        return False


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _server_command(port, lifetime, exit_status):
    return [sys.executable, "-c", _SERVER_PROGRAM, str(port), str(lifetime), str(exit_status)]


def test_health_ready_requires_expected_response(monkeypatch):
    class Response:
        def __init__(self, status, body):
            self.status = status
            self.body = body

        def read(self, size):
            assert size == len(b"OK\n") + 1
            return self.body

    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == (
                "127.0.0.1",
                8000,
                policy_server_module._HEALTH_CHECK_TIMEOUT,
            )
            self.closed = False

        def request(self, method, path):
            assert (method, path) == ("GET", "/healthz")

        def getresponse(self):
            return Response(self.status, self.body)

        def close(self):
            self.closed = True
            closed.append(self)

    closed = []
    monkeypatch.setattr(policy_server_module.http.client, "HTTPConnection", Connection)
    for status, body, expected in [
        (200, b"OK\n", True),
        (200, b"<html>", False),
        (503, b"OK\n", False),
    ]:
        Connection.status = status
        Connection.body = body
        assert policy_server_module._health_ready("127.0.0.1", 8000) is expected
    assert all(connection.closed for connection in closed)


@pytest.mark.parametrize("failure", [OSError, policy_server_module.http.client.BadStatusLine])
def test_health_ready_rejects_unavailable_or_malformed_connections(monkeypatch, failure):
    class Connection:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def request(self, method, path):
            raise failure("unavailable")

        def close(self):
            self.closed = True
            closed.append(self)

    closed = []
    monkeypatch.setattr(policy_server_module.http.client, "HTTPConnection", Connection)

    assert not policy_server_module._health_ready("127.0.0.1", 8000)
    assert closed[0].closed

def test_startup_exit_includes_status_and_redacts_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "inherited-secret")
    program = (
        "import os, sys; "
        "print('HF_TOKEN=' + os.environ['HF_TOKEN']); "
        "print('HUGGINGFACE_HUB_TOKEN=' + os.environ['HUGGINGFACE_HUB_TOKEN']); "
        "print('Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz'); "
        "raise SystemExit(23)"
    )
    supervisor = PolicyServerSupervisor(
        tmp_path, "startup", credential_values=("configured-secret",)
    )

    with pytest.raises(RuntimeError) as exc_info:
        supervisor.start(
            [sys.executable, "-c", program],
            "127.0.0.1",
            _free_port(),
            readiness_timeout=1,
            environment={"HF_TOKEN": "configured-secret"},
        )

    error = str(exc_info.value)
    assert "exit_status=23" in error
    assert str(supervisor.log_path) in error
    assert "configured-secret" not in error
    assert "inherited-secret" not in error
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in error


def test_tail_discards_a_leading_partial_credential_line(tmp_path):
    secret = "configured-secret"
    supervisor = PolicyServerSupervisor(tmp_path, "partial", credential_values=(secret,))
    supervisor.root.mkdir(parents=True)
    safe_line = b"\ncomplete final line\n"
    trailing = (
        policy_server_module._TAIL_BYTES
        - len(secret.encode())
        + 3
        - len(safe_line)
    )
    supervisor.log_path.write_bytes(
        b"previous line\n"
        + secret.encode()
        + (b"x" * trailing)
        + safe_line
    )

    tail = supervisor._log_tail()

    assert secret[3:] not in tail
    assert "complete final line" in tail


def test_redacted_tail_remains_bounded(tmp_path):
    supervisor = PolicyServerSupervisor(tmp_path, "bounded", credential_values=("a",))
    supervisor.root.mkdir(parents=True)
    supervisor.log_path.write_bytes(b"a" * policy_server_module._TAIL_BYTES)

    tail = supervisor._log_tail()

    assert len(tail) == policy_server_module._TAIL_BYTES
    assert "a" not in tail

def test_policy_logs_redacted_startup_failure(tmp_path, monkeypatch):
    async def scenario():
        policy = Policy("startup-log")
        policy._policy_server = PolicyServerSupervisor(tmp_path, "startup-log")
        logger = CapturingLogger()
        monkeypatch.setattr(policy, "logger", logger)
        policy._config.hf_token = "configured-secret"
        policy._config.policy_server_port = _free_port()
        model = tmp_path / "model"
        model.mkdir()
        policy._config.model_source = "bucket:owner/policy:models/startup-log"
        policy._require_model_store = lambda: SimpleNamespace(
            get=lambda source, revision: SimpleNamespace(path=str(model))
        )
        program = (
            "import os, sys; print('HF_TOKEN=' + os.environ['HF_TOKEN']); "
            "raise SystemExit(9)"
        )
        monkeypatch.setattr(
            policy_module,
            "runtime_command",
            lambda *args: [sys.executable, "-c", program],
        )

        with pytest.raises(RuntimeError, match="exit_status=9"):
            await policy.do_command({"command": "policy_server", "action": "start"})

        assert len(logger.errors) == 1
        assert "failed during startup" in logger.errors[0]
        assert "configured-secret" not in logger.errors[0]
        assert len(logger.infos) == 1
        assert "start requested" in logger.infos[0]

    asyncio.run(scenario())


def test_policy_logs_server_start_and_readiness(tmp_path, monkeypatch):
    async def scenario():
        port = _free_port()
        model = tmp_path / "model"
        model.mkdir()
        policy = Policy("startup-ready-log")
        policy._policy_server = PolicyServerSupervisor(tmp_path, "startup-ready-log")
        policy._config.policy_server_port = port
        policy._config.model_source = "bucket:owner/policy:models/startup-ready-log"
        policy._require_model_store = lambda: SimpleNamespace(
            get=lambda source, revision: SimpleNamespace(path=str(model))
        )
        logger = CapturingLogger()
        monkeypatch.setattr(policy, "logger", logger)
        monkeypatch.setattr(
            policy_module,
            "runtime_command",
            lambda *args: _server_command(port, 30, 0),
        )

        result = await policy.do_command(
            {"command": "policy_server", "action": "start"}
        )

        assert result["state"] == "healthy"
        assert len(logger.infos) == 2
        assert "start requested" in logger.infos[0]
        assert "ready" in logger.infos[1]
        assert f"pid={result['pid']}" in logger.infos[1]
        assert logger.errors == []
        await policy.do_command({"command": "policy_server", "action": "stop"})

    asyncio.run(scenario())


@pytest.mark.parametrize("source_value", ["model:owner/policy", "bucket:owner/policy:models/run-1"])
def test_configured_start_reads_only_selected_receipt_and_contract(tmp_path, monkeypatch, source_value):
    store, source, selected = _published_export(tmp_path, source_value)
    other_source = source_value if source.kind == "model" else "bucket:owner/policy:models/run-2"
    _, unrelated_source, unrelated = _published_export(tmp_path, other_source, revision="b" * 40)
    assert store.get(unrelated_source, unrelated.requested_revision) == unrelated
    target = Path(selected.path)
    policy = Policy("targeted-start")
    policy._config.model_data_root = str(tmp_path / "persistent")
    policy._config.model_source = source.canonical
    policy._config.model_revision = selected.requested_revision
    supervisor = CapturingSupervisor()
    policy._policy_server = supervisor
    opened = set()
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.is_relative_to(target), f"startup must not read outside selected export: {path}"
        opened.add(path)
        return original_open(path, *args, **kwargs)

    def confined_traversal(function):
        def guarded(path, *args, **kwargs):
            assert Path(path).is_relative_to(target), f"startup must not traverse outside selected export: {path}"
            return function(path, *args, **kwargs)

        return guarded

    def forbidden_list(*args, **kwargs):
        raise AssertionError("configured startup must not list exports")

    async def scenario():
        try:
            return await policy.do_command({"command": "policy_server", "action": "start"})
        finally:
            policy._detach_policy_server_watcher()

    with monkeypatch.context() as guard:
        guard.setattr(model_store.ModelStore, "list", forbidden_list)
        guard.setattr(Path, "open", guarded_open)
        for method in ("glob", "rglob", "iterdir"):
            guard.setattr(Path, method, confined_traversal(getattr(Path, method)))
        guard.setattr(model_store.os, "scandir", confined_traversal(model_store.os.scandir))
        result = asyncio.run(scenario())

    assert result["state"] == "healthy"
    assert len(supervisor.commands) == 1
    command = supervisor.commands[0]
    assert command[command.index("--model") + 1] == selected.path
    assert opened == {
        target / ".complete.json",
        target / "policy_contract.json",
    }


@pytest.mark.parametrize("source_value", ["model:owner/policy", "bucket:owner/policy:models/run-1"])
@pytest.mark.parametrize("payload", ["model-00001-of-00001.safetensors", "runtime/vae/fixture.bin", "checkpoint.json"])
@pytest.mark.parametrize("mutation", ["tampered", "missing"])
def test_configured_start_trusts_receipt_for_payloads(tmp_path, source_value, payload, mutation):
    store, source, selected = _published_export(tmp_path, source_value)
    path = Path(selected.path) / payload
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(b"tampered")
    policy = Policy("receipt-start")
    policy._config.model_data_root = str(tmp_path / "persistent")
    policy._config.model_source = source.canonical
    policy._config.model_revision = selected.requested_revision
    supervisor = CapturingSupervisor()
    policy._policy_server = supervisor

    async def scenario():
        try:
            return await policy.do_command({"command": "policy_server", "action": "start"})
        finally:
            policy._detach_policy_server_watcher()

    assert asyncio.run(scenario())["state"] == "healthy"
    assert len(supervisor.commands) == 1


@pytest.mark.parametrize("source_value", ["model:owner/policy", "bucket:owner/policy:models/run-1"])
@pytest.mark.parametrize("payload", [".complete.json", "policy_contract.json"])
@pytest.mark.parametrize("mutation", ["tampered", "missing"])
def test_configured_start_rejects_invalid_receipt_or_contract(tmp_path, source_value, payload, mutation):
    store, source, selected = _published_export(tmp_path, source_value)
    path = Path(selected.path) / payload
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(b"tampered")
    assert store.get(source, selected.requested_revision) is None
    policy = Policy("invalid-receipt-start")
    policy._config.model_data_root = str(tmp_path / "persistent")
    policy._config.model_source = source.canonical
    policy._config.model_revision = selected.requested_revision
    supervisor = CapturingSupervisor()
    policy._policy_server = supervisor

    async def scenario():
        try:
            with pytest.raises(ValueError, match="configured model has not been downloaded and validated"):
                await policy.do_command({"command": "policy_server", "action": "start"})
        finally:
            policy._detach_policy_server_watcher()

    asyncio.run(scenario())
    assert supervisor.commands == []


def test_watcher_logs_owned_exit_once_with_exit_status(tmp_path, monkeypatch):
    async def scenario():
        port = _free_port()
        supervisor = PolicyServerSupervisor(tmp_path, "watch")
        supervisor.start(_server_command(port, 0.2, 7), "127.0.0.1", port, 2)
        policy = Policy("watch")
        logger = CapturingLogger()
        monkeypatch.setattr(policy, "logger", logger)
        policy._policy_server = supervisor
        policy._ensure_policy_server_watcher()

        deadline = time.monotonic() + 3
        while not logger.errors and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

        assert len(logger.errors) == 1
        assert "exit_status=7" in logger.errors[0]
        assert str(supervisor.log_path) in logger.errors[0]
        await asyncio.sleep(0.3)
        assert len(logger.errors) == 1
        assert policy._policy_server_watch_task is None

    asyncio.run(scenario())


def test_expected_stop_is_silent_and_releases_watcher(tmp_path, monkeypatch):
    async def scenario():
        port = _free_port()
        supervisor = PolicyServerSupervisor(tmp_path, "stop")
        supervisor.start(_server_command(port, 30, 0), "127.0.0.1", port, 2)
        policy = Policy("stop")
        logger = CapturingLogger()
        monkeypatch.setattr(policy, "logger", logger)
        policy._policy_server = supervisor
        policy._ensure_policy_server_watcher()

        await asyncio.sleep(0.05)
        await policy.do_command({"command": "policy_server", "action": "stop"})
        deadline = time.monotonic() + 2
        while policy._policy_server_watch_task is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

        assert logger.errors == []
        assert policy._policy_server_watch_task is None

    asyncio.run(scenario())


def test_recovered_zombie_reports_exit_without_pid_reuse_kill(tmp_path, monkeypatch):
    supervisor = PolicyServerSupervisor(tmp_path, "recovered")
    supervisor.root.mkdir(parents=True)
    supervisor.state_path.write_text(
        json.dumps(
            {
                "pid": 1234,
                "start_ticks": "original",
                "host": "127.0.0.1",
                "port": 8000,
                "ready_deadline": 0,
            }
        )
    )
    monkeypatch.setattr(
        policy_server_module,
        "_process_info",
        lambda pid: policy_server_module._ProcessInfo("original", "Z"),
    )

    assert supervisor.status()["state"] == "exited"
    failure = supervisor.poll_failure()
    assert failure is not None
    assert failure.exit_status is None

    monkeypatch.setattr(
        policy_server_module,
        "_process_info",
        lambda pid: policy_server_module._ProcessInfo("reused", "S"),
    )
    assert supervisor.poll_failure() is None
    assert supervisor.status()["state"] == "absent"


def test_reconfigure_detaches_watcher_without_stopping_retained_server(
    tmp_path, monkeypatch
):
    class RetainedServer:
        def __init__(self):
            self.stop_calls = 0

        def poll_failure(self):
            return None

        def _has_state(self):
            return True

        def stop(self):
            self.stop_calls += 1

    async def scenario():
        policy = Policy("reconfigure")
        retained = RetainedServer()
        policy._policy_server = retained
        policy._ensure_policy_server_watcher()
        watcher = policy._policy_server_watch_task
        assert watcher is not None
        parsed = SimpleNamespace(
            world=None,
            arm="arm",
            gripper="gripper",
            fixed_camera="fixed",
            fixed_camera_2="",
            wrist_camera="wrist",
            hf_token=None,
            model_data_root=str(tmp_path),
            policy_server_autoclose=False,
            binary_gripper_initial_state=0.0,
        )
        monkeypatch.setattr(
            policy_module.Config,
            "model_validate",
            classmethod(lambda cls, config: parsed),
        )
        monkeypatch.setattr(
            Policy, "_dependency", staticmethod(lambda *args: object())
        )
        monkeypatch.setattr(Policy, "_advance_readiness", lambda self: True)

        policy.reconfigure(
            ComponentConfig(name="reconfigure", attributes=dict_to_struct({})), {}
        )
        await asyncio.sleep(0)

        assert retained.stop_calls == 0
        assert watcher.done()
        policy._stop_command = lambda: asyncio.sleep(0, result={})
        await policy.close()

    asyncio.run(scenario())
