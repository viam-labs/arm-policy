from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from models import model_export
from models.model_export import (
    validate_data_root,
    validate_model_path,
    validate_runtime_manifest,
)
from models.policy_server import PolicyServerSupervisor

FRAMEWORK_COMMIT = "efba2076c80648663a89a7fdbc92eb262fae39ac"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(root: Path) -> None:
    (root / "runtime" / "vae").mkdir(parents=True)
    (root / "runtime" / "training_config.yaml").write_text("training: fixture\n")
    (root / "runtime" / "vae" / "fixture.bin").write_bytes(b"vae")
    (root / "policy_contract.json").write_text(json.dumps({"schema_version": 3}))
    (root / "checkpoint.json").write_text(json.dumps({"runtime_manifest": "runtime_manifest.json"}))
    (root / "config.json").write_text("{}")
    (root / "export_manifest.json").write_text(json.dumps({"framework_commit": FRAMEWORK_COMMIT}))
    (root / "model.safetensors.index.json").write_text("{}")
    runtime = {
        "schema_version": 2,
        "framework_commit": FRAMEWORK_COMMIT,
        "training_config": {"path": "runtime/training_config.yaml", "sha256": _hash(root / "runtime" / "training_config.yaml")},
        "policy_contract": {"path": "policy_contract.json", "sha256": _hash(root / "policy_contract.json")},
        "vae": {"path": "runtime/vae/fixture.bin", "sha256": _hash(root / "runtime" / "vae" / "fixture.bin")},
    }
    (root / "runtime_manifest.json").write_text(json.dumps(runtime))


def _deployed_export(root: Path) -> Path:
    """A portable export as viam-server extracts an ml_model package."""
    root.mkdir(parents=True)
    _export(root)
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return root


def test_runtime_manifest_validation_hashes_only_small_manifest_entries(tmp_path, monkeypatch):
    root = _deployed_export(tmp_path / "package")
    hashed = []
    original = model_export._sha256
    monkeypatch.setattr(
        model_export, "_sha256", lambda path: hashed.append(path.name) or original(path)
    )

    hashes = validate_runtime_manifest(root)

    assert set(hashes) == {
        "runtime/training_config.yaml",
        "policy_contract.json",
    }
    assert "model-00001-of-00001.safetensors" not in hashed
    assert "fixture.bin" not in hashed


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda root: (root / "config.json").unlink(), "missing required files"),
        (lambda root: (root / "runtime" / "vae" / "fixture.bin").unlink(), "missing for vae"),
        (lambda root: (root / "policy_contract.json").write_text(json.dumps({"schema_version": 3, "x": 1})), "checksum failed for policy_contract"),
        (lambda root: (root / "checkpoint.json").write_text("{}"), "does not reference"),
        (lambda root: (root / "export_manifest.json").write_text("{}"), "revisions conflict"),
        (lambda root: (root / "runtime_manifest.json").write_text("[]"), "expected an object"),
    ],
)
def test_runtime_manifest_validation_rejects_invalid_exports(tmp_path, mutation, error):
    root = _deployed_export(tmp_path / "package")
    mutation(root)
    with pytest.raises(ValueError, match=error):
        validate_runtime_manifest(root)


def test_runtime_manifest_paths_cannot_escape_export(tmp_path):
    root = _deployed_export(tmp_path / "package")
    runtime_path = root / "runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["vae"]["path"] = "../outside.bin"
    runtime_path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="escapes"):
        validate_runtime_manifest(root)


def test_path_configuration_validation_is_strict(tmp_path):
    assert validate_data_root(str(tmp_path)) == tmp_path
    assert validate_model_path(str(tmp_path)) == tmp_path
    for validate in (validate_data_root, validate_model_path):
        with pytest.raises(ValueError, match="absolute"):
            validate("relative")
        with pytest.raises(ValueError, match="non-empty"):
            validate(" ")


def test_supervisor_start_status_stop_is_idempotent(tmp_path):
    port = 18191
    program = """
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

assert os.environ.get("POLICY_SERVER_FIXTURE") == "server-env"

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

server = HTTPServer(("127.0.0.1", 18191), HealthHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
signal.signal(signal.SIGTERM, lambda *_: (server.shutdown(), exit(0)))
while True:
    time.sleep(30)
"""
    supervisor = PolicyServerSupervisor(tmp_path, "fixture")
    started = supervisor.start(
        [sys.executable, "-c", program],
        "127.0.0.1",
        port,
        3,
        {"POLICY_SERVER_FIXTURE": "server-env"},
    )
    assert started["state"] == "healthy"
    assert supervisor.start([sys.executable, "-c", program], "127.0.0.1", port, 3)["idempotent"]
    assert supervisor.status()["state"] == "healthy"
    assert supervisor.stop(grace_seconds=1)["state"] == "absent"

def test_supervisor_readiness_timeout_stops_process(tmp_path):
    program = "import signal, time; signal.signal(signal.SIGTERM, lambda *_: None); time.sleep(30)"
    supervisor = PolicyServerSupervisor(tmp_path, "unready")
    with pytest.raises(RuntimeError, match="did not become ready"):
        supervisor.start([sys.executable, "-c", program], "127.0.0.1", 18192, 0.2)
    assert supervisor.status()["state"] == "absent"
