from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from models import model_store
from models.model_store import (
    ModelStore,
    validate_data_root,
    validate_repository,
    validate_revision,
)
from models.policy import Policy
from models.policy_server import PolicyServerSupervisor

COMMIT = "a" * 40
FRAMEWORK_COMMIT = "efba2076c80648663a89a7fdbc92eb262fae39ac"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(root: Path) -> None:
    (root / "runtime" / "vae").mkdir(parents=True)
    (root / "runtime" / "training_config.yaml").write_text("training: fixture\n")
    (root / "runtime" / "vae" / "fixture.bin").write_bytes(b"vae")
    contract = {"schema_version": 2}
    (root / "policy_contract.json").write_text(json.dumps(contract))
    (root / "checkpoint.json").write_text(json.dumps({"runtime_manifest": "runtime_manifest.json"}))
    (root / "config.json").write_text("{}")
    (root / "export_manifest.json").write_text(json.dumps({"framework_commit": FRAMEWORK_COMMIT}))
    (root / "model.safetensors.index.json").write_text("{}")
    (root / ".cosmos-policy-complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "owner/policy",
                "resolved_revision": COMMIT,
                "provenance": {"repository": "owner/training", "revision": "b" * 40},
            }
        )
    )
    runtime = {
        "schema_version": 2,
        "framework_commit": FRAMEWORK_COMMIT,
        "training_config": {"path": "runtime/training_config.yaml", "sha256": _hash(root / "runtime" / "training_config.yaml")},
        "policy_contract": {"path": "policy_contract.json", "sha256": _hash(root / "policy_contract.json")},
        "vae": {"path": "runtime/vae/fixture.bin", "sha256": _hash(root / "runtime" / "vae" / "fixture.bin")},
    }
    (root / "runtime_manifest.json").write_text(json.dumps(runtime))


def test_atomic_download_is_idempotent_and_list_is_offline(tmp_path, monkeypatch):
    source = tmp_path / "published"
    source.mkdir()
    _export(source)
    calls = []

    class FakeHub:
        def __init__(self, repository, token=None):
            assert repository == "owner/policy"
            assert token == "secret"

        def resolve(self, revision):
            calls.append(("resolve", revision))
            return COMMIT, [path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()]

        def download(self, revision, name, destination):
            calls.append(("download", name))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)

    monkeypatch.setattr(model_store, "HuggingFaceClient", FakeHub)
    store = ModelStore(tmp_path / "persistent")
    first = store.download("owner/policy", COMMIT, "secret")
    assert first.validation_state == "validated"
    assert Path(first.path, ".complete.json").is_file()
    second = store.download("owner/policy", COMMIT, "secret")
    assert second == first
    assert calls.count(("resolve", COMMIT)) == 1
    assert [item.response() for item in store.list()] == [first.response()]
    policy = Policy("fixture")
    policy._config.model_data_root = str(tmp_path / "persistent")
    policy._config.model_repository = "owner/policy"
    policy._config.model_revision = COMMIT
    policy._config.hf_token = "secret"
    assert asyncio.run(policy.do_command({"command": "models/download"})) == first.response()
    response = asyncio.run(policy.do_command({"command": "models/list"}))
    assert response == {"models": [first.response()]}
    assert calls.count(("resolve", COMMIT)) == 1


@pytest.mark.parametrize(
    ("marker", "message"),
    [(None, "not published"), ({"schema_version": 1}, "repository does not match")],
)
def test_download_rejects_missing_or_mismatched_publication_marker(
    tmp_path, monkeypatch, marker, message
):
    source = tmp_path / "published"
    source.mkdir()
    _export(source)
    if marker is not None:
        (source / ".cosmos-policy-complete.json").write_text(json.dumps(marker))

    class FakeHub:
        def __init__(self, repository, token=None):
            pass

        def resolve(self, revision):
            files = [path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()]
            if marker is None:
                files.remove(".cosmos-policy-complete.json")
            return COMMIT, files

        def download(self, revision, name, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)

    monkeypatch.setattr(model_store, "HuggingFaceClient", FakeHub)
    store = ModelStore(tmp_path / "persistent")
    with pytest.raises(ValueError, match=message):
        store.download("owner/policy", COMMIT)
    assert store.list() == []

def test_model_configuration_validation_is_strict(tmp_path):
    assert validate_repository("owner/policy") == "owner/policy"
    assert validate_revision(COMMIT) == COMMIT
    assert validate_data_root(str(tmp_path)) == tmp_path
    with pytest.raises(ValueError):
        validate_repository("owner/policy/extra")
    with pytest.raises(ValueError):
        validate_revision("main")
    with pytest.raises(ValueError):
        validate_data_root("relative")


def test_supervisor_start_status_stop_is_idempotent(tmp_path):
    port = 18191
    program = "import socket, signal, time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', " + str(port) + ")); s.listen(); signal.signal(signal.SIGTERM, lambda *_: exit(0)); time.sleep(30)"
    supervisor = PolicyServerSupervisor(tmp_path, "fixture")
    started = supervisor.start([sys.executable, "-c", program], "127.0.0.1", port, 3)
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
