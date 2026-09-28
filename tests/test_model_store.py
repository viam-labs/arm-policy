from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from models import model_store
from models.model_store import (
    ModelSource,
    ModelStore,
    parse_model_source,
    validate_data_root,
    validate_repository,
    validate_revision,
)
from models.policy_server import PolicyServerSupervisor

COMMIT = "a" * 40
FRAMEWORK_COMMIT = "efba2076c80648663a89a7fdbc92eb262fae39ac"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(root: Path, *, remote_marker: bool) -> None:
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
    if remote_marker:
        (root / ".cosmos-policy-complete.json").write_text(
            json.dumps({"schema_version": 1, "repository": "owner/policy", "resolved_revision": COMMIT, "provenance": {"repository": "owner/training", "revision": "b" * 40}})
        )


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("owner/policy", "model:owner/policy"),
        ("model:owner/policy", "model:owner/policy"),
        ("bucket:owner/policy:models/run-1", "bucket:owner/policy:models/run-1"),
    ],
)
def test_parse_model_source_canonicalizes_valid_values(value, canonical):
    assert parse_model_source(value).canonical == canonical


@pytest.mark.parametrize(
    "value",
    ["", "model:owner/policy:extra", "bucket:owner/policy", "bucket:owner/policy:", "bucket:owner/policy:/absolute", "bucket:owner/policy:../escape", "bucket:owner/policy:a/./b", "bucket:owner/policy:a\\b", "other:owner/policy"],
)
def test_parse_model_source_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_model_source(value)
    with pytest.raises(ValueError):
        ModelSource("bucket", "owner/policy", None)


def test_model_download_is_atomic_idempotent_and_list_is_offline(tmp_path, monkeypatch):
    source = tmp_path / "published"
    source.mkdir()
    _export(source, remote_marker=True)
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
    model = parse_model_source("owner/policy")
    first = store.download(model, COMMIT, "secret")
    assert first.response()["model_source"] == "model:owner/policy"
    assert first.requested_revision == first.resolved_revision == COMMIT
    assert Path(first.path, ".complete.json").is_file()
    assert store.download(model, COMMIT, "secret") == first
    assert calls.count(("resolve", COMMIT)) == 1
    assert [item.response() for item in store.list()] == [first.response()]
    assert calls.count(("resolve", COMMIT)) == 1


def test_model_download_rejects_unpublished_or_invalid_exports(tmp_path, monkeypatch):
    source = tmp_path / "published"
    source.mkdir()
    _export(source, remote_marker=False)

    class FakeHub:
        def __init__(self, repository, token=None):
            pass

        def resolve(self, revision):
            return COMMIT, [path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()]

        def download(self, revision, name, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)

    monkeypatch.setattr(model_store, "HuggingFaceClient", FakeHub)
    store = ModelStore(tmp_path / "persistent")
    with pytest.raises(ValueError, match="not published"):
        store.download(parse_model_source("model:owner/policy"), COMMIT)
    assert store.list() == []


def test_bucket_download_syncs_with_token_then_publishes_validated_record(tmp_path, monkeypatch):
    source = tmp_path / "bucket-export"
    source.mkdir()
    _export(source, remote_marker=False)
    calls = []

    def fake_run(argv, *, check, env):
        calls.append((argv, check, env["HF_TOKEN"]))
        shutil.copytree(source, argv[-1], dirs_exist_ok=True)

    monkeypatch.setattr(model_store.subprocess, "run", fake_run)
    store = ModelStore(tmp_path / "persistent")
    bucket = parse_model_source("bucket:owner/policy:models/run-1")
    first = store.download(bucket, None, "bucket-secret")
    assert first.response()["model_source"] == bucket.canonical
    assert first.requested_revision is first.resolved_revision is None
    assert calls[0][0] == ["uvx", "hf@latest", "buckets", "sync", "hf://buckets/owner/policy/models/run-1", calls[0][0][-1]]
    assert calls[0][1:] == (True, "bucket-secret")
    assert store.download(bucket, None, "bucket-secret") == first
    assert len(calls) == 1
    assert [item.response() for item in store.list()] == [first.response()]


def test_bucket_invalid_download_is_not_discoverable_and_revisions_are_rejected(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, *, check, env):
        calls.append(argv)
        Path(argv[-1], "checkpoint.json").write_text("{}")

    monkeypatch.setattr(model_store.subprocess, "run", fake_run)
    store = ModelStore(tmp_path / "persistent")
    bucket = parse_model_source("bucket:owner/policy:models/run-1")
    with pytest.raises(ValueError, match="missing required files"):
        store.download(bucket, None)
    assert store.list() == []
    with pytest.raises(ValueError, match="cannot specify"):
        store.download(bucket, COMMIT)
    with pytest.raises(ValueError, match="immutable"):
        store.download(parse_model_source("owner/policy"), None)
    assert len(calls) == 1


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
    program = "import os, socket, signal, time; assert os.environ.get('HF_TOKEN') == 'server-secret'; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', " + str(port) + ")); s.listen(); signal.signal(signal.SIGTERM, lambda *_: exit(0)); time.sleep(30)"
    supervisor = PolicyServerSupervisor(tmp_path, "fixture")
    started = supervisor.start(
        [sys.executable, "-c", program],
        "127.0.0.1",
        port,
        3,
        {"HF_TOKEN": "server-secret"},
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
