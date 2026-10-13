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
BASE_CHECKPOINT = {
    "schema_version": 1,
    "repo_id": "nvidia/Cosmos3-Edge-Policy-DROID",
    "commit_sha": "b" * 40,
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(root: Path) -> None:
    (root / "runtime" / "vae").mkdir(parents=True)
    (root / "runtime" / "training_config.yaml").write_text("training: fixture\n")
    (root / "runtime" / "vae" / "fixture.bin").write_bytes(b"vae")
    (root / "policy_contract.json").write_text(json.dumps({"schema_version": 3}))
    (root / "checkpoint.json").write_text(json.dumps({"runtime_manifest": "runtime_manifest.json"}))
    (root / "config.json").write_text("{}")
    (root / "export_manifest.json").write_text(
        json.dumps(
            {"framework_commit": FRAMEWORK_COMMIT, "base_checkpoint": BASE_CHECKPOINT}
        )
    )
    (root / "model.safetensors.index.json").write_text("{}")
    runtime = {
        "schema_version": 3,
        "framework_commit": FRAMEWORK_COMMIT,
        "base_checkpoint": BASE_CHECKPOINT,
        "training_config": {"path": "runtime/training_config.yaml", "sha256": _hash(root / "runtime" / "training_config.yaml")},
        "policy_contract": {"path": "policy_contract.json", "sha256": _hash(root / "policy_contract.json")},
        "vae": {"path": "runtime/vae/fixture.bin", "sha256": _hash(root / "runtime" / "vae" / "fixture.bin")},
    }
    (root / "runtime_manifest.json").write_text(json.dumps(runtime))


def _published_export(
    tmp_path: Path,
    source_value: str = "model:owner/policy",
    *,
    revision: str = COMMIT,
    resolved_revision: str | None = None,
):
    store = ModelStore(tmp_path / "persistent")
    source = parse_model_source(source_value)
    if source.kind == "model":
        resolved_revision = revision if resolved_revision is None else resolved_revision
    else:
        revision = resolved_revision = None
    staging = tmp_path / "staging"
    _export(staging)
    (staging / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    target = store._target(source, revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = store._publish(staging, target, source, revision, resolved_revision)
    return store, source, record


@pytest.mark.parametrize("source_value", ["model:owner/policy", "bucket:owner/policy:models/run-1"])
def test_get_reads_only_completion_and_contract_offline(tmp_path, monkeypatch, source_value):
    store, source, published = _published_export(tmp_path, source_value)
    target = Path(published.path)
    allowed = {target / ".complete.json", target / "policy_contract.json"}
    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path in allowed, f"direct readiness lookup must not read {path}"
        opened.append(path)
        return original_open(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("direct readiness lookup must not traverse directories or access the network")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded_open)
        guard.setattr(Path, "glob", forbidden)
        guard.setattr(Path, "rglob", forbidden)
        guard.setattr(Path, "iterdir", forbidden)
        guard.setattr(model_store.os, "scandir", forbidden)
        guard.setattr(model_store, "HuggingFaceClient", forbidden)
        guard.setattr(model_store.subprocess, "run", forbidden)
        assert store.get(source, published.requested_revision) == published

    assert set(opened) == allowed


@pytest.mark.parametrize(
    ("source", "revision"),
    [
        ("model:owner/policy", COMMIT),
        (None, COMMIT),
        (parse_model_source("owner/policy"), None),
        (parse_model_source("owner/policy"), "main"),
        (parse_model_source("owner/policy"), "A" * 40),
        (parse_model_source("bucket:owner/policy:models/run-1"), COMMIT),
    ],
)
def test_get_rejects_invalid_arguments(tmp_path, source, revision):
    with pytest.raises(ValueError):
        ModelStore(tmp_path).get(source, revision)


@pytest.mark.parametrize("source_value", ["model:owner/policy", "bucket:owner/policy:models/run-1"])
def test_get_returns_none_for_absent_export(tmp_path, source_value):
    source = parse_model_source(source_value)
    revision = COMMIT if source.kind == "model" else None
    assert ModelStore(tmp_path).get(source, revision) is None


@pytest.mark.parametrize("completion", [None, "{", "[]", "null", '"not an object"', "{}"])
def test_get_rejects_missing_or_malformed_completion(tmp_path, completion):
    store, source, published = _published_export(tmp_path)
    marker = Path(published.path) / ".complete.json"
    if completion is None:
        marker.unlink()
    else:
        marker.write_text(completion)
    assert store.get(source, COMMIT) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_source", "model:other/policy"),
        ("model_source", "owner/policy"),
        ("model_source", "bucket:owner/policy:models/run-1"),
        ("requested_revision", "b" * 40),
        ("requested_revision", None),
        ("resolved_revision", "b" * 40),
        ("resolved_revision", None),
        ("validation_state", "failed"),
        ("validation_state", None),
    ],
)
def test_get_requires_exact_model_identity_and_validated_state(tmp_path, field, value):
    store, source, published = _published_export(tmp_path)
    marker = Path(published.path) / ".complete.json"
    completion = json.loads(marker.read_text())
    completion[field] = value
    marker.write_text(json.dumps(completion))
    assert store.get(source, COMMIT) is None


def test_get_does_not_accept_a_colliding_repository_directory(tmp_path):
    store, source, published = _published_export(tmp_path, "model:owner--extra/policy")
    collision = parse_model_source("model:owner/extra--policy")
    assert store._target(source, COMMIT) == store._target(collision, COMMIT)
    assert store.get(source, COMMIT) == published
    assert store.get(collision, COMMIT) is None
    assert store.get(source, "b" * 40) is None


@pytest.mark.parametrize("field", ["requested_revision", "resolved_revision"])
@pytest.mark.parametrize("mutation", ["missing", "revision"])
def test_get_requires_explicit_null_bucket_revisions(tmp_path, field, mutation):
    store, source, published = _published_export(tmp_path, "bucket:owner/policy:models/run-1")
    marker = Path(published.path) / ".complete.json"
    completion = json.loads(marker.read_text())
    if mutation == "missing":
        del completion[field]
    else:
        completion[field] = COMMIT
    marker.write_text(json.dumps(completion))
    assert store.get(source) is None


@pytest.mark.parametrize("other_source", ["bucket:owner/policy:models/run-2", "bucket:other/policy:models/run-1"])
def test_get_requires_exact_bucket_source(tmp_path, other_source):
    store, source, published = _published_export(tmp_path, "bucket:owner/policy:models/run-1")
    marker = Path(published.path) / ".complete.json"
    completion = json.loads(marker.read_text())
    completion["model_source"] = other_source
    marker.write_text(json.dumps(completion))
    assert store.get(source) is None


@pytest.mark.parametrize("hashes", [None, [], "invalid", {}])
def test_get_rejects_invalid_completion_hash_mapping(tmp_path, hashes):
    store, source, published = _published_export(tmp_path)
    marker = Path(published.path) / ".complete.json"
    completion = json.loads(marker.read_text())
    completion["hashes"] = hashes
    marker.write_text(json.dumps(completion))
    assert store.get(source, COMMIT) is None


@pytest.mark.parametrize(
    "name",
    ["checkpoint.json", "config.json", "export_manifest.json", "runtime_manifest.json", "policy_contract.json", "model.safetensors.index.json"],
)
def test_get_requires_all_export_hash_entries(tmp_path, name):
    store, source, published = _published_export(tmp_path)
    marker = Path(published.path) / ".complete.json"
    completion = json.loads(marker.read_text())
    del completion["hashes"][name]
    marker.write_text(json.dumps(completion))
    assert store.get(source, COMMIT) is None


@pytest.mark.parametrize("name", ["policy_contract.json", "config.json", "model-00001-of-00001.safetensors"])
@pytest.mark.parametrize("digest", [None, 7, [], "", "a" * 63, "A" * 64, "g" * 64])
def test_get_requires_lowercase_sha256_digests(tmp_path, name, digest):
    store, source, published = _published_export(tmp_path)
    marker = Path(published.path) / ".complete.json"
    completion = json.loads(marker.read_text())
    completion["hashes"][name] = digest
    marker.write_text(json.dumps(completion))
    assert store.get(source, COMMIT) is None


@pytest.mark.parametrize("field", ["model_source", "requested_revision", "resolved_revision", "validation_state", "hashes"])
def test_get_rejects_missing_model_completion_fields(tmp_path, field):
    store, source, published = _published_export(tmp_path)
    marker = Path(published.path) / ".complete.json"
    completion = json.loads(marker.read_text())
    del completion[field]
    marker.write_text(json.dumps(completion))
    assert store.get(source, COMMIT) is None




@pytest.mark.parametrize("source_value", ["model:owner/policy", "bucket:owner/policy:models/run-1"])
@pytest.mark.parametrize("mutation", ["missing", "tampered", "wrong_hash"])
def test_get_rechecks_contract_after_successful_lookup(tmp_path, source_value, mutation):
    store, source, published = _published_export(tmp_path, source_value)
    assert store.get(source, published.requested_revision) == published
    contract = Path(published.path) / "policy_contract.json"
    if mutation == "missing":
        contract.unlink()
    elif mutation == "tampered":
        contract.write_text(json.dumps({"schema_version": 3, "changed": True}))
    else:
        marker = Path(published.path) / ".complete.json"
        completion = json.loads(marker.read_text())
        completion["hashes"]["policy_contract.json"] = "0" * 64
        marker.write_text(json.dumps(completion))
    assert store.get(source, published.requested_revision) is None


@pytest.mark.parametrize("source_value", ["model:owner/policy", "bucket:owner/policy:models/run-1"])
@pytest.mark.parametrize("payload", ["model-00001-of-00001.safetensors", "runtime/vae/fixture.bin", "checkpoint.json"])
@pytest.mark.parametrize("mutation", ["tampered", "missing"])
def test_list_rechecks_payloads_but_get_trusts_receipt(tmp_path, source_value, payload, mutation):
    store, source, published = _published_export(tmp_path, source_value)
    revision = published.requested_revision
    assert store.list() == [published]
    path = Path(published.path) / payload
    if mutation == "missing":
        path.unlink()
    else:
        path.write_bytes(b"tampered")
    assert store.get(source, revision) == published
    assert store.list() == []


@pytest.mark.parametrize("mutation", ["missing_hash", "extra_hash", "wrong_hash", "extra_file"])
def test_list_requires_exact_complete_hash_mapping(tmp_path, mutation):
    store, source, published = _published_export(tmp_path)
    target = Path(published.path)
    marker = target / ".complete.json"
    completion = json.loads(marker.read_text())
    payload = "model-00001-of-00001.safetensors"
    if mutation == "missing_hash":
        del completion["hashes"][payload]
    elif mutation == "extra_hash":
        completion["hashes"]["missing.bin"] = "0" * 64
    elif mutation == "wrong_hash":
        completion["hashes"][payload] = "0" * 64
    else:
        (target / "unexpected.bin").write_bytes(b"unexpected")
    marker.write_text(json.dumps(completion))
    assert store.get(source, COMMIT) == published
    assert store.list() == []


def test_list_validates_export_structure_even_when_receipt_hashes_match(tmp_path):
    store, source, published = _published_export(tmp_path)
    target = Path(published.path)
    runtime_path = target / "runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["schema_version"] = 1
    runtime_path.write_text(json.dumps(runtime))
    marker = target / ".complete.json"
    completion = json.loads(marker.read_text())
    completion["hashes"][runtime_path.name] = _hash(runtime_path)
    marker.write_text(json.dumps(completion))
    assert store.get(source, COMMIT) == published
    assert store.list() == []


@pytest.mark.parametrize("mutation", ["missing", "invalid", "conflicting"])
def test_list_rejects_invalid_base_checkpoint_provenance(tmp_path, mutation):
    store, source, published = _published_export(tmp_path)
    target = Path(published.path)
    runtime_path = target / "runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text())
    if mutation == "missing":
        del runtime["base_checkpoint"]
    elif mutation == "invalid":
        runtime["base_checkpoint"]["commit_sha"] = "main"
    else:
        runtime["base_checkpoint"]["commit_sha"] = "c" * 40
    runtime_path.write_text(json.dumps(runtime))
    marker = target / ".complete.json"
    completion = json.loads(marker.read_text())
    completion["hashes"][runtime_path.name] = _hash(runtime_path)
    marker.write_text(json.dumps(completion))

    assert store.get(source, COMMIT) == published
    assert store.list() == []


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


@pytest.mark.parametrize(("revision", "uses_refs"), [(COMMIT, False), ("release-1", True)])
def test_hugging_face_client_resolves_tags_to_commits(monkeypatch, revision, uses_refs):
    client = model_store.HuggingFaceClient("owner/policy", "secret")
    requests = []
    listed = []

    def request(url):
        requests.append(url)
        return json.dumps({"tags": [{"name": "release-1", "targetCommit": COMMIT}]}).encode()

    def list_files(commit):
        listed.append(commit)
        return ["checkpoint.json"]

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(client, "_list_files", list_files)
    assert client.resolve(revision) == (COMMIT, ["checkpoint.json"])
    assert any(url.endswith("/refs") for url in requests) is uses_refs
    assert listed == [COMMIT]


def test_hugging_face_client_rejects_missing_tag(monkeypatch):
    client = model_store.HuggingFaceClient("owner/policy")
    monkeypatch.setattr(
        client,
        "_request",
        lambda url: json.dumps({"tags": [{"name": "other", "targetCommit": COMMIT}]}).encode(),
    )
    with pytest.raises(ValueError, match="existing immutable model tag"):
        client.resolve("release-1")


def test_hugging_face_client_lists_exact_commit_files_with_pagination(monkeypatch):
    client = model_store.HuggingFaceClient("owner/policy")
    pages = iter([
        ([{"type": "directory", "path": "runtime"}, {"type": "file", "path": "checkpoint.json"}], '<https://huggingface.co/next>; rel="next"'),
        ([{"type": "file", "path": "runtime/vae/model.bin"}], ""),
    ])

    class Response:
        def __init__(self, value, link):
            self.value = value
            self.headers = {"Link": link}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(self.value).encode()

    monkeypatch.setattr(client, "_open", lambda url: Response(*next(pages)))
    assert client._list_files(COMMIT) == ["checkpoint.json", "runtime/vae/model.bin"]


def test_tagged_model_receipt_records_resolved_commit(tmp_path):
    store, source, published = _published_export(
        tmp_path, revision="release-1", resolved_revision=COMMIT
    )
    assert published.requested_revision == "release-1"
    assert published.resolved_revision == COMMIT
    assert store.get(source, "release-1") == published
    assert store.list() == [published]


@pytest.mark.parametrize("revision", [COMMIT, "release-1"])
def test_model_download_is_atomic_idempotent_and_list_is_offline(tmp_path, monkeypatch, revision):
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
    model = parse_model_source("owner/policy")
    first = store.download(model, revision, "secret")
    assert first.response()["model_source"] == "model:owner/policy"
    assert first.requested_revision == revision
    assert first.resolved_revision == COMMIT
    assert Path(first.path, ".complete.json").is_file()
    assert store.download(model, revision, "secret") == first
    assert calls.count(("resolve", revision)) == 1
    assert [item.response() for item in store.list()] == [first.response()]
    assert calls.count(("resolve", revision)) == 1


def test_model_download_rejects_invalid_exports(tmp_path, monkeypatch):
    source = tmp_path / "published"
    source.mkdir()
    _export(source)
    (source / "checkpoint.json").unlink()

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
    with pytest.raises(ValueError, match="missing required files"):
        store.download(parse_model_source("model:owner/policy"), COMMIT)
    assert store.list() == []


@pytest.mark.parametrize(
    ("token", "expected_token", "expected_legacy"),
    [("configured-secret", "configured-secret", None), (None, "inherited-secret", "legacy-secret")],
)
def test_bucket_preflight_uses_sync_credentials_then_publishes_and_caches_offline(
    tmp_path, monkeypatch, token, expected_token, expected_legacy
):
    source = tmp_path / "bucket-export"
    source.mkdir()
    _export(source)
    calls = []
    monkeypatch.setenv("HF_TOKEN", "inherited-secret")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "legacy-secret")

    def fake_run(argv, **kwargs):
        calls.append((argv[3], kwargs["env"].copy()))
        if argv[3] == "list":
            return model_store.subprocess.CompletedProcess(argv, 0, '[{"type":"file","path":"checkpoint.json"}]', "")
        shutil.copytree(source, argv[-1], dirs_exist_ok=True)
        return model_store.subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(model_store.subprocess, "run", fake_run)
    store = ModelStore(tmp_path / "persistent")
    bucket = parse_model_source("bucket:owner/policy:models/run-1")
    first = store.download(bucket, None, token)

    assert first.response()["model_source"] == bucket.canonical
    assert first.requested_revision is first.resolved_revision is None
    assert [command for command, _ in calls] == ["list", "sync"]
    assert [environment.get("HF_TOKEN") for _, environment in calls] == [expected_token, expected_token]
    assert [environment.get("HUGGINGFACE_HUB_TOKEN") for _, environment in calls] == [expected_legacy, expected_legacy]

    def offline_run(*args, **kwargs):
        raise AssertionError("a valid cached bucket model must not access the network")

    monkeypatch.setattr(model_store.subprocess, "run", offline_run)
    assert store.download(bucket, None, token) == first
    assert [item.response() for item in store.list()] == [first.response()]


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "error"),
    [
        (1, "", "HTTP 401 Unauthorized", "access denied"),
        (1, "", "HTTP 404 Not Found", "missing or inaccessible"),
        (0, "[]", "", "missing or empty"),
        (0, "(empty)", "", "missing or empty"),
        (1, "", "network connection timed out", "cannot contact"),
    ],
)
def test_bucket_preflight_rejects_inaccessible_prefix_before_transfer(
    tmp_path, monkeypatch, returncode, stdout, stderr, error
):
    transfers = []

    def fake_run(argv, **kwargs):
        if argv[3] == "list":
            return model_store.subprocess.CompletedProcess(argv, returncode, stdout, stderr)
        transfers.append(argv)
        raise AssertionError("bucket payload transfer must not start after preflight failure")

    monkeypatch.setattr(model_store.subprocess, "run", fake_run)
    store = ModelStore(tmp_path / "persistent")
    bucket = parse_model_source("bucket:owner/policy:models/run-1")

    with pytest.raises(ValueError, match=error):
        store.download(bucket, None, "bucket-secret")
    assert transfers == []
    assert store.list() == []


def test_bucket_invalid_download_is_not_discoverable_and_revisions_are_rejected(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv[3])
        if argv[3] == "list":
            return model_store.subprocess.CompletedProcess(argv, 0, '[{"type":"file","path":"checkpoint.json"}]', "")
        Path(argv[-1], "checkpoint.json").write_text("{}")
        return model_store.subprocess.CompletedProcess(argv, 0)

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
    assert calls == ["list", "sync"]


def test_model_configuration_validation_is_strict(tmp_path):
    assert validate_repository("owner/policy") == "owner/policy"
    assert validate_revision(COMMIT) == COMMIT
    assert validate_revision("release-1") == "release-1"
    assert validate_data_root(str(tmp_path)) == tmp_path
    with pytest.raises(ValueError):
        validate_repository("owner/policy/extra")
    with pytest.raises(ValueError):
        validate_revision("main")
    with pytest.raises(ValueError):
        validate_data_root("relative")

def test_supervisor_start_status_stop_is_idempotent(tmp_path):
    port = 18191
    program = """
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

assert os.environ.get("HF_TOKEN") == "server-secret"

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
