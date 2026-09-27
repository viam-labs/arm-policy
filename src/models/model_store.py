"""Atomic, offline-readable storage for immutable portable policy exports."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_EXPORT_FILES = frozenset(
    {
        "checkpoint.json",
        "config.json",
        "export_manifest.json",
        "runtime_manifest.json",
        "policy_contract.json",
        "model.safetensors.index.json",
    }
)

REMOTE_COMPLETION_MARKER = ".cosmos-policy-complete.json"
REMOTE_MARKER_SCHEMA_VERSION = 1


def validate_repository(value: object) -> str:
    if not isinstance(value, str) or not _REPOSITORY.fullmatch(value):
        raise ValueError("model_repository must be an HF repository in the form owner/name")
    return value


def validate_revision(value: object) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError("model_revision must be an immutable 40-character lowercase git commit")
    return value


def validate_data_root(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_data_root must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("model_data_root must be an absolute path")
    return path

def default_data_root() -> Path:
    """Return the persistent module-data convention when configuration omits a root."""
    for variable in ("VIAM_MODULE_DATA", "XDG_DATA_HOME"):
        value = os.environ.get(variable)
        if value:
            path = Path(value).expanduser()
            if path.is_absolute():
                return path / "arm-policy"
    return Path.home() / ".local" / "share" / "viam" / "arm-policy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid required JSON file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid required JSON file {path.name}: expected an object")
    return value

def validate_remote_completion_marker(
    marker: Mapping[str, Any], repository: str, resolved_revision: str
) -> None:
    """Reject unpublished, redirected, or provenance-free Hub revisions."""
    if not isinstance(marker, Mapping):
        raise ValueError("remote portable export completion marker must be an object")
    if marker.get("schema_version") != REMOTE_MARKER_SCHEMA_VERSION:
        raise ValueError("remote portable export completion marker has an unsupported schema")
    if marker.get("repository") != repository:
        raise ValueError("remote portable export completion marker repository does not match")
    if marker.get("resolved_revision") != resolved_revision:
        raise ValueError("remote portable export completion marker revision does not match")
    provenance = marker.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("remote portable export completion marker has no provenance")
    source_repository = provenance.get("repository")
    source_revision = provenance.get("revision")
    if not isinstance(source_repository, str) or not source_repository:
        raise ValueError("remote portable export completion marker provenance repository is invalid")
    validate_revision(source_revision)


def _safe_relative(name: object) -> Path:
    if not isinstance(name, str) or not name or name.startswith("/"):
        raise ValueError("Hub file name is invalid")
    path = Path(name)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Hub file name escapes the export: {name!r}")
    return path


def _export_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".complete.json"
    }


def validate_portable_export(root: Path) -> dict[str, str]:
    """Validate the portable bundle shape and its runtime artifact digests."""
    missing = sorted(name for name in _REQUIRED_EXPORT_FILES if not (root / name).is_file())
    if missing:
        raise ValueError(f"portable export is missing required files: {', '.join(missing)}")
    checkpoint = _read_json(root / "checkpoint.json")
    export_manifest = _read_json(root / "export_manifest.json")
    runtime_manifest = _read_json(root / "runtime_manifest.json")
    contract = _read_json(root / "policy_contract.json")
    if runtime_manifest.get("schema_version") != 2:
        raise ValueError("portable export has an unsupported runtime manifest schema")
    if runtime_manifest.get("framework_commit") != export_manifest.get("framework_commit"):
        raise ValueError("portable export runtime and export framework revisions conflict")
    if checkpoint.get("runtime_manifest") != "runtime_manifest.json":
        raise ValueError("portable export checkpoint does not reference runtime_manifest.json")
    if contract.get("schema_version") != 2:
        raise ValueError("portable export policy contract has an unsupported schema")
    for field in ("training_config", "policy_contract", "vae"):
        entry = runtime_manifest.get(field)
        if not isinstance(entry, dict):
            raise ValueError(f"portable export runtime manifest has no {field} entry")
        relative = _safe_relative(entry.get("path"))
        actual = root / relative
        if not actual.is_file() or entry.get("sha256") != _sha256(actual):
            raise ValueError(f"portable export runtime checksum failed for {field}")
    return _export_hashes(root)


@dataclass(frozen=True)
class ModelRecord:
    repository: str
    requested_revision: str
    resolved_revision: str
    path: str
    validation_state: str

    def response(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "path": self.path,
            "validation_state": self.validation_state,
        }


class HuggingFaceClient:
    """Small stdlib client so model management does not import heavyweight runtimes."""

    def __init__(self, repository: str, token: str | None = None) -> None:
        self.repository = validate_repository(repository)
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    def _open(self, url: str):
        headers = {"User-Agent": "arm-policy-model-store/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            auth = " (check HF_TOKEN/HUGGINGFACE_HUB_TOKEN or configured hf_token)" if exc.code in (401, 403) else ""
            raise ValueError(f"Hugging Face request failed ({exc.code}){auth}: {url}") from exc
        except OSError as exc:
            raise ValueError(f"cannot contact Hugging Face: {exc}") from exc

    def _request(self, url: str) -> bytes:
        with self._open(url) as response:
            return response.read()

    def resolve(self, requested_revision: str) -> tuple[str, list[str]]:
        requested_revision = validate_revision(requested_revision)
        encoded_repo = urllib.parse.quote(self.repository, safe="/")
        encoded_revision = urllib.parse.quote(requested_revision, safe="")
        value = json.loads(self._request(f"https://huggingface.co/api/models/{encoded_repo}/revision/{encoded_revision}"))
        resolved = value.get("sha") if isinstance(value, dict) else None
        if not isinstance(resolved, str) or not _REVISION.fullmatch(resolved):
            raise ValueError("Hugging Face did not resolve the requested revision to a commit")
        if resolved != requested_revision:
            raise ValueError("Hugging Face resolved a different commit than the immutable requested revision")
        siblings = value.get("siblings")
        if not isinstance(siblings, list):
            raise ValueError("Hugging Face response did not list repository files")
        files = [_safe_relative(item.get("rfilename")).as_posix() for item in siblings if isinstance(item, dict)]
        if not files:
            raise ValueError("Hugging Face repository revision contains no files")
        return resolved, files

    def download(self, revision: str, name: str, destination: Path) -> None:
        encoded_repo = urllib.parse.quote(self.repository, safe="/")
        encoded_revision = urllib.parse.quote(revision, safe="")
        encoded_name = urllib.parse.quote(name, safe="/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._open(
                f"https://huggingface.co/{encoded_repo}/resolve/{encoded_revision}/{encoded_name}"
            ) as response, destination.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise


class ModelStore:
    """A lock-serialized immutable store; only completion manifests are discoverable."""

    def __init__(self, data_root: Path) -> None:
        self.root = data_root / "arm-policy-models"

    def _target(self, repository: str, revision: str) -> Path:
        return self.root / repository.replace("/", "--") / revision

    @staticmethod
    def _completion(path: Path) -> Path:
        return path / ".complete.json"

    def _record_if_valid(self, target: Path) -> ModelRecord | None:
        marker = self._completion(target)
        if not marker.is_file():
            return None
        try:
            manifest = _read_json(marker)
            repository = validate_repository(manifest.get("repository"))
            requested_revision = validate_revision(manifest.get("requested_revision"))
            resolved_revision = validate_revision(manifest.get("resolved_revision"))
            validate_remote_completion_marker(
                manifest.get("remote_completion_marker"), repository, resolved_revision
            )
            hashes = validate_portable_export(target)
        except ValueError:
            return None
        if manifest.get("hashes") != hashes or manifest.get("validation_state") != "validated":
            return None
        return ModelRecord(
            repository=repository,
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
            path=str(target),
            validation_state="validated",
        )

    def download(self, repository: str, requested_revision: str, token: str | None = None) -> ModelRecord:
        repository = validate_repository(repository)
        requested_revision = validate_revision(requested_revision)
        target = self._target(repository, requested_revision)
        lock_path = target.parent / f".{target.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = self._record_if_valid(target)
            if cached is not None:
                return cached
            client = HuggingFaceClient(repository, token)
            resolved, files = client.resolve(requested_revision)
            if REMOTE_COMPLETION_MARKER not in files:
                raise ValueError(
                    f"Hugging Face revision is not published: missing {REMOTE_COMPLETION_MARKER}"
                )
            staging = Path(tempfile.mkdtemp(prefix=f".{resolved}.", dir=target.parent))
            try:
                remote_marker_path = staging / REMOTE_COMPLETION_MARKER
                client.download(resolved, REMOTE_COMPLETION_MARKER, remote_marker_path)
                remote_marker = _read_json(remote_marker_path)
                validate_remote_completion_marker(remote_marker, repository, resolved)
                for name in files:
                    if name != REMOTE_COMPLETION_MARKER:
                        client.download(resolved, name, staging / _safe_relative(name))
                hashes = validate_portable_export(staging)
                marker = {
                    "repository": repository,
                    "requested_revision": requested_revision,
                    "resolved_revision": resolved,
                    "validation_state": "validated",
                    "remote_completion_marker": remote_marker,
                    "hashes": hashes,
                }
                (staging / ".complete.json").write_text(
                    json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                if target.exists():
                    shutil.rmtree(target)
                os.replace(staging, target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            return ModelRecord(repository, requested_revision, resolved, str(target), "validated")

    def list(self) -> list[ModelRecord]:
        if not self.root.is_dir():
            return []
        records: list[ModelRecord] = []
        for marker in sorted(self.root.glob("*/*/.complete.json")):
            record = self._record_if_valid(marker.parent)
            if record is not None:
                records.append(record)
        return records
