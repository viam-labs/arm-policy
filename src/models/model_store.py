"""Atomic, offline-readable storage for immutable portable policy exports."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

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


def _validate_bucket_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("bucket path must be a non-empty safe relative path")
    if "\\" in value:
        raise ValueError("bucket path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("bucket path must be a non-empty safe relative path")
    return value


@dataclass(frozen=True)
class ModelSource:
    """An immutable Hugging Face model repository or storage bucket location."""

    kind: Literal["model", "bucket"]
    repository: str
    bucket_path: str | None = None

    def __post_init__(self) -> None:
        validate_repository(self.repository)
        if self.kind == "model":
            if self.bucket_path is not None:
                raise ValueError("model sources cannot include a bucket path")
        elif self.kind == "bucket":
            _validate_bucket_path(self.bucket_path)
        else:
            raise ValueError("model source kind must be model or bucket")

    @property
    def canonical(self) -> str:
        if self.kind == "model":
            return f"model:{self.repository}"
        return f"bucket:{self.repository}:{self.bucket_path}"

    def __str__(self) -> str:
        return self.canonical


def parse_model_source(value: object) -> ModelSource:
    """Parse the strict configuration syntax into a canonical source value."""
    if not isinstance(value, str):
        raise ValueError("model_source must be a string")
    if value.startswith("model:"):
        return ModelSource("model", validate_repository(value.removeprefix("model:")))
    if value.startswith("bucket:"):
        repository, separator, bucket_path = value.removeprefix("bucket:").partition(":")
        if not separator:
            raise ValueError("bucket model_source must be bucket:owner/name:path")
        return ModelSource("bucket", validate_repository(repository), _validate_bucket_path(bucket_path))
    if ":" not in value:
        return ModelSource("model", validate_repository(value))
    raise ValueError("model_source must be owner/name, model:owner/name, or bucket:owner/name:path")


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
    if contract.get("schema_version") != 3:
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
    source: ModelSource
    requested_revision: str | None
    resolved_revision: str | None
    path: str
    validation_state: Literal["validated"]

    def response(self) -> dict[str, str | None]:
        return {
            "model_source": self.source.canonical,
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
    """A lock-serialized immutable store; only validated completions are discoverable."""

    def __init__(self, data_root: Path) -> None:
        self.root = data_root / "arm-policy-models"

    def _target(self, source: ModelSource, requested_revision: str | None) -> Path:
        repository = source.repository.replace("/", "--")
        if source.kind == "model":
            assert requested_revision is not None
            return self.root / "model" / repository / requested_revision
        source_id = hashlib.sha256(source.canonical.encode("utf-8")).hexdigest()
        return self.root / "bucket" / repository / source_id

    @staticmethod
    def _completion(path: Path) -> Path:
        return path / ".complete.json"

    def get(self, source: ModelSource, requested_revision: str | None = None) -> ModelRecord | None:
        """Find a published source and verify its contract, not its model payload.

        Readiness needs only the completion metadata and safety contract. Full
        bundle integrity remains the responsibility of list/download validation.
        """
        if not isinstance(source, ModelSource):
            raise ValueError("source must be a ModelSource")
        if source.kind == "model":
            requested_revision = validate_revision(requested_revision)
        elif requested_revision is not None:
            raise ValueError("bucket sources cannot specify model_revision")
        target = self._target(source, requested_revision)
        try:
            manifest = _read_json(self._completion(target))
            if (
                manifest.get("model_source") != source.canonical
                or "requested_revision" not in manifest
                or "resolved_revision" not in manifest
                or manifest["requested_revision"] != requested_revision
                or manifest["resolved_revision"] != requested_revision
                or manifest.get("validation_state") != "validated"
            ):
                return None
            if source.kind == "model":
                validate_remote_completion_marker(
                    manifest.get("remote_completion_marker"), source.repository, requested_revision
                )
            hashes = manifest.get("hashes")
            if (
                not isinstance(hashes, dict)
                or not _REQUIRED_EXPORT_FILES.issubset(hashes)
                or any(
                    not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    for digest in hashes.values()
                )
            ):
                return None
            if hashes["policy_contract.json"] != _sha256(target / "policy_contract.json"):
                return None
        except (OSError, ValueError):
            return None
        return ModelRecord(source, requested_revision, requested_revision, str(target), "validated")

    def _record_if_valid(self, target: Path) -> ModelRecord | None:
        marker = self._completion(target)
        if not marker.is_file():
            return None
        try:
            manifest = _read_json(marker)
            source = parse_model_source(manifest.get("model_source"))
            requested_revision = manifest.get("requested_revision")
            resolved_revision = manifest.get("resolved_revision")
            if source.kind == "model":
                requested_revision = validate_revision(requested_revision)
                resolved_revision = validate_revision(resolved_revision)
                validate_remote_completion_marker(
                    manifest.get("remote_completion_marker"), source.repository, resolved_revision
                )
            elif requested_revision is not None or resolved_revision is not None:
                raise ValueError("bucket completion manifests cannot have model revisions")
            hashes = validate_portable_export(target)
        except ValueError:
            return None
        if manifest.get("hashes") != hashes or manifest.get("validation_state") != "validated":
            return None
        return ModelRecord(source, requested_revision, resolved_revision, str(target), "validated")

    @staticmethod
    def _bucket_environment(token: str | None) -> dict[str, str]:
        """Build the one credential environment shared by bucket operations."""
        environment = os.environ.copy()
        if token is not None:
            # HF_TOKEN is the documented CLI credential.  Do not let the
            # legacy inherited variable select a different credential.
            environment["HF_TOKEN"] = token
            environment.pop("HUGGINGFACE_HUB_TOKEN", None)
        return environment

    @staticmethod
    def _bucket_preflight(source: ModelSource, environment: Mapping[str, str]) -> None:
        """Confirm the configured prefix is listable before syncing its payload."""
        assert source.kind == "bucket" and source.bucket_path is not None
        uri = f"hf://buckets/{source.repository}/{source.bucket_path}"
        try:
            result = subprocess.run(
                ["uvx", "hf@latest", "buckets", "list", uri, "--format", "json"],
                check=False,
                env=environment,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise ValueError(
                f"cannot start Hugging Face bucket preflight for {source.canonical}: {exc}"
            ) from exc
        if result.returncode:
            detail = f"{result.stdout}\n{result.stderr}".casefold()
            credentials = "check configured hf_token or HF_TOKEN/HUGGINGFACE_HUB_TOKEN"
            if any(marker in detail for marker in ("401", "403", "unauthorized", "forbidden", "access denied", "permission denied")):
                raise ValueError(
                    f"Hugging Face bucket preflight access denied for {source.canonical}; {credentials}"
                )
            if any(marker in detail for marker in ("timeout", "timed out", "connection", "network", "dns", "resolve", "proxy", "ssl")):
                raise ValueError(
                    f"cannot contact Hugging Face while checking bucket prefix {source.canonical}; check network connectivity"
                )
            raise ValueError(
                f"Hugging Face bucket prefix is missing or inaccessible: {source.canonical}; check model_source and access"
            )
        listing = result.stdout.strip()
        if listing == "(empty)":
            entries: object = []
        else:
            try:
                entries = json.loads(listing)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Hugging Face bucket preflight returned invalid JSON for {source.canonical}"
                ) from exc
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ValueError(f"Hugging Face bucket preflight returned an invalid listing for {source.canonical}")
        if not entries:
            raise ValueError(
                f"Hugging Face bucket prefix is missing or empty: {source.canonical}; check model_source"
            )

    @staticmethod
    def _bucket_sync(source: ModelSource, destination: Path, environment: Mapping[str, str]) -> None:
        assert source.kind == "bucket" and source.bucket_path is not None
        subprocess.run(
            ["uvx", "hf@latest", "buckets", "sync", f"hf://buckets/{source.repository}/{source.bucket_path}", str(destination)],
            check=True,
            env=environment,
        )

    def _publish(
        self,
        staging: Path,
        target: Path,
        source: ModelSource,
        requested_revision: str | None,
        resolved_revision: str | None,
        remote_marker: Mapping[str, Any] | None = None,
    ) -> ModelRecord:
        hashes = validate_portable_export(staging)
        marker: dict[str, Any] = {
            "model_source": source.canonical,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "validation_state": "validated",
            "hashes": hashes,
        }
        if remote_marker is not None:
            marker["remote_completion_marker"] = remote_marker
        self._completion(staging).write_text(
            json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
        return ModelRecord(source, requested_revision, resolved_revision, str(target), "validated")

    def download(
        self, source: ModelSource, requested_revision: str | None, token: str | None = None
    ) -> ModelRecord:
        if not isinstance(source, ModelSource):
            raise ValueError("source must be a ModelSource")
        if source.kind == "model":
            requested_revision = validate_revision(requested_revision)
        elif requested_revision is not None:
            raise ValueError("bucket sources cannot specify model_revision")
        target = self._target(source, requested_revision)
        lock_path = target.parent / f".{target.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = self._record_if_valid(target)
            if cached is not None:
                return cached
            if source.kind == "bucket":
                environment = self._bucket_environment(token)
                self._bucket_preflight(source, environment)
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
            try:
                if source.kind == "model":
                    client = HuggingFaceClient(source.repository, token)
                    resolved, files = client.resolve(requested_revision)
                    if REMOTE_COMPLETION_MARKER not in files:
                        raise ValueError(f"Hugging Face revision is not published: missing {REMOTE_COMPLETION_MARKER}")
                    remote_marker_path = staging / REMOTE_COMPLETION_MARKER
                    client.download(resolved, REMOTE_COMPLETION_MARKER, remote_marker_path)
                    remote_marker = _read_json(remote_marker_path)
                    validate_remote_completion_marker(remote_marker, source.repository, resolved)
                    for name in files:
                        if name != REMOTE_COMPLETION_MARKER:
                            client.download(resolved, name, staging / _safe_relative(name))
                    return self._publish(staging, target, source, requested_revision, resolved, remote_marker)
                self._bucket_sync(source, staging, environment)
                return self._publish(staging, target, source, None, None)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise

    def list(self) -> list[ModelRecord]:
        if not self.root.is_dir():
            return []
        records: list[ModelRecord] = []
        for marker in sorted(self.root.rglob(".complete.json")):
            record = self._record_if_valid(marker.parent)
            if record is not None:
                records.append(record)
        return records
