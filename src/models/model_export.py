"""Validation for portable policy exports deployed by viam-server."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

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


def validate_model_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_path must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("model_path must be an absolute path")
    return path


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


def _safe_relative(name: object) -> Path:
    if not isinstance(name, str) or not name or name.startswith("/"):
        raise ValueError("runtime manifest path is invalid")
    path = Path(name)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"runtime manifest path escapes the export: {name!r}")
    return path


def validate_runtime_manifest(root: Path) -> dict[str, str]:
    """Validate the bundle shape and digests of the runtime manifest entries.

    Model weights and the VAE are not hashed, so this is cheap enough for every
    server start.
    """
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
    hashes: dict[str, str] = {}
    for field in ("training_config", "policy_contract", "vae"):
        entry = runtime_manifest.get(field)
        if not isinstance(entry, dict):
            raise ValueError(f"portable export runtime manifest has no {field} entry")
        relative = _safe_relative(entry.get("path"))
        actual = root / relative
        name = relative.as_posix()
        if not actual.is_file():
            raise ValueError(f"portable export runtime file is missing for {field}")
        if field == "vae":
            # The VAE is hundreds of MB; like the weight shards, its payload is
            # left to the policy server's loader.
            continue
        if name not in hashes:
            hashes[name] = _sha256(actual)
        if entry.get("sha256") != hashes[name]:
            raise ValueError(f"portable export runtime checksum failed for {field}")
    return hashes
