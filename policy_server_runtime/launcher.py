#!/usr/bin/env python3
"""Validate a portable export and exec the pinned Cosmos policy server."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from service_adapter import PortableRobolabPolicyService

FRAMEWORK_COMMIT = "efba2076c80648663a89a7fdbc92eb262fae39ac"


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid portable export metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"invalid portable export metadata {path}: expected object")
    return value


def _validate(model: Path) -> dict:
    required = ("checkpoint.json", "config.json", "export_manifest.json", "runtime_manifest.json", "policy_contract.json", "model.safetensors.index.json")
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise SystemExit(f"portable export missing required files: {', '.join(missing)}")
    checkpoint, manifest, runtime, contract = (_json(model / name) for name in ("checkpoint.json", "export_manifest.json", "runtime_manifest.json", "policy_contract.json"))
    if manifest.get("framework_commit") != FRAMEWORK_COMMIT or runtime.get("framework_commit") != FRAMEWORK_COMMIT:
        raise SystemExit(f"portable export requires Cosmos revision {FRAMEWORK_COMMIT}")
    action = contract.get("action_schema", {})
    framework = contract.get("framework", {})
    shape = action.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int) or shape[0] < 2:
        raise SystemExit("portable export action schema must have a positive joint-plus-gripper width")
    if checkpoint.get("policy", {}).get("action_chunk_size") != contract.get("action_chunk_size"):
        raise SystemExit("portable export checkpoint policy conflicts with contract")
    if framework.get("action_space") != "joint_pos" or framework.get("use_state") is not True:
        raise SystemExit("portable export framework contract is unsupported")
    return contract


def _model_view(model: Path) -> Path:
    config = _json(model / "config.json")
    runtime = _json(model / "runtime_manifest.json")
    vae_rel = runtime.get("vae", {}).get("path")
    if not isinstance(vae_rel, str) or not (model / vae_rel).is_file():
        raise SystemExit("portable export has no packaged VAE")
    digest = hashlib.sha256((model / "config.json").read_bytes() + (model / "runtime_manifest.json").read_bytes()).hexdigest()[:16]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arm-policy" / f"policy-model-{digest}"
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "config.json").is_file():
            return root
        staging = root.with_name(f".{root.name}.{os.getpid()}.tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        for child in model.iterdir():
            if child.name != "config.json":
                (staging / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
        try:
            config["model"]["config"]["tokenizer"]["vae_path"] = str((model / vae_rel).resolve())
        except (KeyError, TypeError) as exc:
            raise SystemExit("portable export config has no tokenizer VAE path") from exc
        (staging / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    return root

def _overlay_root() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "arm-policy" / f"policy-server-{FRAMEWORK_COMMIT}"


def _prepare_overlay() -> Path:
    """Copy and patch a cache overlay; never mutate uv's installed package."""
    try:
        import cosmos_framework
    except ImportError as exc:
        raise SystemExit("isolated Cosmos runtime is not provisioned; run through uv --frozen") from exc
    root = _overlay_root()
    marker = root / ".joint-width-patch"
    target = root / "cosmos_framework/scripts/action_policy_server_robolab.py"
    old = '\"observation/joint_position\"], \"observation/joint_position\", 7)'
    new = '\"observation/joint_position\"], \"observation/joint_position\", self.cfg.action_dim - 1)'
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.is_file() and marker.read_text(encoding="utf-8") == FRAMEWORK_COMMIT + "\n":
            if target.is_file() and new in target.read_text(encoding="utf-8"):
                return root
            shutil.rmtree(root, ignore_errors=True)
        staging = root.with_name(f".{root.name}.{os.getpid()}.tmp")
        shutil.rmtree(staging, ignore_errors=True)
        source_package = Path(cosmos_framework.__file__).resolve().parent
        shutil.copytree(source_package, staging / "cosmos_framework")
        try:
            subprocess.run(
                ["patch", "-p1", "--batch", "--forward", "-i", str(Path(__file__).with_name("cosmos-joint-width.patch"))],
                cwd=staging,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise SystemExit("pinned Cosmos source does not match the packaged joint-width patch") from exc
        patched = staging / "cosmos_framework/scripts/action_policy_server_robolab.py"
        text = patched.read_text(encoding="utf-8")
        if old in text or text.count(new) != 1:
            shutil.rmtree(staging, ignore_errors=True)
            raise SystemExit("packaged joint-width patch did not produce the expected pinned source")
        (staging / marker.name).write_text(FRAMEWORK_COMMIT + "\n", encoding="utf-8")
        shutil.rmtree(root, ignore_errors=True)
        os.replace(staging, root)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    contract = _validate(model)
    view = _model_view(model)
    command = PortableRobolabPolicyService.command(model_view=view, contract=contract, host=args.host, port=args.port)
    overlay = _overlay_root()
    if args.print_command:
        print(f"PYTHONPATH={shlex.quote(str(overlay))} {shlex.join(command)}")
        return 0
    overlay = _prepare_overlay()
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(overlay) + (os.pathsep + prior if prior else "")
    os.chdir(view)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
