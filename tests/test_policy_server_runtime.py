import json
import sys
from pathlib import Path

import pytest

from models.policy_server import runtime_command
from policy_server_runtime import service_adapter
from policy_server_runtime.service_adapter import (
    _audit_hub_tree_metadata,
    _effective_hub_cache,
    _parse_wrapper_args,
)


def _tree_path(hub: Path, repository: str, revision: str, *, kind: str = "models") -> Path:
    return hub / f"{kind}--{repository.replace('/', '--')}" / "trees" / f"{revision}.json"


def _poisoned_tree_bytes() -> bytes:
    return b'{"format_version":1,"token":"sensitive-token","files":{"config.json":{"xet_hash":"64-*"}}}\n'


def test_audit_quarantines_poisoned_trees_across_hub_repositories(tmp_path, monkeypatch):
    hub = tmp_path / "hf-home" / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    poisoned_model = _tree_path(hub, "nvidia/model", "first")
    poisoned_dataset = _tree_path(hub, "nvidia/dataset", "second", kind="datasets")
    healthy_space = _tree_path(hub, "nvidia/space", "third", kind="spaces")
    ignored = hub / "other" / "trees" / "not-a-repo.json"
    for path, payload in (
        (poisoned_model, _poisoned_tree_bytes()),
        (poisoned_dataset, _poisoned_tree_bytes()),
        (healthy_space, b'{"format_version":1,"files":{"x":{"xet_hash":"64-a"}}}'),
        (ignored, _poisoned_tree_bytes()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    blob = poisoned_model.parents[1] / "blobs" / "sentinel"
    blob.parent.mkdir()
    blob.write_text("keep", encoding="utf-8")
    warnings: list[str] = []

    quarantined = _audit_hub_tree_metadata(warnings.append)

    assert len(quarantined) == 2
    assert not poisoned_model.exists() and not poisoned_dataset.exists()
    assert {path.read_bytes() for path in quarantined} == {_poisoned_tree_bytes()}
    assert healthy_space.exists() and ignored.exists()
    assert blob.read_text(encoding="utf-8") == "keep"
    assert len(warnings) == 2
    assert all("sensitive-token" not in warning and "64-*" not in warning for warning in warnings)


def test_audit_leaves_healthy_unknown_and_malformed_metadata_untouched(tmp_path, monkeypatch):
    hub = tmp_path / "hf-home" / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    trees = {
        _tree_path(hub, "nvidia/healthy", "one"): b'{"format_version":1,"files":{"x":{"xet_hash":"64-a"}}}',
        _tree_path(hub, "nvidia/unknown", "two"): b'{"format_version":2,"files":{"x":{"xet_hash":"64-*"}}}',
        _tree_path(hub, "nvidia/malformed", "three"): b"{not-json",
    }
    for path, payload in trees.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    warnings: list[str] = []

    assert _audit_hub_tree_metadata(warnings.append) == []
    assert {path: path.read_bytes() for path in trees} == trees
    assert warnings == []


def test_repeated_audit_preserves_existing_evidence_without_duplicates(tmp_path, monkeypatch):
    hub = tmp_path / "hf-home" / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    tree = _tree_path(hub, "nvidia/model", "revision")
    tree.parent.mkdir(parents=True)
    original = _poisoned_tree_bytes()
    tree.write_bytes(original)
    warnings: list[str] = []

    first = _audit_hub_tree_metadata(warnings.append)
    second = _audit_hub_tree_metadata(warnings.append)

    assert len(first) == 1 and second == []
    assert first[0].read_bytes() == original and not tree.exists()
    assert len(warnings) == 1


def test_effective_hub_cache_respects_hugging_face_precedence(tmp_path):
    env = {
        "HF_HUB_CACHE": str(tmp_path / "explicit"),
        "HUGGINGFACE_HUB_CACHE": str(tmp_path / "legacy"),
        "HF_HOME": str(tmp_path / "hf"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg"),
    }
    assert _effective_hub_cache(env) == tmp_path / "explicit"
    del env["HF_HUB_CACHE"]
    assert _effective_hub_cache(env) == tmp_path / "legacy"
    del env["HUGGINGFACE_HUB_CACHE"]
    assert _effective_hub_cache(env) == tmp_path / "hf" / "hub"
    del env["HF_HOME"]
    assert _effective_hub_cache(env) == tmp_path / "xdg" / "huggingface" / "hub"


def test_audit_surfaces_quarantine_rename_errors(tmp_path, monkeypatch):
    hub = tmp_path / "hf-home" / "hub"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    tree = _tree_path(hub, "nvidia/model", "revision")
    tree.parent.mkdir(parents=True)
    tree.write_bytes(_poisoned_tree_bytes())

    def deny_rename(source, destination):
        raise PermissionError("read-only cache")

    monkeypatch.setattr(service_adapter.os, "rename", deny_rename)
    with pytest.raises(RuntimeError, match="cannot quarantine poisoned Hugging Face tree metadata"):
        _audit_hub_tree_metadata()
    assert tree.exists()


def test_upstream_port_flag_does_not_abbreviate_portable_contract():
    wrapper, remaining = _parse_wrapper_args(
        [
            "--portable-contract",
            "/tmp/policy_contract.runtime.json",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )

    assert wrapper.portable_contract == Path("/tmp/policy_contract.runtime.json")
    assert remaining == ["--host", "127.0.0.1", "--port", "8000"]


def test_runtime_command_uses_module_python_for_lightweight_provisioner(tmp_path):
    runtime = tmp_path / "runtime"
    command = runtime_command(runtime, tmp_path / "model", "127.0.0.1", 8000)

    assert command[:2] == [sys.executable, str(runtime / "launcher.py")]
    assert command[2:] == [
        "--model",
        str(tmp_path / "model"),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def _contract():
    return {
        "action_chunk_size": 32,
        "action_schema": {"shape": [7]},
        "conditioning_fps": 15.0,
        "domain_name": "droid_lerobot",
        "framework": {"action_space": "joint_pos", "use_state": True},
        "transform": {"format_prompt_as_json": True, "resolution": "480"},
    }


def test_server_command_passes_the_trained_camera_composite_size(tmp_path):
    meta = tmp_path / "runtime/dataset/meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.wrist_image_left": {"shape": [480, 640, 3]},
                    "observation.images.exterior_image_1_left": {"shape": [720, 1280, 3]},
                }
            }
        )
    )
    command = service_adapter.PortableRobolabPolicyService.command(
        model_view=tmp_path, contract=_contract(), host="127.0.0.1", port=8000
    )
    height = command.index("--image-height")
    assert command[height : height + 4] == ["--image-height", "720", "--image-width", "640"]


def test_server_command_keeps_upstream_image_size_without_dataset_info(tmp_path):
    command = service_adapter.PortableRobolabPolicyService.command(
        model_view=tmp_path, contract=_contract(), host="127.0.0.1", port=8000
    )
    assert "--image-height" not in command
    assert "--image-width" not in command


def test_server_command_override_file_wins_over_trained_size(tmp_path, monkeypatch):
    meta = tmp_path / "runtime/dataset/meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"features": {"observation.images.wrist_image_left": {"shape": [480, 640, 3]}}})
    )
    monkeypatch.setattr(
        service_adapter, "_override_args", lambda: ["--image-height", "540", "--image-width", "640"]
    )
    command = service_adapter.PortableRobolabPolicyService.command(
        model_view=tmp_path, contract=_contract(), host="127.0.0.1", port=8000
    )
    assert command.count("--image-height") == 1
    assert command[command.index("--image-height") + 1] == "540"


def test_override_file_is_read_from_the_runtime_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(service_adapter, "__file__", str(tmp_path / "service_adapter.py"))
    assert service_adapter._override_args() == []
    (tmp_path / "server_overrides.json").write_text('{"args": ["--num-steps", "8"]}')
    assert service_adapter._override_args() == ["--num-steps", "8"]
    (tmp_path / "server_overrides.json").write_text('{"args": "bad"}')
    with pytest.raises(SystemExit):
        service_adapter._override_args()
