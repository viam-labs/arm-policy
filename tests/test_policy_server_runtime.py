import sys
from pathlib import Path

from models.policy_server import runtime_command
from policy_server_runtime.service_adapter import _parse_wrapper_args


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
