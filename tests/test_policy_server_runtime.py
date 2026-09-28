from pathlib import Path

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
