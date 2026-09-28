"""Local Cosmos service adapter for the portable arm-policy contract.

Cosmos remains unmodified apart from the one-line joint-width patch.  This adapter
owns contract-specific ActionTransformPipeline construction instead of teaching
upstream about an export's historical training dataloader shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class PortableRobolabPolicyService:
    @staticmethod
    def command(*, model_view: Path, contract: dict[str, Any], host: str, port: int) -> list[str]:
        action = contract["action_schema"]
        framework = contract["framework"]
        transform = contract.get("transform", {})
        contract_path = model_view / "policy_contract.runtime.json"
        contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
        return [
            sys.executable, str(Path(__file__).resolve()), "--portable-contract", str(contract_path),
            "--checkpoint-path", str(model_view), "--host", host, "--port", str(port),
            "--domain-name", str(contract["domain_name"]), "--action-space", str(framework["action_space"]),
            "--action-dim", str(action["shape"][0]), "--action-chunk-size", str(contract["action_chunk_size"]),
            "--conditioning-fps", str(contract["conditioning_fps"]),
            "--format-prompt-as-json", str(bool(transform.get("format_prompt_as_json", False))),
            "--resolution", str(transform.get("resolution", "720p")),
            "--guidance", "3.0", "--guidance-interval", "960", "1001", "--num-steps", "4", "--shift", "5.0",
            "--use-state" if framework["use_state"] else "--no-use-state",
        ]


def _parse_wrapper_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--portable-contract", type=Path, required=True)
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper, remaining = _parse_wrapper_args()
    try:
        contract = json.loads(wrapper.portable_contract.read_text(encoding="utf-8"))
        action_dim = contract["action_schema"]["shape"][0]
        if not isinstance(action_dim, int) or action_dim < 2:
            raise ValueError("action schema must contain at least one joint and a gripper")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid portable policy contract: {exc}") from exc
    from cosmos_framework.data.generator.action.utils.transforms import (
        ActionTransformPipeline,
    )
    from cosmos_framework.scripts import action_policy_server_robolab as upstream

    class ContractRobolabPolicyService(upstream.RobolabPolicyService):
        def _build_transform(self, training_config: Any | None, args: Any) -> tuple[Any, dict[str, Any]]:
            del training_config
            model_max = getattr(getattr(self.model, "config", None), "max_action_dim", 64)
            transform = contract.get("transform", {})
            return (
                ActionTransformPipeline(
                    max_action_dim=int(model_max),
                    cfg_dropout_rate=0.0,
                    format_prompt_as_json=bool(transform.get("format_prompt_as_json", False)),
                    append_viewpoint_info=bool(transform.get("append_viewpoint_info", True)),
                    append_duration_fps_timestamps=bool(transform.get("append_duration_fps_timestamps", True)),
                    append_resolution_info=bool(transform.get("append_resolution_info", True)),
                    append_idle_frames=bool(transform.get("append_idle_frames", False)),
                ),
                {"action_chunk_size": int(contract["action_chunk_size"]), "conditioning_fps": float(contract["conditioning_fps"]), "resolution": str(transform.get("resolution", "720p"))},
            )

    sys.argv = [sys.argv[0], *remaining]
    args = upstream.tyro_cli(upstream.RobolabServerArgs, description=upstream.__doc__)
    if args.action_dim is None or args.action_dim < 2:
        raise SystemExit("portable policy server requires a positive joint-plus-gripper action dimension")
    service = ContractRobolabPolicyService(args)
    server_cls = upstream._load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


if __name__ == "__main__":
    main()
