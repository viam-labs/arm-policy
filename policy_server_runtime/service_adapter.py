"""Local Cosmos service adapter for the portable arm-policy contract.

Cosmos remains unmodified apart from the one-line joint-width patch.  This adapter
owns contract-specific ActionTransformPipeline construction instead of teaching
upstream about an export's historical training dataloader shape.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


def _effective_hub_cache(environ: Mapping[str, str] | None = None) -> Path:
    """Return the Hub cache selected by Hugging Face's environment precedence."""
    env = os.environ if environ is None else environ
    cache = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if cache is None:
        hf_home = env.get("HF_HOME")
        if hf_home:
            cache = str(Path(hf_home) / "hub")
        else:
            xdg_cache_home = env.get("XDG_CACHE_HOME")
            cache = str(Path(xdg_cache_home) / "huggingface" / "hub") if xdg_cache_home else "~/.cache/huggingface/hub"
    return Path(os.path.expandvars(cache)).expanduser().resolve()


def _hub_tree_metadata_paths(hub_cache: Path) -> Iterable[Path]:
    for prefix in ("models--", "datasets--", "spaces--"):
        yield from hub_cache.glob(f"{prefix}*/trees/*.json")


def _has_poisoned_xet_hash(metadata: object) -> bool:
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        return False
    files = metadata.get("files")
    return isinstance(files, dict) and any(
        isinstance(file_metadata, dict)
        and isinstance(file_metadata.get("xet_hash"), str)
        and "*" in file_metadata["xet_hash"]
        for file_metadata in files.values()
    )


def _quarantine_poisoned_tree_metadata(metadata_path: Path, warning: Callable[[str], None]) -> Path | None:
    try:
        source_stat = metadata_path.stat()
        payload = metadata_path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not _has_poisoned_xet_hash(metadata):
        return None

    for _ in range(10):
        quarantine_path = metadata_path.with_name(
            f"{metadata_path.name}.poisoned.{time.time_ns()}.{os.getpid()}.{uuid.uuid4().hex}"
        )
        if quarantine_path.exists():
            continue
        try:
            current_stat = metadata_path.stat()
        except FileNotFoundError:
            return None
        if (current_stat.st_dev, current_stat.st_ino) != (source_stat.st_dev, source_stat.st_ino):
            return None
        try:
            os.rename(metadata_path, quarantine_path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RuntimeError(
                f"cannot quarantine poisoned Hugging Face tree metadata at {metadata_path}: {exc}"
            ) from exc
        warning(
            "[arm-policy] quarantined poisoned Hugging Face tree metadata: "
            f"original={metadata_path} quarantine={quarantine_path}"
        )
        return quarantine_path
    raise RuntimeError(f"cannot reserve a quarantine path for poisoned Hugging Face tree metadata at {metadata_path}")


def _audit_hub_tree_metadata(warning: Callable[[str], None] = logging.warning) -> list[Path]:
    quarantined: list[Path] = []
    for metadata_path in _hub_tree_metadata_paths(_effective_hub_cache()):
        if path := _quarantine_poisoned_tree_metadata(metadata_path, warning):
            quarantined.append(path)
    return quarantined


def _override_args() -> list[str]:
    """Extra upstream server flags from an optional ``server_overrides.json`` next to this file.

    The file holds ``{"args": ["--image-height", "720", ...]}``. It exists so an operator
    can change inference-only settings (input image size, guidance, denoising steps)
    without editing the packaged adapter; absent file means no extra flags.
    """
    path = Path(__file__).with_name("server_overrides.json")
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        args = document["args"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"invalid policy server overrides {path}: {exc}") from exc
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise SystemExit(f"invalid policy server overrides {path}: args must be a list of strings")
    return list(args)


def training_image_size(model_view: Path) -> tuple[int, int] | None:
    """Height and width of the camera composite the model was trained on.

    Training stacks the wrist view on top of the two exterior views, each
    resized to half the wrist size, so the composite is 1.5x the wrist height
    by the wrist width. The upstream server otherwise resizes every composite to
    its DROID default of 540x640, which squashes other camera layouts.
    """
    info_path = model_view / "runtime/dataset/meta/info.json"
    try:
        features = json.loads(info_path.read_text(encoding="utf-8"))["features"]
        height, width = features["observation.images.wrist_image_left"]["shape"][:2]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        return None
    return height + height // 2, width


class PortableRobolabPolicyService:
    @staticmethod
    def command(
        *,
        model_view: Path,
        contract: dict[str, Any],
        host: str,
        port: int,
        python: str | Path = sys.executable,
    ) -> list[str]:
        action = contract["action_schema"]
        framework = contract["framework"]
        transform = contract.get("transform", {})
        overrides = _override_args()
        contract_path = model_view / "policy_contract.runtime.json"
        contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8")
        image_size = training_image_size(model_view)
        image_size_args = (
            ["--image-height", str(image_size[0]), "--image-width", str(image_size[1])]
            if image_size is not None
            and "--image-height" not in overrides
            and "--image-width" not in overrides
            else []
        )
        return [
            str(python), str(Path(__file__).resolve()), "--portable-contract", str(contract_path),
            "--checkpoint-path", str(model_view), "--host", host, "--port", str(port),
            "--domain-name", str(contract["domain_name"]), "--action-space", str(framework["action_space"]),
            "--action-dim", str(action["shape"][0]), "--action-chunk-size", str(contract["action_chunk_size"]),
            "--conditioning-fps", str(contract["conditioning_fps"]),
            "--format-prompt-as-json", str(bool(transform.get("format_prompt_as_json", False))),
            "--resolution", str(transform.get("resolution", "720p")),
            "--guidance", "3.0", "--guidance-interval", "960", "1001", "--num-steps", "4", "--shift", "5.0",
            "--use-state" if framework["use_state"] else "--no-use-state",
            *image_size_args,
            *overrides,
        ]


def _parse_wrapper_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--portable-contract", type=Path, required=True)
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper, remaining = _parse_wrapper_args()
    _audit_hub_tree_metadata()
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
        def _build_setup_args(self, args: Any) -> Any:
            # Cosmos guardrails are prompt-safety models fetched from a gated
            # Hugging Face repository; the module does not depend on Hugging
            # Face, and arm safety is enforced by the module's own limits.
            return super()._build_setup_args(args).model_copy(
                update={"guardrails": False}
            )

        def _build_transform(self, training_config: Any | None, args: Any) -> tuple[Any, dict[str, Any]]:
            del training_config
            model_max = getattr(getattr(self.model, "config", None), "max_action_dim", 64)
            transform = contract.get("transform", {})

            # Do not inject an idle-frame count into the request. With
            # append_idle_frames, training prompts carry "idle_frame: N out of
            # 33" computed from the chunk's own future actions, so the model
            # learns to read N as "how long to hold still". A request has no
            # future. Serving a constant 0 (NVIDIA's LIBERO choice) tells the
            # model every chunk is a no-pause chunk: on the Lite6 pick task it
            # descends to the cube and never closes (0/15 and 0/4 on Oct 13,
            # 0/1 on arm-policy 0.0.4 on Oct 14). Leaving the field out lets the
            # model decide from the images, which grabs 9/10 to 10/10 across
            # the multi-1 models with the same executor.
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
