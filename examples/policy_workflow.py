#!/usr/bin/env python3
"""Run one real Cosmos policy workflow through Viam's production Generic API.

Success means the server produced policy output followed by safe execution or an
explicit safe rejection. It does not assert that the simulated pick succeeded.
"""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, cast

from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

HOME_JOINT_DEGREES = (0.0, 0.0, 90.0, 0.0, 90.0, 0.0)
PICK_REGION_MIN_M = (0.25, -0.12, 0.0)
PICK_REGION_MAX_M = (0.38, 0.12, 0.0)
PICK_YAW_RANGE_DEG = (-180.0, 180.0)


async def prepare_lite6_scene(
    world: Any,
    arm: Any,
    gripper: Any,
    *,
    seed: int,
    prop: str,
) -> Mapping[str, Any]:
    """Put the simulator in the same initial state used by episode capture."""
    await arm.move_to_joint_positions(JointPositions(values=list(HOME_JOINT_DEGREES)))
    await gripper.open()
    reset = await world.do_command(
        {
            "command": "reset_props",
            "seed": seed,
            "props": [prop],
            "region": {
                "min": list(PICK_REGION_MIN_M),
                "max": list(PICK_REGION_MAX_M),
            },
            "yaw_range_deg": list(PICK_YAW_RANGE_DEG),
        }
    )
    poses = reset.get("poses")
    if not isinstance(poses, Mapping) or prop not in poses:
        raise RuntimeError(f"scene reset did not return pose for {prop!r}")
    return {
        "performed": True,
        "seed": int(cast(Any, reset.get("seed", seed))),
        "prop": prop,
        "pose": poses[prop],
        "home_joint_degrees": list(HOME_JOINT_DEGREES),
        "region": {
            "min": list(PICK_REGION_MIN_M),
            "max": list(PICK_REGION_MAX_M),
        },
        "yaw_range_deg": list(PICK_YAW_RANGE_DEG),
    }


async def run(args: argparse.Namespace) -> Mapping[str, Any]:
    api_key = os.environ.get("VIAM_API_KEY")
    api_key_id = os.environ.get("VIAM_API_KEY_ID")
    if bool(api_key) != bool(api_key_id):
        raise ValueError("set both VIAM_API_KEY and VIAM_API_KEY_ID, or neither")
    options = (
        RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
        if api_key and api_key_id
        else RobotClient.Options(dial_options=DialOptions(insecure=True))
    )
    if args.disable_webrtc:
        assert options.dial_options is not None
        options.dial_options.disable_webrtc = True

    async with await RobotClient.at_address(args.address, options) as machine:
        policy = Generic.from_robot(machine, args.policy)
        before = await policy.do_command({"command": "status"})
        print(json.dumps({"before": before}, indent=2, sort_keys=True))
        setup: Mapping[str, Any] = {"performed": False}
        if not args.no_setup:
            embodiment = before.get("embodiment")
            arm_model = (
                embodiment.get("arm_model") if isinstance(embodiment, Mapping) else None
            )
            if arm_model != "lite6":
                raise RuntimeError(
                    "automatic scene setup only supports Lite6; pass --no-setup "
                    f"for embodiment {arm_model!r}"
                )
            setup = await prepare_lite6_scene(
                Generic.from_robot(machine, args.world),
                Arm.from_robot(machine, args.arm),
                Gripper.from_robot(machine, args.gripper),
                seed=args.seed,
                prop=args.prop,
            )
            print(json.dumps({"setup": setup}, indent=2, sort_keys=True))
        command: dict[str, Any] = {
            "command": "execute",
            "max_chunks": args.max_chunks,
        }
        if args.action_prefix_steps is not None:
            command["action_prefix_steps"] = args.action_prefix_steps
        if args.action_prefix_duration_sec is not None:
            command["action_prefix_duration_sec"] = args.action_prefix_duration_sec
        if args.conditioner_algorithm:
            command["conditioner_algorithm"] = args.conditioner_algorithm
        if args.deadline_tolerance_sec is not None:
            command["deadline_tolerance_sec"] = args.deadline_tolerance_sec
        if args.prop:
            command["progress_prop"] = args.prop
        if args.prompt:
            command["prompt"] = args.prompt
        result = await policy.do_command(command)
        after = await policy.do_command({"command": "status"})
        evidence = {
            "setup": setup,
            "result": result,
            "status": after,
            "pick_success_claimed": False,
        }
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(evidence, indent=2, sort_keys=True))
        completed = (
            result.get("state") == "completed"
            and int(cast(Any, result.get("chunks_inferred", 0))) > 0
        )
        safe_rejection = (
            result.get("state") in ("failed", "stopped")
            and int(cast(Any, result.get("inference_responses", 0))) > 0
            and bool(result.get("error"))
        )
        if not (completed or safe_rejection):
            raise RuntimeError(
                "workflow produced neither completed inference nor explicit "
                "output rejection"
            )
        return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="machine address from the Viam CONNECT tab")
    parser.add_argument("--policy", default="policy")
    parser.add_argument(
        "--world",
        default="sim-world",
        help="world resource used for default scene setup",
    )
    parser.add_argument(
        "--arm", default="pick-arm", help="arm resource homed before inference"
    )
    parser.add_argument(
        "--gripper", default="pick-gripper", help="gripper opened before inference"
    )
    parser.add_argument(
        "--prop", default="pick_block", help="prop placed in the training region"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="deterministic scene reset seed"
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="skip homing and deterministic Lite6 pick reset",
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-chunks", type=int, default=1)
    prefix = parser.add_mutually_exclusive_group()
    prefix.add_argument("--action-prefix-steps", type=int)
    parser.add_argument("--conditioner-algorithm", choices=("quintic", "toppra"))
    prefix.add_argument("--action-prefix-duration-sec", type=float)
    parser.add_argument("--deadline-tolerance-sec", type=float)
    parser.add_argument(
        "--output", help="write structured workflow evidence to this JSON file"
    )
    parser.add_argument("--disable-webrtc", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_chunks <= 100:
        parser.error("--max-chunks must be in [1, 100]")
    if args.action_prefix_steps is not None and not 1 <= args.action_prefix_steps <= 32:
        parser.error("--action-prefix-steps must be in [1, 32]")
    if args.action_prefix_duration_sec is not None:
        steps = args.action_prefix_duration_sec * 20.0
        if not 0.05 <= args.action_prefix_duration_sec <= 1.6 or not math.isclose(
            steps, round(steps), abs_tol=1e-9
        ):
            parser.error(
                "--action-prefix-duration-sec must be a 20 Hz duration in [0.05, 1.6]"
            )
    if args.deadline_tolerance_sec is not None and args.deadline_tolerance_sec <= 0:
        parser.error("--deadline-tolerance-sec must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
