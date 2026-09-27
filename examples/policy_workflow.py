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
from typing import Any, Mapping, Sequence, cast

from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

HOME_JOINT_DEGREES = (0.0, 0.0, 90.0, 0.0, 90.0, 0.0)
PICK_REGION_MIN_M = (0.25, -0.12, 0.0)
PICK_REGION_MAX_M = (0.38, 0.12, 0.0)
PICK_YAW_RANGE_DEG = (-180.0, 180.0)
WORLD_FRAME = "world"
DEFAULT_GRIPPER_SETTLE_SEC = 3.0


async def _world_pick_region(machine: Any, arm_frame: str) -> dict[str, list[float]]:
    corners: list[tuple[float, float]] = []
    for x_m in (PICK_REGION_MIN_M[0], PICK_REGION_MAX_M[0]):
        for y_m in (PICK_REGION_MIN_M[1], PICK_REGION_MAX_M[1]):
            transformed = await machine.transform_pose(
                PoseInFrame(
                    reference_frame=f"{arm_frame}:base",
                    pose=Pose(x=x_m * 1000.0, y=y_m * 1000.0, o_z=1),
                ),
                WORLD_FRAME,
            )
            corners.append((transformed.pose.x / 1000.0, transformed.pose.y / 1000.0))
    xs, ys = zip(*corners)
    support_height = PICK_REGION_MIN_M[2]
    return {
        "min": [min(xs), min(ys), support_height],
        "max": [max(xs), max(ys), support_height],
    }


async def _arm_base_pose(machine: Any, arm_frame: str) -> Pose:
    transformed = await machine.transform_pose(
        PoseInFrame(reference_frame=f"{arm_frame}:base", pose=Pose(o_z=1)),
        WORLD_FRAME,
    )
    pose = transformed.pose
    values = (pose.x, pose.y, pose.z, pose.o_x, pose.o_y, pose.o_z, pose.theta)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("arm base pose contains non-finite values")
    return pose


async def prepare_lite6_setup(
    machine: Any,
    world: Any | None,
    arm: Any,
    gripper: Any,
    *,
    arm_frame: str,
    seed: int,
    prop: str,
    gripper_settle_sec: float = DEFAULT_GRIPPER_SETTLE_SEC,
) -> Mapping[str, Any]:
    """Home the Lite6 and optionally reset its simulated pick scene."""
    await arm.move_to_joint_positions(JointPositions(values=list(HOME_JOINT_DEGREES)))
    await gripper.open()
    await asyncio.sleep(gripper_settle_sec)
    robot_setup = {
        "performed": True,
        "robot_prepared": True,
        "home_joint_degrees": list(HOME_JOINT_DEGREES),
    }
    if world is None:
        return {**robot_setup, "scene_reset": False}
    base_pose, world_region = await asyncio.gather(
        _arm_base_pose(machine, arm_frame),
        _world_pick_region(machine, arm_frame),
    )
    reset = await world.do_command(
        {
            "command": "reset_props",
            "seed": seed,
            "props": [prop],
            "region": world_region,
            "yaw_range_deg": list(PICK_YAW_RANGE_DEG),
        }
    )
    poses = reset.get("poses")
    if not isinstance(poses, Mapping) or prop not in poses:
        raise RuntimeError(f"scene reset did not return pose for {prop!r}")
    return {
        **robot_setup,
        "scene_reset": True,
        "seed": int(cast(Any, reset.get("seed", seed))),
        "prop": prop,
        "pose": poses[prop],
        "region": world_region,
        "region_in_arm_frame": {
            "min": list(PICK_REGION_MIN_M),
            "max": list(PICK_REGION_MAX_M),
        },
        "arm_base_pose_world": {
            "position_m": [
                base_pose.x / 1000.0,
                base_pose.y / 1000.0,
                base_pose.z / 1000.0,
            ],
            "orientation_vector_degrees": [
                base_pose.o_x,
                base_pose.o_y,
                base_pose.o_z,
                base_pose.theta,
            ],
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
            setup = await prepare_lite6_setup(
                machine,
                (
                    Generic.from_robot(machine, args.world)
                    if args.world is not None
                    else None
                ),
                Arm.from_robot(machine, args.arm),
                Gripper.from_robot(machine, args.gripper),
                arm_frame=args.arm,
                seed=args.seed,
                gripper_settle_sec=args.gripper_settle_sec,
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
        if args.progress_prop:
            command["progress_prop"] = args.progress_prop
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", help="machine address from the Viam CONNECT tab")
    parser.add_argument("--policy", default="policy")
    parser.add_argument(
        "--world",
        help="world resource; providing it enables deterministic Lite6 scene setup",
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
        "--progress-prop",
        default="",
        help="optional world prop queried before and after each executed prefix",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="deterministic scene reset seed"
    )
    parser.add_argument(
        "--gripper-settle-sec",
        type=float,
        default=DEFAULT_GRIPPER_SETTLE_SEC,
        help="seconds to wait after the nonblocking setup Open command",
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
    args = parser.parse_args(argv)
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
    if args.gripper_settle_sec < 0:
        parser.error("--gripper-settle-sec must be non-negative")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
