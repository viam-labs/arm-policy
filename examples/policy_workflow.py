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
import random
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
PICK_BLOCK_WIDTH_M = 0.075
PICK_BLOCK_DEPTH_M = 0.015
PICK_BLOCK_HEIGHT_M = 0.025


def policy_arm_model(status: Mapping[str, Any]) -> str:
    embodiment = status.get("embodiment")
    arm_model = embodiment.get("arm_model") if isinstance(embodiment, Mapping) else None
    if isinstance(arm_model, str) and arm_model:
        return arm_model
    state = status.get("state", "unknown")
    detail = status.get("last_error")
    reason = f": {detail}" if isinstance(detail, str) and detail else ""
    raise RuntimeError(
        f"policy contract is not ready (state={state!r}){reason}; "
        "run models/download or correct model_source/model_data_root"
    )


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


def sample_prop_pose(
    seed: int, region: Mapping[str, Sequence[float]]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Sample a deterministic block pose contained within a support region."""
    region_min = tuple(float(value) for value in region["min"])
    region_max = tuple(float(value) for value in region["max"])
    if len(region_min) != 3 or len(region_max) != 3:
        raise ValueError("pick region bounds must each contain three values")
    rng = random.Random(seed)
    yaw_deg = rng.uniform(*PICK_YAW_RANGE_DEG)
    yaw_rad = math.radians(yaw_deg)
    half_x = (
        abs(math.cos(yaw_rad)) * PICK_BLOCK_WIDTH_M
        + abs(math.sin(yaw_rad)) * PICK_BLOCK_DEPTH_M
    ) / 2.0
    half_y = (
        abs(math.sin(yaw_rad)) * PICK_BLOCK_WIDTH_M
        + abs(math.cos(yaw_rad)) * PICK_BLOCK_DEPTH_M
    ) / 2.0
    x_min, x_max = region_min[0] + half_x, region_max[0] - half_x
    y_min, y_max = region_min[1] + half_y, region_max[1] - half_y
    if x_min > x_max or y_min > y_max:
        raise ValueError(f"pick region cannot contain the block at yaw {yaw_deg}")
    half_yaw = yaw_rad / 2.0
    return (
        (
            rng.uniform(x_min, x_max),
            rng.uniform(y_min, y_max),
            region_min[2] + PICK_BLOCK_HEIGHT_M / 2.0,
        ),
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
    )


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
    position, orientation = sample_prop_pose(seed, world_region)
    pose = await world.do_command(
        {
            "command": "set_prop_pose",
            "prop": prop,
            "position": list(position),
            "orientation_wxyz": list(orientation),
        }
    )
    if not isinstance(pose, Mapping) or not all(
        key in pose for key in ("position", "orientation_wxyz")
    ):
        raise RuntimeError(f"scene placement returned no pose for {prop!r}")
    return {
        **robot_setup,
        "scene_reset": True,
        "seed": seed,
        "prop": prop,
        "pose": dict(pose),
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
            arm_model = policy_arm_model(before)
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
    if (
        args.action_prefix_duration_sec is not None
        and args.action_prefix_duration_sec <= 0
    ):
        parser.error("--action-prefix-duration-sec must be positive")
    if args.deadline_tolerance_sec is not None and args.deadline_tolerance_sec <= 0:
        parser.error("--deadline-tolerance-sec must be positive")
    if args.gripper_settle_sec < 0:
        parser.error("--gripper-settle-sec must be non-negative")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
