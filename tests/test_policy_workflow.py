import asyncio
import json

import pytest
from viam.proto.common import Pose, PoseInFrame

from examples.policy_workflow import (
    HOME_JOINT_DEGREES,
    PICK_REGION_MAX_M,
    PICK_REGION_MIN_M,
    parse_args,
    policy_arm_model,
    prepare_lite6_setup,
    robot_client_options,
    sample_prop_pose,
    workflow_summary,
)


class FakeMachine:
    def __init__(self, pose):
        self.pose = pose

    async def transform_pose(self, query, destination):
        assert query.reference_frame == "pick-arm:base"
        assert query.pose.o_z == 1
        assert destination == "world"
        if self.pose.theta == 90:
            x = self.pose.x - query.pose.y
            y = self.pose.y + query.pose.x
        else:
            x = self.pose.x + query.pose.x
            y = self.pose.y + query.pose.y
        return PoseInFrame(
            reference_frame=destination,
            pose=Pose(
                x=x,
                y=y,
                z=self.pose.z + query.pose.z,
                o_x=self.pose.o_x,
                o_y=self.pose.o_y,
                o_z=self.pose.o_z,
                theta=self.pose.theta,
            ),
        )


class FakeArm:
    def __init__(self, events):
        self.events = events

    async def move_to_joint_positions(self, positions):
        self.events.append(("home", positions.values))


class FakeGripper:
    def __init__(self, events):
        self.events = events

    async def open(self):
        self.events.append(("open",))


class FakeWorld:
    def __init__(self, events, *, include_pose=True):
        self.events = events
        self.include_pose = include_pose

    async def do_command(self, command):
        self.events.append(("place", command))
        if not self.include_pose:
            return {}
        return {
            "position": command["position"],
            "orientation_wxyz": command["orientation_wxyz"],
        }


def test_world_is_optional_and_enables_scene_setup_explicitly():
    without_world = parse_args(["machine.example"])
    with_world = parse_args(["machine.example", "--world", "sim-world"])

    assert without_world.world is None
    assert with_world.world == "sim-world"


def test_viam_config_auth_is_default_and_api_keys_take_precedence(
    tmp_path, monkeypatch
):
    default_args = parse_args(["machine.example"])
    assert default_args.viam_config == "/etc/viam.json"

    config_path = tmp_path / "viam.json"
    config_path.write_text(
        json.dumps({"cloud": {"id": "part-id", "secret": "part-secret"}}),
        encoding="utf-8",
    )
    args = parse_args(["machine.example", "--viam-config", str(config_path)])
    options = robot_client_options(args)
    assert options.dial_options.auth_entity == "part-id"
    assert options.dial_options.credentials.type == "robot-secret"
    assert options.dial_options.credentials.payload == "part-secret"

    config_path.write_text("not json", encoding="utf-8")
    monkeypatch.setenv("VIAM_API_KEY", "api-key")
    monkeypatch.setenv("VIAM_API_KEY_ID", "00000000-0000-0000-0000-000000000001")
    options = robot_client_options(args)
    assert options.dial_options.credentials.type == "api-key"
    assert options.dial_options.credentials.payload == "api-key"


def test_policy_arm_model_reports_contract_readiness_error():
    with pytest.raises(
        RuntimeError,
        match="invalid required JSON file policy_contract.json",
    ):
        policy_arm_model(
            {
                "state": "wait_contract",
                "embodiment": {"arm_model": ""},
                "last_error": "invalid required JSON file policy_contract.json",
            }
        )

    assert (
        policy_arm_model({"state": "ready", "embodiment": {"arm_model": "lite6"}})
        == "lite6"
    )


def test_lite6_setup_matches_episode_capture_initialization():
    events = []
    result = asyncio.run(
        prepare_lite6_setup(
            FakeMachine(Pose(x=1000, y=2000, z=500, o_z=1, theta=90)),
            FakeWorld(events),
            FakeArm(events),
            FakeGripper(events),
            seed=7,
            arm_frame="pick-arm",
            prop="pick_block",
            gripper_settle_sec=0.0,
        )
    )

    assert events[0] == ("home", list(HOME_JOINT_DEGREES))
    assert events[1] == ("open",)
    placement = events[2][1]
    assert placement["command"] == "set_prop_pose"
    assert placement["prop"] == "pick_block"
    assert 0.88 <= placement["position"][0] <= 1.12
    assert 2.25 <= placement["position"][1] <= 2.38
    assert placement["position"][2] == pytest.approx(0.0125)
    assert sum(value**2 for value in placement["orientation_wxyz"]) == pytest.approx(
        1.0
    )
    assert result["performed"] is True
    assert result["robot_prepared"] is True
    assert result["scene_reset"] is True
    assert result["seed"] == 7
    assert result["pose"] == {
        "position": placement["position"],
        "orientation_wxyz": placement["orientation_wxyz"],
    }
    assert result["region_in_arm_frame"] == {
        "min": list(PICK_REGION_MIN_M),
        "max": list(PICK_REGION_MAX_M),
    }
    assert result["arm_base_pose_world"] == {
        "position_m": [1.0, 2.0, 0.5],
        "orientation_vector_degrees": [0.0, 0.0, 1.0, 90.0],
    }


def test_prop_pose_sampling_is_seeded_and_footprint_safe():
    region = {"min": [0.25, -0.12, 0.0], "max": [0.38, 0.12, 0.0]}

    first = sample_prop_pose(10, region)
    replay = sample_prop_pose(10, region)

    assert replay == first
    assert first[0][2] == pytest.approx(0.0125)
    assert 0.25 <= first[0][0] <= 0.38
    assert -0.12 <= first[0][1] <= 0.12


def test_lite6_setup_without_world_still_homes_arm_and_opens_gripper():
    events = []

    result = asyncio.run(
        prepare_lite6_setup(
            object(),
            None,
            FakeArm(events),
            FakeGripper(events),
            seed=0,
            arm_frame="pick-arm",
            prop="pick_block",
            gripper_settle_sec=0.0,
        )
    )

    assert events == [
        ("home", list(HOME_JOINT_DEGREES)),
        ("open",),
    ]
    assert result == {
        "performed": True,
        "robot_prepared": True,
        "scene_reset": False,
        "home_joint_degrees": list(HOME_JOINT_DEGREES),
    }


def test_lite6_scene_setup_rejects_missing_placement_pose():
    with pytest.raises(RuntimeError, match="scene placement returned no pose"):
        asyncio.run(
            prepare_lite6_setup(
                FakeMachine(Pose()),
                FakeWorld([], include_pose=False),
                FakeArm([]),
                FakeGripper([]),
                seed=0,
                arm_frame="pick-arm",
                prop="pick_block",
                gripper_settle_sec=0.0,
            )
        )


def test_workflow_summary_is_bounded_and_reports_run_outcome():
    args = parse_args(
        [
            "machine.example",
            "--policy",
            "inline-policy",
            "--prompt",
            "Pick up the block",
            "--max-chunks",
            "40",
            "--conditioner-algorithm",
            "toppra",
            "--action-prefix-duration-sec",
            "0.8",
            "--output",
            "run.json",
        ]
    )
    before = {
        "embodiment": {
            "arm_model": "lite6",
            "gripper_model": "lite6_gripper",
            "gripper_mode": "binary",
        }
    }
    setup = {
        "performed": True,
        "scene_reset": True,
        "prop": "pick_block",
        "seed": 10,
        "pose": {"position": [0.31, -0.04, 0.0125]},
    }
    result = {
        **before,
        "state": "failed",
        "error": "ValueError: action 0 TCP is below workspace floor",
        "safety_profile": "lite6-integrated-gripper-sim-v2",
        "conditioner_algorithm": "toppra",
        "prompt": "Pick up the block",
        "started_at": "2026-10-10T17:05:29+00:00",
        "finished_at": "2026-10-10T17:06:43+00:00",
        "inference_responses": 21,
        "inference_ms": [2100.0, 2200.0],
        "executed_prefixes": 20,
        "actions_executed": 315,
        "conditioning": [
            {
                "raw_metrics": {
                    "workspace": {
                        "tcp_min_m": [0.1, 0.0, 0.03],
                        "minimum_floor_margin_m": 0.01,
                    }
                },
                "deferred_horizon_rejection_reason": "below floor",
                "prefix_truncation_reason": "below floor",
                "fallback_reason": "TOPPRA dynamic limit",
            }
        ],
        "progress": [
            {"tcp_position_m": [0.2, 0.0, 0.38], "gripper_position": 0.0},
            {"tcp_position_m": [0.3, 0.0, 0.021], "gripper_position": 0.0},
        ],
        "actuation_timing": [{"barrier_transition": 0}],
    }

    summary = workflow_summary(args, before, setup, result)
    lines = summary.splitlines()

    assert 10 <= len(lines) <= 30
    assert "Result:     FAILED after 74.0 s" in summary
    assert "Inference:  21 responses; mean=2.15 s" in summary
    assert "Execution:  20 prefixes; 315 action points" in summary
    assert "minimum measured margin=0.0010 m" in summary
    assert "Output:     run.json" in summary
    assert "raw_action_chunks" not in summary
