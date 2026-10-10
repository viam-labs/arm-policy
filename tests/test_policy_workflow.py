import asyncio

import pytest
from viam.proto.common import Pose, PoseInFrame

from examples.policy_workflow import (
    HOME_JOINT_DEGREES,
    PICK_REGION_MAX_M,
    PICK_REGION_MIN_M,
    parse_args,
    policy_arm_model,
    prepare_lite6_setup,
    sample_prop_pose,
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


def test_policy_arm_model_reports_contract_readiness_error():
    with pytest.raises(
        RuntimeError,
        match="configured model_source has not been downloaded and validated",
    ):
        policy_arm_model(
            {
                "state": "wait_contract",
                "embodiment": {"arm_model": ""},
                "last_error": "configured model_source has not been downloaded and validated",
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
