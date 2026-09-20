import asyncio

import pytest

from examples.policy_workflow import (
    HOME_JOINT_DEGREES,
    PICK_REGION_MAX_M,
    PICK_REGION_MIN_M,
    PICK_YAW_RANGE_DEG,
    prepare_lite6_scene,
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
        self.events.append(("reset", command))
        poses = (
            {"pick_block": {"position": [0.3, 0.0, 0.0125]}}
            if self.include_pose
            else {}
        )
        return {"seed": command["seed"], "poses": poses}


def test_lite6_scene_setup_matches_episode_capture_initialization():
    events = []
    result = asyncio.run(
        prepare_lite6_scene(
            FakeWorld(events),
            FakeArm(events),
            FakeGripper(events),
            seed=7,
            prop="pick_block",
        )
    )

    assert events[0] == ("home", list(HOME_JOINT_DEGREES))
    assert events[1] == ("open",)
    reset = events[2][1]
    assert reset == {
        "command": "reset_props",
        "seed": 7,
        "props": ["pick_block"],
        "region": {
            "min": list(PICK_REGION_MIN_M),
            "max": list(PICK_REGION_MAX_M),
        },
        "yaw_range_deg": list(PICK_YAW_RANGE_DEG),
    }
    assert result["performed"] is True
    assert result["seed"] == 7
    assert result["pose"] == {"position": [0.3, 0.0, 0.0125]}


def test_lite6_scene_setup_rejects_missing_reset_pose():
    with pytest.raises(RuntimeError, match="did not return pose"):
        asyncio.run(
            prepare_lite6_scene(
                FakeWorld([], include_pose=False),
                FakeArm([]),
                FakeGripper([]),
                seed=0,
                prop="pick_block",
            )
        )
