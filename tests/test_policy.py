import asyncio
import json
import time
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from viam.components.arm import Arm, JointPositions
from viam.media.video import CameraMimeType, ViamImage
from viam.operations import Operation
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from models.policy import (
    CAMERA_ALIASES,
    CAMERA_KEYS,
    SAFETY_PROFILES,
    Policy,
    _read_policy_contract,
    pack_message,
    unpack_message,
)


def _image(width, height, color):
    output = BytesIO()
    Image.new("RGB", (width, height), color).save(output, "PNG")
    return ViamImage(output.getvalue(), CameraMimeType.PNG)


class FakeCamera:
    def __init__(self, image):
        self.image = image

    async def get_images(self):
        return [self.image], None


class FakeArm:
    def __init__(self, joints=None, *, block=False):
        self.joints = list(joints or [0.0] * 6)
        self.targets = []
        self.stops = 0
        self.target_times = []
        self.streamed_batches = []
        self.stream_started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block

    async def get_joint_positions(self):
        return JointPositions(values=[np.degrees(value) for value in self.joints])

    async def move_through_joint_positions_streamed(self, batches):
        async for batch in batches:
            self.streamed_batches.append(batch)
            self.stream_started.set()
            if self.block:
                await self.release.wait()
            else:
                started = time.monotonic()
                for point in batch[1:]:
                    delay = started + point.time.total_seconds() - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    self.targets.append(
                        [np.radians(value) for value in point.positions]
                    )
                    self.target_times.append(time.monotonic())
                if self.targets:
                    self.joints = list(self.targets[-1])
            yield Arm.TrajectoryUpdate()

    async def stop(self):
        self.stops += 1
        self.release.set()

    async def do_command(self, command):
        raise AssertionError(f"arm received custom command: {command}")


class FakeGripper:
    def __init__(self, delay=0.0):
        self.targets = []
        self.stops = 0
        self.delay = delay

    async def get_current_inputs(self):
        return [0.0]

    async def do_command(self, command):
        assert command["command"] == "set_input"
        if self.delay:
            await asyncio.sleep(self.delay)
        self.targets.append(command["value"])
        return {"accepted": True}

    async def stop(self):
        self.stops += 1


class FakeSocket:
    def __init__(self, response=None, block=False):
        self.response = response
        self.block = block
        self.sent = []
        self.closed = asyncio.Event()

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if self.block:
            await self.closed.wait()
        return self.response

    async def close(self):
        self.closed.set()


class FakeWorld:
    async def do_command(self, command):
        assert command == {"command": "get_prop_pose", "prop": "pick_block"}
        return {
            "prop": "pick_block",
            "pose": {"position": [0.3, 0.0, 0.02]},
        }


class FakePolicy(Policy, register=False):
    def __init__(
        self, response, *, arm=None, gripper=None, block=False, gripper_mode="position"
    ):
        super().__init__("policy")
        self._world = FakeWorld()
        self._arm = arm or FakeArm()
        self._gripper = gripper or FakeGripper()
        self._fixed_camera = FakeCamera(_image(1280, 720, (1, 2, 3)))
        self._wrist_camera = FakeCamera(_image(640, 480, (4, 5, 6)))
        self.socket = FakeSocket(response, block)
        self._config.server_uri = "ws://policy.test"
        self._config.prompt = "pick"
        self._config.contract.embodiment.gripper_mode = gripper_mode
        self._config.binary_gripper_settle_sec = 0.0
        self._config._profile = SAFETY_PROFILES[
            ("lite6", "lite6_gripper")
            if gripper_mode == "binary"
            else ("ur20", "robotiq_2f_85")
        ]

    async def _connect(self, operation, generation):
        self._socket = self.socket
        return self.socket, {}


class PhysicalFakeArm:
    def __init__(self, *, block=False):
        self.joints_deg = [0.0] * 6
        self.streamed_batches = []
        self.stream_started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block
        self.stops = 0

    async def get_joint_positions(self):
        return JointPositions(values=list(self.joints_deg))

    async def move_through_joint_positions_streamed(self, batches):
        async for batch in batches:
            self.streamed_batches.append(batch)
            self.stream_started.set()
            if self.block:
                await self.release.wait()
            else:
                await asyncio.sleep(batch[-1].time.total_seconds())
            self.joints_deg = list(batch[-1].positions)
            yield Arm.TrajectoryUpdate()

    async def stop(self):
        self.stops += 1
        self.release.set()

    async def do_command(self, command):
        raise AssertionError(f"physical arm received custom command: {command}")


class PhysicalFakeGripper:
    def __init__(self):
        self.commands = []
        self.stops = 0

    async def open(self):
        self.commands.append(("open", time.monotonic()))

    async def grab(self):
        self.commands.append(("close", time.monotonic()))
        return False

    async def stop(self):
        self.stops += 1

    async def get_current_inputs(self):
        raise AssertionError("physical gripper input API is not normalized")

    async def do_command(self, command):
        raise AssertionError(f"physical gripper received custom command: {command}")


def _actions():
    result = np.zeros((32, 7), dtype=np.float32)
    result[:, 6] = 0.25
    return result


def _config(attrs):
    return ComponentConfig(name="policy", attributes=dict_to_struct(attrs))


def _write_contract(tmp_path, arm_model="ur20"):
    profile = next(
        profile
        for profile in SAFETY_PROFILES.values()
        if profile.arm_model == arm_model
    )
    gripper_mode = "binary" if profile.gripper_model == "lite6_gripper" else "position"
    gripper_units = "binary" if gripper_mode == "binary" else "normalized"
    contract = {
        "schema_version": 3,
        "embodiment": {
            "arm_model": profile.arm_model,
            "gripper_model": profile.gripper_model,
            "gripper_mode": gripper_mode,
            "joint_names": list(profile.joint_names),
        },
        "action_schema": {
            "key": "action",
            "type": "absolute_joint_target",
            "shape": [7],
            "names": [*profile.joint_names, "gripper"],
            "units": [*(["rad"] * 6), gripper_units],
            "fps": 20,
            "gripper_convention": "0=open, 1=closed",
        },
        "observation_schema": {
            "joint_names": list(profile.joint_names),
            "joint_units": "rad",
            "gripper_units": gripper_units,
            "gripper_convention": "0=open, 1=closed",
            "camera_keys": list(CAMERA_KEYS),
            "camera_aliases": CAMERA_ALIASES,
        },
        "framework": {"action_space": "joint_pos", "joint_count": 6, "use_state": True},
        "transform": {
            "camera_keys": list(CAMERA_KEYS),
            "camera_aliases": CAMERA_ALIASES,
        },
        "action_chunk_size": 32,
        "conditioning_fps": 20.0,
        "domain_name": "droid_lerobot",
    }
    path = tmp_path / f"{arm_model}-policy-contract.json"
    path.write_text(json.dumps(contract))
    return path, contract, profile


def test_policy_config_requires_exact_contract_and_typed_dependency_names(tmp_path):
    contract_path, _, profile = _write_contract(tmp_path)
    attrs = {
        "world": "sim-world",
        "arm": "pick-arm",
        "gripper": "pick-gripper",
        "fixed_camera": "scene-cam",
        "wrist_camera": "wrist-cam",
        "server_uri": "ws://127.0.0.1:8000",
        "prompt": "pick",
        "policy_contract_path": str(contract_path),
    }
    dependencies, optional = Policy.validate_config(_config(attrs))
    assert list(dependencies) == [
        "sim-world",
        "pick-arm",
        "pick-gripper",
        "scene-cam",
        "wrist-cam",
    ]
    assert list(optional) == []
    dependencies_without_world, _ = Policy.validate_config(
        _config({key: value for key, value in attrs.items() if key != "world"})
    )
    assert list(dependencies_without_world) == [
        "pick-arm",
        "pick-gripper",
        "scene-cam",
        "wrist-cam",
    ]
    with pytest.raises(ValueError, match="action_chunk_size"):
        Policy.validate_config(_config({**attrs, "action_chunk_size": 31}))
    with pytest.raises(ValueError, match="fixed_camera_width"):
        Policy.validate_config(_config({**attrs, "fixed_camera_width": 640}))
    with pytest.raises(ValueError, match="server_uri"):
        Policy.validate_config(_config({**attrs, "server_uri": "http://bad"}))
    with pytest.raises(ValueError, match="fixed by safety profile"):
        Policy.validate_config(
            _config({**attrs, "max_reach_m": profile.max_reach_m + 1})
        )
    Policy.validate_config(
        _config(
            {
                **attrs,
                "conditioner_algorithm": "toppra",
                "action_prefix_duration_sec": 1.0,
            }
        )
    )
    with pytest.raises(ValueError, match="conditioner_algorithm"):
        Policy.validate_config(_config({**attrs, "conditioner_algorithm": "unknown"}))
    with pytest.raises(ValueError, match="execution_backend"):
        Policy.validate_config(_config({**attrs, "execution_backend": "simulation"}))
    with pytest.raises(ValueError, match="binary_gripper_settle_sec"):
        Policy.validate_config(_config({**attrs, "binary_gripper_settle_sec": -1}))
    with pytest.raises(ValueError, match="set only one"):
        Policy.validate_config(
            _config(
                {
                    **attrs,
                    "action_prefix_steps": 8,
                    "action_prefix_duration_sec": 1.0,
                }
            )
        )
    with pytest.raises(ValueError, match="model_revision"):
        Policy.validate_config(
            _config(
                {
                    **attrs,
                    "model_repository": "owner/policy",
                    "model_revision": "main",
                    "model_data_root": str(tmp_path),
                }
            )
        )
    with pytest.raises(ValueError, match="absolute path"):
        Policy.validate_config(
            _config({**attrs, "model_data_root": "relative-model-store"})
        )

def test_reconfigure_stops_replaced_policy_server(tmp_path, monkeypatch):
    contract_path, _, _ = _write_contract(tmp_path)
    attrs = {
        "arm": "pick-arm",
        "gripper": "pick-gripper",
        "fixed_camera": "scene-cam",
        "wrist_camera": "wrist-cam",
        "server_uri": "ws://127.0.0.1:8000",
        "prompt": "pick",
        "policy_contract_path": str(contract_path),
    }

    class ExistingServer:
        stopped = False

        def stop(self):
            self.stopped = True

    existing = ExistingServer()
    policy = Policy("policy")
    policy._policy_server = existing
    monkeypatch.setattr(Policy, "_dependency", staticmethod(lambda *args: object()))
    policy.reconfigure(_config(attrs), {})
    assert existing.stopped


def test_contract_selects_matching_profile_and_rejects_wrong_joint_order(tmp_path):
    path, contract, profile = _write_contract(tmp_path, "lite6")
    loaded, selected = _read_policy_contract(str(path))
    assert loaded.model_dump(mode="json") == contract
    assert loaded.embodiment.gripper_mode == "binary"
    assert loaded.observation_schema.gripper_units == "binary"
    assert selected is profile

    contract["action_schema"]["names"][0:2] = reversed(
        contract["action_schema"]["names"][0:2]
    )
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="action schema"):
        _read_policy_contract(str(path))


def test_lite6_and_ur20_position_limits_cannot_be_interchanged():
    policy = FakePolicy(b"")
    actions = _actions()
    actions[:, 1] = 2.7
    actions[:, 6] = 2.0
    history = [np.asarray(actions[0, :6]), np.asarray(actions[0, :6])]

    policy._config._profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    with pytest.raises(ValueError, match="lite6 joint position"):
        policy._validate_actions(actions, history, dynamics=False)
    policy._config._profile = SAFETY_PROFILES[("ur20", "robotiq_2f_85")]
    with pytest.raises(ValueError, match="gripper"):
        policy._validate_actions(actions, history, dynamics=False)


def test_fk_workspace_rejection_is_atomic():
    policy = FakePolicy(b"")
    policy._config.workspace_task_min_m = (10.0, 10.0, 10.0)
    policy._config.workspace_task_max_m = (11.0, 11.0, 11.0)
    with pytest.raises(ValueError, match="task workspace"):
        policy._validate_actions(_actions(), [np.zeros(6), np.zeros(6)], dynamics=False)


def test_arm_observation_uses_standard_degree_api():
    async def scenario():
        expected = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        policy = FakePolicy(b"", arm=FakeArm(joints=expected))

        observation = await policy._observation("pick")

        assert observation["observation/joint_position"] == pytest.approx(expected)

    asyncio.run(scenario())


def test_valid_execution_sends_exact_protocol_and_timestamped_prefix():
    async def scenario():
        policy = FakePolicy(
            pack_message({"action": _actions(), "server_timing": {"infer_ms": 3.5}})
        )
        result = await policy.do_command(
            {"command": "execute", "action_prefix_steps": 3}
        )
        assert result["state"] == "completed"
        assert result["actions_executed"] == 3
        assert result["executed_prefixes"] == 1
        assert result["prompt"] == "pick"
        assert np.asarray(result["raw_action_chunks"]).shape == (1, 32, 7)
        assert np.asarray(result["conditioned_action_chunks"]).shape == (1, 32, 7)
        assert result["conditioning"][0]["conditioned"] is False
        assert result["conditioning"][0]["selected_prefix_steps"] == 3
        assert len(policy._arm.targets) == len(policy._gripper.targets) == 3
        assert len(policy._arm.streamed_batches) == 1
        points = policy._arm.streamed_batches[0]
        assert [point.time.total_seconds() for point in points] == pytest.approx(
            [0.0, 0.05, 0.1, 0.15]
        )
        assert points[0].positions == [0.0] * 6
        observation = unpack_message(policy.socket.sent[0])
        assert set(observation) == {
            "prompt",
            "observation/wrist_image_left",
            "observation/exterior_image_1_left",
            "observation/exterior_image_2_left",
            "observation/joint_position",
            "observation/gripper_position",
        }
        intervals = np.diff(policy._arm.target_times)
        assert intervals == pytest.approx([0.05, 0.05], abs=0.02)
        assert observation["prompt"] == "pick"
        assert observation["observation/wrist_image_left"].shape == (480, 640, 3)
        assert observation["observation/exterior_image_1_left"].shape == (720, 1280, 3)
        assert observation["observation/wrist_image_left"].dtype == np.uint8
        assert observation["observation/joint_position"].dtype == np.float32
        assert observation["observation/joint_position"].shape == (6,)
        assert observation["observation/gripper_position"].shape == (1,)
        np.testing.assert_array_equal(
            observation["observation/exterior_image_1_left"],
            observation["observation/exterior_image_2_left"],
        )
        status = await policy.do_command({"command": "status"})
        assert status["state"] == "completed"
        assert policy._arm.stops and policy._gripper.stops

    asyncio.run(scenario())


def test_binary_lite6_stops_at_gripper_barrier_and_settles():
    async def scenario():
        actions = _actions()
        actions[:4, 6] = [0.0, 0.75, 0.75, 0.2]
        arm = PhysicalFakeArm()
        gripper = PhysicalFakeGripper()
        policy = FakePolicy(
            pack_message({"action": actions}),
            arm=arm,
            gripper=gripper,
            gripper_mode="binary",
        )

        result = await policy.do_command(
            {"command": "execute", "action_prefix_steps": 4}
        )

        assert result["state"] == "completed"
        assert result["gripper_mode"] == "binary"
        assert result["actions_executed"] == 2
        assert len(arm.streamed_batches) == 1
        points = arm.streamed_batches[0]
        assert [point.time.total_seconds() for point in points] == pytest.approx(
            [0.0, 0.05, 0.1]
        )
        assert points[0].positions == [0.0] * 6
        assert [command for command, _ in gripper.commands] == ["close"]
        timing = result["actuation_timing"][0]
        assert timing == {
            "gripper_mode": "binary",
            "requested_steps": 4,
            "scheduled_steps": 2,
            "barrier_transition": "grab",
            "settle_sec": 0.0,
            "elapsed_sec": pytest.approx(0.1, abs=0.025),
            "max_dispatch_lateness_sec": 0.0,
        }
        assert arm.stops and gripper.stops

    asyncio.run(scenario())


def test_physical_stop_cancels_stream_and_stops_viam_resources():
    async def scenario():
        arm = PhysicalFakeArm(block=True)
        gripper = PhysicalFakeGripper()
        policy = FakePolicy(
            pack_message({"action": _actions()}),
            arm=arm,
            gripper=gripper,
            gripper_mode="binary",
        )
        executing = asyncio.create_task(
            policy.do_command({"command": "execute", "action_prefix_steps": 20})
        )
        await asyncio.wait_for(arm.stream_started.wait(), timeout=1.0)

        stopped = await policy.do_command({"command": "stop"})
        result = await asyncio.wait_for(executing, timeout=1.0)

        assert result["state"] == stopped["state"] == "stopped"
        assert arm.stops and gripper.stops

    asyncio.run(scenario())


def test_duration_prefix_executes_exact_20_hz_horizon():
    async def scenario():
        policy = FakePolicy(pack_message({"action": _actions()}))

        result = await policy.do_command(
            {
                "command": "execute",
                "action_prefix_duration_sec": 0.5,
                "conditioner_algorithm": "toppra",
                "progress_prop": "pick_block",
            }
        )

        assert result["state"] == "completed"
        assert result["actions_executed"] == 10
        assert result["conditioning"][0]["selected_prefix_duration_s"] == 0.5
        assert result["conditioning"][0]["algorithm"] == (
            "toppra-spline-time-parameterizer"
        )
        assert len(policy._arm.targets) == 10
        assert [record["phase"] for record in result["progress"]] == [
            "before_prefix",
            "after_prefix",
        ]
        assert all("tcp_to_prop_distance_m" in record for record in result["progress"])

    asyncio.run(scenario())


def test_execution_without_world_succeeds_unless_progress_prop_is_requested():
    async def scenario():
        policy = FakePolicy(pack_message({"action": _actions()}))
        policy._world = None

        result = await policy.do_command(
            {"command": "execute", "action_prefix_steps": 1}
        )
        assert result["state"] == "completed"

        with pytest.raises(ValueError, match="progress_prop requires"):
            await policy.do_command(
                {
                    "command": "execute",
                    "action_prefix_steps": 1,
                    "progress_prop": "pick_block",
                }
            )

    asyncio.run(scenario())


def test_finite_gripper_outputs_are_projected_to_actuator_limits():
    async def scenario():
        actions = _actions()
        actions[:3, 6] = (-0.4, 1.4, -0.2)
        policy = FakePolicy(pack_message({"action": actions}))

        result = await policy.do_command(
            {"command": "execute", "action_prefix_steps": 3}
        )

        assert result["state"] == "completed"
        assert policy._gripper.targets == [0.0, 1.0, 0.0]
        assert result["conditioning"][0]["gripper_projection_max"] == pytest.approx(
            0.4, abs=1e-6
        )

    asyncio.run(scenario())


def test_late_workspace_violation_does_not_block_safe_prefix():
    async def scenario():
        actions = _actions().astype(np.float64)
        actions[4:, 0] = 0.4
        policy = FakePolicy(pack_message({"action": actions}))
        profile = SAFETY_PROFILES[("ur20", "robotiq_2f_85")]
        policy._config._profile = replace(
            profile,
            workspace_floor_m=0.1,
            fk=lambda joints, _offset: (
                0.2,
                0.0,
                0.3 if float(joints[0]) < 0.2 else 0.0,
            ),
        )

        result = await policy.do_command(
            {"command": "execute", "action_prefix_steps": 8}
        )

        assert result["state"] == "completed"
        assert result["actions_executed"] == 8
        diagnostics = result["conditioning"][0]
        assert diagnostics["validation_scope"] == "selected_conditioned_prefix"
        assert "workspace floor" in diagnostics["deferred_horizon_rejection_reason"]
        assert diagnostics["rejection_reason"] == ""

    asyncio.run(scenario())


def test_prefix_truncates_before_first_unsafe_conditioned_target():
    async def scenario():
        actions = _actions().astype(np.float64)
        actions[1:, 0] = 0.4
        policy = FakePolicy(pack_message({"action": actions}))
        profile = SAFETY_PROFILES[("ur20", "robotiq_2f_85")]
        policy._config._profile = replace(
            profile,
            workspace_floor_m=0.1,
            fk=lambda joints, _offset: (
                0.2,
                0.0,
                0.3 if float(joints[0]) < 0.2 else 0.0,
            ),
        )

        result = await policy.do_command(
            {"command": "execute", "action_prefix_steps": 8}
        )

        assert result["state"] == "completed"
        assert 0 < result["actions_executed"] < 8
        diagnostics = result["conditioning"][0]
        assert diagnostics["validated_prefix_steps"] == result["actions_executed"]
        assert "workspace floor" in diagnostics["prefix_truncation_reason"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "action,error",
    [
        (np.zeros((31, 7), dtype=np.float32), "shape"),
        (np.full((32, 7), np.nan, dtype=np.float32), "finite"),
        (
            np.column_stack((np.full((32, 6), 7.0), np.zeros(32))).astype(np.float32),
            "position",
        ),
    ],
)
def test_malformed_nonfinite_and_unsafe_chunks_move_nothing(action, error):
    async def scenario():
        policy = FakePolicy(pack_message({"action": action}))
        result = await policy.do_command({"command": "execute"})
        assert result["state"] == "failed"
        assert error in result["error"]
        if action.shape == (32, 7) and np.isfinite(action).all():
            assert np.asarray(result["raw_action_chunks"]).shape == (1, 32, 7)
        assert policy._arm.targets == []
        assert policy._gripper.targets == []
        assert policy._arm.stops and policy._gripper.stops

    asyncio.run(scenario())


def test_text_error_fails_closed_before_motion():
    async def scenario():
        policy = FakePolicy("server traceback")
        result = await policy.do_command({"command": "execute"})
        assert result["state"] == "failed"
        assert "text error" in result["error"]
        assert not policy._arm.targets and not policy._gripper.targets

    asyncio.run(scenario())


def test_stop_cancels_inference_and_holds_both_resources():
    async def scenario():
        policy = FakePolicy(None, block=True)
        executing = asyncio.create_task(policy.do_command({"command": "execute"}))
        while not policy.socket.sent:
            await asyncio.sleep(0)
        stopped = await policy.do_command({"command": "stop"})
        result = await executing
        assert result["state"] == stopped["state"] == "stopped"
        assert policy._arm.targets == [] and policy._gripper.targets == []
        assert policy._arm.stops and policy._gripper.stops
        await policy.close()
        await policy.close()

    asyncio.run(scenario())


def test_sdk_operation_cancellation_stops_inference():
    async def scenario():
        policy = FakePolicy(None, block=True)
        cancelled = asyncio.Event()
        operation = Operation("policy.execute", cancelled)
        executing = asyncio.create_task(
            policy.do_command({"command": "execute"}, viam_operation=operation)
        )
        while not policy.socket.sent:
            await asyncio.sleep(0)
        cancelled.set()
        result = await asyncio.wait_for(executing, timeout=0.5)
        assert result["state"] == "stopped"
        assert not policy._arm.targets and not policy._gripper.targets
        assert policy._arm.stops and policy._gripper.stops

    asyncio.run(scenario())


def test_slow_gripper_latch_causes_deadline_rejection_without_catchup():
    async def scenario():
        policy = FakePolicy(
            pack_message({"action": _actions()}), gripper=FakeGripper(delay=0.08)
        )
        policy._deadline_tolerance = 0.01
        result = await policy.do_command(
            {"command": "execute", "action_prefix_steps": 4}
        )
        assert result["state"] == "failed"
        assert "dispatch" in result["error"]
        assert policy._arm.stops and policy._gripper.stops
        assert len(policy._arm.targets) <= 1 and len(policy._gripper.targets) <= 1

    asyncio.run(scenario())


def test_per_run_deadline_tolerance_allows_slow_simulator_gripper():
    async def scenario():
        policy = FakePolicy(
            pack_message({"action": _actions()}), gripper=FakeGripper(delay=0.08)
        )

        result = await policy.do_command(
            {
                "command": "execute",
                "action_prefix_steps": 4,
                "deadline_tolerance_sec": 0.15,
            }
        )

        assert result["state"] == "completed"
        assert result["actions_executed"] == 4
        assert result["deadline_tolerance_sec"] == 0.15

    asyncio.run(scenario())


def test_full_validator_uses_measured_history_for_dynamic_limits():
    policy = FakePolicy(b"")
    history = [np.zeros(6), np.full(6, 0.03)]
    actions = _actions().astype(np.float64)
    actions[:, :6] = 0.10
    with pytest.raises(ValueError, match="acceleration"):
        policy._validate_actions(actions, history, dynamics=True)
