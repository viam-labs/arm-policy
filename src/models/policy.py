"""Contract-driven simulator bridge for Cosmos OpenPI policies."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)
from uuid import uuid4

import msgpack
import numpy as np
from PIL import Image
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    StringConstraints,
    WebsocketUrl,
    model_validator,
)
from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .trajectory_conditioner import (
    ALGORITHM,
    ALGORITHM_VERSION,
    TOPPRA_ALGORITHM,
    TOPPRA_ALGORITHM_VERSION,
    condition_trajectory,
    condition_trajectory_toppra,
    trajectory_metrics,
)

NAMESPACE = "awinter"
FAMILY = "arm-policy"


def get_attrs(config: ComponentConfig) -> dict[str, Any]:
    return struct_to_dict(config.attributes)


GRIPPER_PROJECTION_TOLERANCE = 0.05
ACTION_SHAPE = (32, 7)
CONTROL_HZ = 20.0
PERIOD_S = 1.0 / CONTROL_HZ
CAMERA_KEYS = (
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
)
CAMERA_ALIASES = {
    key: key.replace("observation.images.", "observation/", 1) for key in CAMERA_KEYS
}


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
PinnedControlHz = Annotated[float, Field(ge=20.0, le=20.0, allow_inf_nan=False)]
BinaryFloat = Annotated[
    float, Field(ge=0.0, le=1.0, multiple_of=1.0, allow_inf_nan=False)
]
Prompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Vector6 = tuple[
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
]


class Config(BaseModel):
    """Validated policy configuration, including its policy contract."""

    world: NonEmptyStr
    arm: NonEmptyStr
    gripper: NonEmptyStr
    fixed_camera: NonEmptyStr
    wrist_camera: NonEmptyStr
    server_uri: WebsocketUrl
    prompt: Prompt
    policy_contract_path: NonEmptyStr

    # Fixed by the checkpoint.
    action_chunk_size: Literal[32] = 32
    control_hz: PinnedControlHz = 20.0
    fixed_camera_width: Literal[1280] = 1280
    fixed_camera_height: Literal[720] = 720
    wrist_camera_width: Literal[640] = 640
    wrist_camera_height: Literal[480] = 480

    connect_timeout_sec: PositiveFiniteFloat = 10.0
    inference_timeout_sec: PositiveFiniteFloat = 120.0
    deadline_tolerance_sec: PositiveFiniteFloat = 0.025

    # Optional declarations may only repeat values selected by the contract.
    max_reach_m: FiniteFloat | None = None
    workspace_floor_m: FiniteFloat | None = None
    tcp_offset_m: FiniteFloat | None = None
    joint_min_radians: Vector6 | None = None
    joint_max_radians: Vector6 | None = None
    velocity_limits_rad_s: Vector6 | None = None
    acceleration_limits_rad_s2: Vector6 | None = None

    workspace_task_min_m: Vector3 | None = None
    workspace_task_max_m: Vector3 | None = None
    conditioner_algorithm: Literal["quintic", "toppra"] = "quintic"
    execution_backend: Literal["simulation", "physical_lite6"] = "simulation"
    physical_gripper_open_threshold: UnitFloat = 0.4
    physical_gripper_close_threshold: UnitFloat = 0.6
    physical_gripper_initial_state: BinaryFloat = 0.0
    max_chunks: Annotated[int, Field(ge=1, le=100)] = 1
    action_prefix_steps: Annotated[int, Field(ge=1, le=32)] | None = None
    action_prefix_duration_sec: PositiveFiniteFloat | None = None

    _contract: dict[str, Any] = PrivateAttr(default_factory=dict)
    _profile: SafetyProfile

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        contract, profile = _read_policy_contract(self.policy_contract_path)
        self._contract = contract
        self._profile = profile

        pinned_scalars = {
            "max_reach_m": profile.max_reach_m,
            "workspace_floor_m": profile.workspace_floor_m,
            "tcp_offset_m": profile.tcp_offset_m,
        }
        for key, expected in pinned_scalars.items():
            value = getattr(self, key)
            if value is not None and value != expected:
                raise ValueError(f"{key} is fixed by safety profile {profile.name}")

        pinned_vectors = {
            "joint_min_radians": profile.joint_min,
            "joint_max_radians": profile.joint_max,
            "velocity_limits_rad_s": profile.velocity_limits,
            "acceleration_limits_rad_s2": profile.acceleration_limits,
        }
        for key, expected in pinned_vectors.items():
            value = getattr(self, key)
            if value is not None and value != expected:
                raise ValueError(f"{key} is fixed by safety profile {profile.name}")

        if (self.workspace_task_min_m is None) != (self.workspace_task_max_m is None):
            raise ValueError(
                "workspace_task_min_m and workspace_task_max_m must be set together"
            )
        if (
            self.workspace_task_min_m is not None
            and self.workspace_task_max_m is not None
        ):
            if any(
                low >= high
                for low, high in zip(
                    self.workspace_task_min_m, self.workspace_task_max_m
                )
            ):
                raise ValueError("workspace task minimums must be below maximums")
        if self.execution_backend == "physical_lite6" and profile.arm_model != "lite6":
            raise ValueError(
                "physical_lite6 execution requires a Lite6 policy contract"
            )
        if not (
            self.physical_gripper_open_threshold < self.physical_gripper_close_threshold
        ):
            raise ValueError(
                "physical gripper thresholds must satisfy 0 <= open < close <= 1"
            )
        if (
            self.action_prefix_steps is not None
            and self.action_prefix_duration_sec is not None
        ):
            raise ValueError(
                "set only one of action_prefix_steps and action_prefix_duration_sec"
            )
        if self.action_prefix_duration_sec is not None:
            _prefix_steps_from_duration(self.action_prefix_duration_sec)
        return self

    @property
    def contract(self) -> dict[str, Any]:
        return self._contract

    @property
    def profile(self) -> SafetyProfile:
        return self._profile

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (
            self.world,
            self.arm,
            self.gripper,
            self.fixed_camera,
            self.wrist_camera,
        )

    @property
    def fixed_camera_size(self) -> tuple[int, int]:
        return self.fixed_camera_width, self.fixed_camera_height

    @property
    def wrist_camera_size(self) -> tuple[int, int]:
        return self.wrist_camera_width, self.wrist_camera_height

    @property
    def default_prefix_steps(self) -> int:
        if self.action_prefix_duration_sec is not None:
            return _prefix_steps_from_duration(self.action_prefix_duration_sec)
        return self.action_prefix_steps if self.action_prefix_steps is not None else 8

    @classmethod
    def _unconfigured(cls) -> Self:
        config = cls.model_construct(
            world="",
            arm="",
            gripper="",
            fixed_camera="",
            wrist_camera="",
            server_uri="",
            prompt="",
            policy_contract_path="",
        )
        config._contract = {}
        config._profile = SAFETY_PROFILES[("ur20", "robotiq_2f_85")]
        return config


@dataclass(frozen=True)
class SafetyProfile:
    name: str
    arm_model: str
    gripper_model: str
    joint_names: tuple[str, ...]
    joint_min: tuple[float, ...]
    joint_max: tuple[float, ...]
    velocity_limits: tuple[float, ...]
    acceleration_limits: tuple[float, ...]
    max_reach_m: float
    workspace_floor_m: float
    tcp_offset_m: float
    fk: Callable[[Sequence[float], float], tuple[float, float, float]]


def _dh_tcp_position(
    joints: Sequence[float],
    tcp_offset_m: float,
    *,
    a: Sequence[float],
    d: Sequence[float],
    alpha: Sequence[float],
    theta_offset: Sequence[float],
) -> tuple[float, float, float]:
    transform = np.eye(4, dtype=np.float64)
    for joint, offset, link_a, link_d, link_alpha in zip(
        joints, theta_offset, a, d, alpha
    ):
        theta = joint + offset
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(link_alpha), math.sin(link_alpha)
        transform = transform @ np.asarray(
            (
                (ct, -st * ca, st * sa, link_a * ct),
                (st, ct * ca, -ct * sa, link_a * st),
                (0.0, sa, ca, link_d),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
    point = transform @ np.asarray((0.0, 0.0, tcp_offset_m, 1.0))
    return float(point[0]), float(point[1]), float(point[2])


def ur20_tcp_position(
    joints: Sequence[float], tcp_offset_m: float
) -> tuple[float, float, float]:
    return _dh_tcp_position(
        joints,
        tcp_offset_m,
        a=(0.0, -0.8620, -0.7287, 0.0, 0.0, 0.0),
        d=(0.2363, 0.0, 0.0, 0.2010, 0.1593, 0.1543),
        alpha=(math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0),
        theta_offset=(0.0,) * 6,
    )


def lite6_tcp_position(
    joints: Sequence[float], tcp_offset_m: float
) -> tuple[float, float, float]:
    return _dh_tcp_position(
        joints,
        tcp_offset_m,
        a=(0.0, 0.200, 0.087, 0.0, 0.0, 0.0),
        d=(0.2433, 0.0, 0.0, 0.2276, 0.0, 0.0615),
        alpha=(-math.pi / 2, math.pi, math.pi / 2, math.pi / 2, -math.pi / 2, 0.0),
        theta_offset=(0.0, -math.pi / 2, -math.pi / 2, 0.0, 0.0, 0.0),
    )


UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
LITE6_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
SAFETY_PROFILES = {
    ("ur20", "robotiq_2f_85"): SafetyProfile(
        name="ur20-robotiq-sim-v1",
        arm_model="ur20",
        gripper_model="robotiq_2f_85",
        joint_names=UR_JOINT_NAMES,
        joint_min=(
            -2 * math.pi,
            -2 * math.pi,
            -math.pi,
            -2 * math.pi,
            -2 * math.pi,
            -2 * math.pi,
        ),
        joint_max=(
            2 * math.pi,
            2 * math.pi,
            math.pi,
            2 * math.pi,
            2 * math.pi,
            2 * math.pi,
        ),
        velocity_limits=(2.094395, 2.094395, 2.617994, 3.665191, 3.665191, 3.665191),
        acceleration_limits=(13.962634,) * 6,
        max_reach_m=1.70,
        workspace_floor_m=0.02,
        tcp_offset_m=0.134,
        fk=ur20_tcp_position,
    ),
    ("lite6", "lite6_gripper"): SafetyProfile(
        name="lite6-integrated-gripper-sim-v2",
        arm_model="lite6",
        gripper_model="lite6_gripper",
        joint_names=LITE6_JOINT_NAMES,
        joint_min=tuple(
            math.radians(value) for value in (-359, -149, -3, -359, -124, -359)
        ),
        joint_max=tuple(
            math.radians(value) for value in (359, 149, 299, 359, 124, 359)
        ),
        # Isaac 5.0 step-response measurement at 20 Hz (2026-09-17): every
        # joint achieved >=0.593 rad/s and >=8.242 rad/s² across three safe
        # poses. The articulation exposes drive gains/effort but no explicit
        # velocity or acceleration cap, so v2 stays below both measured minima.
        velocity_limits=(0.5,) * 6,
        acceleration_limits=(8.0,) * 6,
        max_reach_m=0.70,
        workspace_floor_m=0.02,
        tcp_offset_m=0.0836,
        fk=lite6_tcp_position,
    ),
}


def _read_policy_contract(path_value: Any) -> tuple[dict[str, Any], SafetyProfile]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("policy_contract_path must name a policy contract JSON file")
    path = Path(path_value).expanduser()
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read policy contract {path}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("schema_version") != 2:
        raise ValueError("policy contract must use schema_version 2")
    embodiment = contract.get("embodiment")
    if not isinstance(embodiment, dict):
        raise ValueError("policy contract embodiment must be an object")
    identity = (
        str(embodiment.get("arm_model", "")),
        str(embodiment.get("gripper_model", "")),
    )
    profile = SAFETY_PROFILES.get(identity)
    if profile is None:
        raise ValueError(f"no simulator safety profile for embodiment {identity!r}")
    joint_names = embodiment.get("joint_names")
    if joint_names != list(profile.joint_names):
        raise ValueError(
            "policy contract joint order does not match its safety profile"
        )
    expected_action = {
        "key": "action",
        "type": "absolute_joint_target",
        "shape": [7],
        "names": [*profile.joint_names, "gripper"],
        "units": [*(["rad"] * 6), "normalized"],
        "fps": 20,
        "gripper_convention": "0=open, 1=closed",
    }
    if contract.get("action_schema") != expected_action:
        raise ValueError("policy contract action schema does not match its embodiment")
    observation = contract.get("observation_schema")
    expected_observation = {
        "joint_names": list(profile.joint_names),
        "joint_units": "rad",
        "gripper_units": "normalized",
        "gripper_convention": "0=open, 1=closed",
        "camera_keys": list(CAMERA_KEYS),
        "camera_aliases": CAMERA_ALIASES,
    }
    if observation != expected_observation:
        raise ValueError(
            "policy contract observation schema does not match the Viam bridge"
        )
    framework = contract.get("framework")
    if not isinstance(framework, dict) or any(
        (
            framework.get("action_space") != "joint_pos",
            framework.get("joint_count") != 6,
            framework.get("use_state") is not True,
        )
    ):
        raise ValueError("policy contract framework is not six-joint stateful control")
    transform = contract.get("transform")
    if (
        not isinstance(transform, dict)
        or transform.get("camera_keys") != list(CAMERA_KEYS)
        or transform.get("camera_aliases") != CAMERA_ALIASES
    ):
        raise ValueError(
            "policy contract transform camera mapping does not match the Viam bridge"
        )
    if (
        contract.get("action_chunk_size") != ACTION_SHAPE[0]
        or contract.get("conditioning_fps") != CONTROL_HZ
    ):
        raise ValueError("policy contract must declare 32 steps at 20 Hz")
    return contract, profile


def _pack_array(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"unsupported MessagePack dtype {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: Any) -> Any:
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported MessagePack dtype {dtype}")
        shape = tuple(int(item) for item in value[b"shape"])
        expected = math.prod(shape) * dtype.itemsize
        data = value[b"data"]
        if expected != len(data):
            raise ValueError("ndarray byte length does not match its shape")
        return np.ndarray(buffer=data, dtype=dtype, shape=shape)
    if b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported MessagePack dtype {dtype}")
        return dtype.type(value[b"data"])
    return value


def pack_message(value: Any) -> bytes:
    return cast(bytes, msgpack.packb(value, default=_pack_array, use_bin_type=True))


def unpack_message(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_unpack_array, raw=False)


def _positive(attrs: Mapping[str, Any], key: str, default: float) -> float:
    value = float(attrs.get(key, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be positive and finite")
    return value


def _prefix_steps_from_duration(value: Any) -> int:
    duration = float(value)
    steps_float = duration / PERIOD_S
    steps = round(steps_float)
    if (
        not math.isfinite(duration)
        or not math.isclose(steps_float, steps, abs_tol=1e-9)
        or not 1 <= steps <= ACTION_SHAPE[0]
    ):
        raise ValueError(
            "action_prefix_duration_sec must be a 20 Hz duration in [0.05, 1.6]"
        )
    return steps


def _value_types(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _value_types(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value_types(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Policy(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "policy")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._world: Generic
        self._arm: Arm
        self._gripper: Gripper
        self._fixed_camera: Camera
        self._wrist_camera: Camera
        self._config = Config._unconfigured()
        self._physical_gripper_state = 0.0
        self._owner_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._execution_task: Optional[asyncio.Task[Mapping[str, ValueTypes]]] = None
        self._socket: Any = None
        self._generation = 0
        self._closed = False
        self._status: dict[str, Any] = self._idle_status()
        self._last_result: Optional[dict[str, Any]] = None

    def _idle_status(self) -> dict[str, Any]:
        return {
            "state": "idle",
            "run_id": "",
            "deadline_tolerance_sec": self._config.deadline_tolerance_sec,
            "prompt": "",
            "chunk_index": 0,
            "action_index": 0,
            "actions_executed": 0,
            "executed_prefixes": 0,
            "chunks_inferred": 0,
            "inference_responses": 0,
            "inference_ms": [],
            "server_timing": [],
            "raw_action_chunks": [],
            "conditioned_action_chunks": [],
            "conditioning": [],
            "progress": [],
            "actuation_timing": [],
            "server_metadata": {},
            "started_at": "",
            "finished_at": "",
            "last_error": "",
            "stop_errors": [],
            "embodiment": dict(self._config.contract.get("embodiment", {})),
            "safety_profile": self._config.profile.name,
            "conditioner_algorithm": self._config.conditioner_algorithm,
            "execution_backend": self._config.execution_backend,
        }

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        policy = cls(config.name)
        policy.reconfigure(config, dependencies)
        return policy

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        parsed = Config.model_validate(get_attrs(config))
        return parsed.dependencies, []

    @staticmethod
    def _dependency(
        dependencies: Mapping[ResourceName, ResourceBase],
        api: Any,
        name: str,
        kind: str,
    ) -> Any:
        resource = dependencies.get(api.get_resource_name(name))
        if resource is None or not isinstance(resource, api):
            raise ValueError(f"missing typed {kind} dependency {name!r}")
        return resource

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        parsed = Config.model_validate(get_attrs(config))
        self._world = self._dependency(dependencies, Generic, parsed.world, "world")
        self._arm = self._dependency(dependencies, Arm, parsed.arm, "arm")
        self._gripper = self._dependency(
            dependencies, Gripper, parsed.gripper, "gripper"
        )
        self._fixed_camera = self._dependency(
            dependencies, Camera, parsed.fixed_camera, "fixed camera"
        )
        self._wrist_camera = self._dependency(
            dependencies, Camera, parsed.wrist_camera, "wrist camera"
        )
        self._config = parsed
        self._physical_gripper_state = parsed.physical_gripper_initial_state
        self._status = self._idle_status()

    @staticmethod
    def _decode_image(image: Any, expected: tuple[int, int], label: str) -> np.ndarray:
        mime = str(getattr(image, "mime_type", ""))
        if mime and not mime.startswith("image/"):
            raise ValueError(f"{label} returned unsupported MIME type {mime!r}")
        try:
            decoded = Image.open(BytesIO(image.data)).convert("RGB")
            decoded.load()
        except Exception as exc:
            raise ValueError(f"{label} returned an invalid encoded image") from exc
        if decoded.size != expected:
            raise ValueError(f"{label} image is {decoded.size}, expected {expected}")
        return np.ascontiguousarray(np.asarray(decoded, dtype=np.uint8))

    async def _joint_radians(self) -> list[float]:
        positions = await self._arm.get_joint_positions()
        return [math.radians(float(value)) for value in positions.values]

    async def _gripper_input(self) -> list[float]:
        if self._config.execution_backend == "physical_lite6":
            return [self._physical_gripper_state]
        return [float(value) for value in await self._gripper.get_current_inputs()]

    async def _observation(self, prompt: str) -> dict[str, Any]:
        fixed_result, wrist_result, joint_values, gripper_inputs = await asyncio.gather(
            self._fixed_camera.get_images(),
            self._wrist_camera.get_images(),
            self._joint_radians(),
            self._gripper_input(),
        )
        fixed_images, _ = fixed_result
        wrist_images, _ = wrist_result
        if len(fixed_images) != 1 or len(wrist_images) != 1:
            raise ValueError("policy cameras must each return exactly one image")
        joints = np.asarray(joint_values, dtype=np.float32)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("arm returned invalid six-joint state")
        gripper = np.asarray(gripper_inputs, dtype=np.float32)
        if (
            gripper.shape != (1,)
            or not np.isfinite(gripper).all()
            or not 0.0 <= float(gripper[0]) <= 1.0
        ):
            raise ValueError("gripper returned invalid normalized state")
        fixed = self._decode_image(
            fixed_images[0], self._config.fixed_camera_size, "fixed camera"
        )
        wrist = self._decode_image(
            wrist_images[0], self._config.wrist_camera_size, "wrist camera"
        )
        return {
            "prompt": prompt,
            "observation/wrist_image_left": wrist,
            "observation/exterior_image_1_left": fixed,
            "observation/exterior_image_2_left": fixed,
            "observation/joint_position": joints,
            "observation/gripper_position": gripper,
        }

    async def _cancelled(self, operation: Any, generation: int) -> bool:
        return (
            self._stop_event.is_set()
            or generation != self._generation
            or await operation.is_cancelled()
        )

    async def _wait_cancellable(
        self,
        awaitable: Any,
        timeout: float,
        operation: Any,
        generation: int,
        label: str,
    ) -> Any:
        task = asyncio.ensure_future(awaitable)
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{label} timed out")
                done, _ = await asyncio.wait((task,), timeout=min(PERIOD_S, remaining))
                if done:
                    return task.result()
                if await self._cancelled(operation, generation):
                    raise asyncio.CancelledError
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _recv(
        self, socket: Any, timeout: float, operation: Any, generation: int
    ) -> Any:
        receive = asyncio.create_task(socket.recv())
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("policy server receive timed out")
                done, _ = await asyncio.wait(
                    (receive,), timeout=min(PERIOD_S, remaining)
                )
                if done:
                    return receive.result()
                if await self._cancelled(operation, generation):
                    raise asyncio.CancelledError
        finally:
            if not receive.done():
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)

    async def _connect(
        self, operation: Any, generation: int
    ) -> tuple[Any, Mapping[str, Any]]:
        import websockets

        socket = await asyncio.wait_for(
            websockets.connect(
                str(self._config.server_uri),
                ping_interval=None,
                compression=None,
                max_size=64 * 1024 * 1024,
                open_timeout=self._config.connect_timeout_sec,
            ),
            timeout=self._config.connect_timeout_sec,
        )
        self._socket = socket
        try:
            raw = await self._recv(
                socket, self._config.connect_timeout_sec, operation, generation
            )
            if not isinstance(raw, bytes):
                raise RuntimeError(f"policy server metadata was text: {raw}")
            metadata = unpack_message(raw)
            if not isinstance(metadata, Mapping):
                raise ValueError("policy server metadata must be an object")
            return socket, metadata
        except BaseException:
            await socket.close()
            self._socket = None
            raise

    def _parse_chunk(self, response: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
        if not isinstance(response, Mapping):
            raise ValueError("policy response must be an object")
        actions = response.get("action")
        if not isinstance(actions, np.ndarray) or actions.shape != ACTION_SHAPE:
            shape = getattr(actions, "shape", None)
            raise ValueError(
                "policy action must be a NumPy array with shape "
                f"{ACTION_SHAPE}, got {shape}"
            )
        if actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
            raise ValueError("policy action must contain only finite numeric values")
        timing = response.get("server_timing", {})
        if not isinstance(timing, Mapping):
            raise ValueError("server_timing must be an object")
        return np.asarray(actions, dtype=np.float64), dict(timing)

    def _validate_joint_positions(self, actions: np.ndarray) -> None:
        profile = self._config.profile
        for index, action in enumerate(actions):
            if any(
                value < low or value > high
                for value, low, high in zip(
                    action[:6], profile.joint_min, profile.joint_max
                )
            ):
                raise ValueError(
                    f"action {index} exceeds {profile.arm_model} joint position limits"
                )

    def _validate_position_and_gripper(self, actions: np.ndarray) -> None:
        self._validate_joint_positions(actions)
        for index, action in enumerate(actions):
            if not 0.0 <= float(action[6]) <= 1.0:
                raise ValueError(f"action {index} gripper is outside [0, 1]")

    def _validate_actions(
        self,
        actions: np.ndarray,
        history: Sequence[np.ndarray],
        *,
        dynamics: bool,
    ) -> None:
        previous_previous, previous = history[-2], history[-1]
        profile = self._config.profile
        velocity_limits = np.asarray(profile.velocity_limits)
        acceleration_limits = np.asarray(profile.acceleration_limits)
        self._validate_position_and_gripper(actions)
        for index, action in enumerate(actions):
            joints = action[:6]
            if dynamics:
                velocity = np.abs(joints - previous) / PERIOD_S
                if np.any(velocity > velocity_limits + 1e-9):
                    raise ValueError(
                        f"action {index} exceeds {profile.arm_model} joint velocity limits"
                    )
                acceleration = np.abs(joints - 2.0 * previous + previous_previous) / (
                    PERIOD_S * PERIOD_S
                )
                if np.any(acceleration > acceleration_limits + 1e-9):
                    raise ValueError(
                        f"action {index} exceeds {profile.arm_model} joint "
                        "acceleration limits"
                    )
            x, y, z = profile.fk(joints, profile.tcp_offset_m)
            if math.sqrt(x * x + y * y + z * z) > profile.max_reach_m + 1e-9:
                raise ValueError(f"action {index} TCP exceeds {profile.name} max reach")
            if z < profile.workspace_floor_m - 1e-9:
                raise ValueError(
                    f"action {index} TCP is below {profile.name} workspace floor"
                )
            if (
                self._config.workspace_task_min_m is not None
                and self._config.workspace_task_max_m is not None
                and any(
                    value < low or value > high
                    for value, low, high in zip(
                        (x, y, z),
                        self._config.workspace_task_min_m,
                        self._config.workspace_task_max_m,
                    )
                )
            ):
                raise ValueError(
                    f"action {index} TCP is outside the configured task workspace"
                )
            previous_previous, previous = previous, joints

    def _workspace_metrics(self, actions: np.ndarray) -> Mapping[str, Any]:
        positions = np.asarray(
            [
                self._config.profile.fk(action[:6], self._config.profile.tcp_offset_m)
                for action in actions
            ],
            dtype=np.float64,
        )
        reach = np.linalg.norm(positions, axis=1)
        return {
            "tcp_min_m": np.min(positions, axis=0).tolist(),
            "tcp_max_m": np.max(positions, axis=0).tolist(),
            "max_reach_m": float(np.max(reach)),
            "minimum_floor_margin_m": float(
                np.min(positions[:, 2]) - self._config.profile.workspace_floor_m
            ),
        }

    def _progress_record(
        self,
        chunk_index: int,
        phase: str,
        joints: Any,
        gripper: float,
        prop_result: Optional[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        joint_values = np.asarray(joints, dtype=np.float64)
        if joint_values.shape != (6,) or not np.isfinite(joint_values).all():
            raise ValueError("progress capture received invalid joint state")
        tcp = np.asarray(
            self._config.profile.fk(
                joint_values.tolist(), self._config.profile.tcp_offset_m
            ),
            dtype=np.float64,
        )
        record: dict[str, Any] = {
            "chunk_index": chunk_index,
            "phase": phase,
            "joint_position_rad": joint_values.tolist(),
            "gripper_position": float(gripper),
            "tcp_position_m": tcp.tolist(),
        }
        if prop_result is not None:
            pose = prop_result.get("pose")
            if not isinstance(pose, Mapping):
                raise ValueError("progress prop query returned no pose")
            prop_position = np.asarray(pose.get("position"), dtype=np.float64)
            if prop_position.shape != (3,) or not np.isfinite(prop_position).all():
                raise ValueError("progress prop query returned invalid position")
            record["prop_position_m"] = prop_position.tolist()
            record["tcp_to_prop_distance_m"] = float(
                np.linalg.norm(tcp - prop_position)
            )
        return record

    def _condition_chunk(
        self,
        response: Any,
        history: Sequence[np.ndarray],
        prefix: int,
        conditioner_algorithm: str,
    ) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
        raw_actions, timing = self._parse_chunk(response)
        algorithm, version = (
            (TOPPRA_ALGORITHM, TOPPRA_ALGORITHM_VERSION)
            if conditioner_algorithm == "toppra"
            else (ALGORITHM, ALGORITHM_VERSION)
        )
        diagnostics: dict[str, Any] = {
            "algorithm": algorithm,
            "version": version,
            "control_hz": CONTROL_HZ,
            "raw_steps": int(raw_actions.shape[0]),
            "raw_metrics": {
                **trajectory_metrics(raw_actions, history, PERIOD_S),
                "workspace": self._workspace_metrics(raw_actions),
            },
            "selected_profile": self._config.profile.name,
            "position_limits_rad": {
                "minimum": list(self._config.profile.joint_min),
                "maximum": list(self._config.profile.joint_max),
            },
            "velocity_limits_rad_s": list(self._config.profile.velocity_limits),
            "acceleration_limits_rad_s2": list(
                self._config.profile.acceleration_limits
            ),
            "selected_prefix_steps": prefix,
            "selected_prefix_duration_s": prefix * PERIOD_S,
            "observation_age_at_prefix_end_s": prefix * PERIOD_S,
            "rejection_reason": "",
        }
        self._status["conditioning"][-1] = _value_types(diagnostics)
        self._validate_joint_positions(raw_actions)
        try:
            conditioner = (
                condition_trajectory_toppra
                if conditioner_algorithm == "toppra"
                else condition_trajectory
            )
            actions, raw_diagnostics = conditioner(
                raw_actions, history, self._config.profile, period_s=PERIOD_S
            )
        except ValueError as exc:
            if conditioner_algorithm != "toppra":
                raise
            actions, raw_diagnostics = condition_trajectory(
                raw_actions, history, self._config.profile, period_s=PERIOD_S
            )
            raw_diagnostics = {
                **dict(raw_diagnostics),
                "requested_algorithm": TOPPRA_ALGORITHM,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            }
        clipped_gripper = np.clip(actions[:, 6], 0.0, 1.0)
        gripper_delta = np.abs(clipped_gripper - actions[:, 6])
        max_gripper_delta = float(np.max(gripper_delta))
        if max_gripper_delta <= GRIPPER_PROJECTION_TOLERANCE:
            actions = actions.copy()
            actions[:, 6] = clipped_gripper
            raw_diagnostics = {
                **dict(raw_diagnostics),
                "gripper_projection_max": max_gripper_delta,
                "gripper_projection_tolerance": GRIPPER_PROJECTION_TOLERANCE,
            }
        diagnostics.update(raw_diagnostics)
        diagnostics["selected_profile"] = self._config.profile.name
        diagnostics["raw_metrics"] = {
            **dict(diagnostics["raw_metrics"]),
            "workspace": self._workspace_metrics(raw_actions),
        }
        diagnostics["conditioned_metrics"] = {
            **dict(diagnostics["conditioned_metrics"]),
            "workspace": self._workspace_metrics(actions),
        }
        diagnostics["selected_prefix_steps"] = prefix
        diagnostics["selected_prefix_duration_s"] = prefix * PERIOD_S
        diagnostics["validation_scope"] = "selected_conditioned_prefix"
        try:
            self._validate_actions(actions, history, dynamics=True)
        except ValueError as exc:
            diagnostics["deferred_horizon_rejection_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
        else:
            diagnostics["deferred_horizon_rejection_reason"] = ""
        validated_prefix = prefix
        try:
            self._validate_actions(actions[:prefix], history, dynamics=True)
        except ValueError as exc:
            validated_prefix = 0
            for candidate in range(1, prefix + 1):
                try:
                    self._validate_actions(actions[:candidate], history, dynamics=True)
                except ValueError:
                    break
                validated_prefix = candidate
            if validated_prefix == 0:
                raise
            diagnostics["prefix_truncation_reason"] = f"{type(exc).__name__}: {exc}"
        else:
            diagnostics["prefix_truncation_reason"] = ""
        diagnostics["validated_prefix_steps"] = validated_prefix
        diagnostics["validated_prefix_duration_s"] = validated_prefix * PERIOD_S
        diagnostics["observation_age_at_prefix_end_s"] = validated_prefix * PERIOD_S
        diagnostics["rejection_reason"] = ""
        self._status["conditioned_action_chunks"].append(actions.tolist())
        self._status["conditioning"][-1] = _value_types(diagnostics)
        return actions, timing, diagnostics

    async def _stream_arm_prefix(
        self, actions: np.ndarray, current_joints: Sequence[float]
    ) -> None:
        async def batches():
            trajectory_point = cast(Any, Arm).TrajectoryPoint
            points = [
                trajectory_point(
                    time=timedelta(0),
                    positions=[math.degrees(float(value)) for value in current_joints],
                )
            ]
            points.extend(
                trajectory_point(
                    time=timedelta(seconds=(index + 1) * PERIOD_S),
                    positions=[math.degrees(float(value)) for value in action[:6]],
                )
                for index, action in enumerate(actions)
            )
            yield points

        stream = cast(Any, self._arm).move_through_joint_positions_streamed(batches())
        async for _ in stream:
            pass

    async def _dispatch_simulated_gripper(
        self,
        actions: np.ndarray,
        prefix_started: float,
        operation: Any,
        generation: int,
        deadline_tolerance: float,
    ) -> list[float]:
        lateness_samples: list[float] = []
        for action_index, action in enumerate(actions):
            deadline = prefix_started + action_index * PERIOD_S
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            if await self._cancelled(operation, generation):
                raise asyncio.CancelledError
            lateness = time.monotonic() - deadline
            if lateness > deadline_tolerance:
                raise TimeoutError(
                    f"action {action_index} missed its deadline by {lateness:.6f}s"
                )
            lateness_samples.append(lateness)
            await self._wait_cancellable(
                self._gripper.do_command(
                    {"command": "set_input", "value": float(action[6])}
                ),
                PERIOD_S + deadline_tolerance,
                operation,
                generation,
                "gripper target dispatch",
            )
            self._status["action_index"] = action_index + 1
            self._status["actions_executed"] += 1
        return lateness_samples

    async def _dispatch_physical_gripper(
        self,
        actions: np.ndarray,
        prefix_started: float,
        operation: Any,
        generation: int,
        deadline_tolerance: float,
    ) -> list[float]:
        lateness_samples: list[float] = []
        for action_index, action in enumerate(actions):
            deadline = prefix_started + action_index * PERIOD_S
            delay = deadline - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            if await self._cancelled(operation, generation):
                raise asyncio.CancelledError
            lateness = time.monotonic() - deadline
            if lateness > deadline_tolerance:
                raise TimeoutError(
                    f"action {action_index} missed its deadline by {lateness:.6f}s"
                )
            lateness_samples.append(lateness)
            value = float(action[6])
            desired = self._physical_gripper_state
            if value <= self._config.physical_gripper_open_threshold:
                desired = 0.0
            elif value >= self._config.physical_gripper_close_threshold:
                desired = 1.0
            if desired != self._physical_gripper_state:
                if desired == 0.0:
                    await self._gripper.open()
                else:
                    await self._gripper.grab()
                self._physical_gripper_state = desired
            self._status["action_index"] = action_index + 1
            self._status["actions_executed"] += 1
        return lateness_samples

    async def _dispatch_prefix(
        self,
        actions: np.ndarray,
        current_joints: Sequence[float],
        operation: Any,
        generation: int,
        deadline_tolerance: float,
    ) -> dict[str, Any]:
        prefix_started = time.monotonic()
        gripper_dispatch = (
            self._dispatch_physical_gripper(
                actions,
                prefix_started,
                operation,
                generation,
                deadline_tolerance,
            )
            if self._config.execution_backend == "physical_lite6"
            else self._dispatch_simulated_gripper(
                actions,
                prefix_started,
                operation,
                generation,
                deadline_tolerance,
            )
        )
        _, lateness_samples = await self._wait_cancellable(
            asyncio.gather(
                self._stream_arm_prefix(actions, current_joints), gripper_dispatch
            ),
            len(actions) * PERIOD_S + self._config.connect_timeout_sec,
            operation,
            generation,
            "trajectory dispatch",
        )
        return {
            "backend": self._config.execution_backend,
            "scheduled_steps": len(actions),
            "elapsed_sec": time.monotonic() - prefix_started,
            "max_dispatch_lateness_sec": max(lateness_samples, default=0.0),
        }

    async def _safe_stop(self) -> list[str]:
        results = await asyncio.gather(
            self._arm.stop(), self._gripper.stop(), return_exceptions=True
        )
        return [
            f"{type(value).__name__}: {value}"
            for value in results
            if isinstance(value, BaseException)
        ]

    async def _close_socket(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                await socket.close()
            except Exception:
                pass

    async def _execute_run(
        self,
        prompt: str,
        max_chunks: int,
        prefix: int,
        conditioner_algorithm: str,
        deadline_tolerance: float,
        progress_prop: str,
        operation: Any,
        generation: int,
    ) -> Mapping[str, ValueTypes]:
        terminal_state = "completed"
        error = ""
        try:
            # The configured backend selects simulator target latching or the
            # Viam timestamped trajectory API used by physical arms.
            self._status["state"] = "connecting"
            socket, metadata = await self._connect(operation, generation)
            self._status["server_metadata"] = _value_types(metadata)
            for chunk_index in range(max_chunks):
                if await self._cancelled(operation, generation):
                    raise asyncio.CancelledError
                self._status.update(
                    state="inferring", chunk_index=chunk_index, action_index=0
                )
                older_values = await self._wait_cancellable(
                    self._joint_radians(),
                    self._config.connect_timeout_sec,
                    operation,
                    generation,
                    "first measured joint sample",
                )
                older = np.asarray(older_values, dtype=np.float64)
                if older.shape != (6,) or not np.isfinite(older).all():
                    raise ValueError("arm returned invalid first measured joint sample")
                await self._wait_cancellable(
                    asyncio.sleep(PERIOD_S),
                    PERIOD_S + deadline_tolerance,
                    operation,
                    generation,
                    "joint history sampling",
                )
                observation = await self._wait_cancellable(
                    self._observation(prompt),
                    self._config.inference_timeout_sec,
                    operation,
                    generation,
                    "observation capture",
                )
                current = np.asarray(
                    observation["observation/joint_position"], dtype=np.float64
                )
                history = [older, current]
                prop_before = None
                if progress_prop:
                    prop_before = await self._wait_cancellable(
                        self._world.do_command(
                            {"command": "get_prop_pose", "prop": progress_prop}
                        ),
                        self._config.connect_timeout_sec,
                        operation,
                        generation,
                        "progress prop query",
                    )
                self._status["progress"].append(
                    self._progress_record(
                        chunk_index,
                        "before_prefix",
                        current,
                        float(observation["observation/gripper_position"][0]),
                        prop_before,
                    )
                )
                if await self._cancelled(operation, generation):
                    raise asyncio.CancelledError
                started = time.monotonic()
                await self._wait_cancellable(
                    socket.send(pack_message(observation)),
                    self._config.inference_timeout_sec,
                    operation,
                    generation,
                    "policy request send",
                )
                raw = await self._recv(
                    socket, self._config.inference_timeout_sec, operation, generation
                )
                inference_ms = (time.monotonic() - started) * 1000.0
                if await self._cancelled(operation, generation):
                    raise asyncio.CancelledError
                if not isinstance(raw, bytes):
                    raise RuntimeError(f"policy server returned text error: {raw}")
                response = unpack_message(raw)
                self._status["inference_responses"] += 1
                self._status["inference_ms"].append(inference_ms)
                if isinstance(response, Mapping):
                    raw_actions = response.get("action")
                    if (
                        isinstance(raw_actions, np.ndarray)
                        and raw_actions.shape == ACTION_SHAPE
                        and np.isfinite(raw_actions).all()
                    ):
                        self._status["raw_action_chunks"].append(raw_actions.tolist())
                        self._status["conditioning"].append(
                            {
                                "algorithm": (
                                    TOPPRA_ALGORITHM
                                    if conditioner_algorithm == "toppra"
                                    else ALGORITHM
                                ),
                                "version": (
                                    TOPPRA_ALGORITHM_VERSION
                                    if conditioner_algorithm == "toppra"
                                    else ALGORITHM_VERSION
                                ),
                                "selected_profile": self._config.profile.name,
                                "rejection_reason": "",
                            }
                        )
                    raw_timing = response.get("server_timing")
                    if isinstance(raw_timing, Mapping):
                        self._status["server_timing"].append(_value_types(raw_timing))
                actions, _, diagnostics = self._condition_chunk(
                    response, history, prefix, conditioner_algorithm
                )
                execution_prefix = int(diagnostics["validated_prefix_steps"])
                self._status["chunks_inferred"] += 1
                self._status["state"] = "executing"
                timing = await self._dispatch_prefix(
                    actions[:execution_prefix],
                    current.tolist(),
                    operation,
                    generation,
                    deadline_tolerance,
                )
                self._status["actuation_timing"].append(timing)
                self._status["executed_prefixes"] += 1
                after_arm, after_gripper = await self._wait_cancellable(
                    asyncio.gather(self._joint_radians(), self._gripper_input()),
                    self._config.connect_timeout_sec,
                    operation,
                    generation,
                    "post-prefix progress capture",
                )
                prop_after = None
                if progress_prop:
                    prop_after = await self._wait_cancellable(
                        self._world.do_command(
                            {"command": "get_prop_pose", "prop": progress_prop}
                        ),
                        self._config.connect_timeout_sec,
                        operation,
                        generation,
                        "post-prefix prop query",
                    )
                self._status["progress"].append(
                    self._progress_record(
                        chunk_index,
                        "after_prefix",
                        cast(Any, after_arm),
                        float(cast(Any, after_gripper)[0]),
                        prop_after,
                    )
                )
            return self._finish_result("completed", "")
        except asyncio.CancelledError:
            terminal_state = "stopped"
            error = "execution cancelled"
            return self._finish_result(terminal_state, error)
        except Exception as exc:
            if self._stop_event.is_set() or generation != self._generation:
                terminal_state = "stopped"
                error = "execution cancelled"
            else:
                terminal_state = "failed"
                error = f"{type(exc).__name__}: {exc}"
            if self._status["conditioning"]:
                latest = self._status["conditioning"][-1]
                if not latest.get("rejection_reason"):
                    latest["rejection_reason"] = error
            return self._finish_result(terminal_state, error)
        finally:
            self._generation += 1
            await self._close_socket()
            stop_errors = await self._safe_stop()
            self._status["stop_errors"] = stop_errors
            if stop_errors:
                stop_error = "safe stop failed: " + "; ".join(stop_errors)
                self._status["state"] = "failed"
                self._status["last_error"] = stop_error
                if self._last_result is not None:
                    self._last_result.update(state="failed", error=stop_error)
            self._status["finished_at"] = _utc_now()
            if self._last_result is not None:
                self._last_result.update(
                    stop_errors=stop_errors, finished_at=self._status["finished_at"]
                )

    def _finish_result(self, state: str, error: str) -> dict[str, Any]:
        self._status["state"] = state
        self._status["last_error"] = error
        result = {
            "run_id": self._status["run_id"],
            "state": state,
            "prompt": self._status["prompt"],
            "chunks_inferred": self._status["chunks_inferred"],
            "inference_responses": self._status["inference_responses"],
            "actions_executed": self._status["actions_executed"],
            "executed_prefixes": self._status["executed_prefixes"],
            "inference_ms": list(self._status["inference_ms"]),
            "server_timing": list(self._status["server_timing"]),
            "raw_action_chunks": list(self._status["raw_action_chunks"]),
            "conditioned_action_chunks": list(
                self._status["conditioned_action_chunks"]
            ),
            "conditioning": list(self._status["conditioning"]),
            "progress": list(self._status["progress"]),
            "actuation_timing": list(self._status["actuation_timing"]),
            "embodiment": dict(self._status["embodiment"]),
            "safety_profile": self._status["safety_profile"],
            "conditioner_algorithm": self._status["conditioner_algorithm"],
            "deadline_tolerance_sec": self._status["deadline_tolerance_sec"],
            "execution_backend": self._config.execution_backend,
            "started_at": self._status["started_at"],
            "finished_at": _utc_now(),
            "error": error,
            "stop_errors": [],
        }
        self._last_result = result
        return result

    async def _execute_command(
        self, command: Mapping[str, ValueTypes], kwargs: Mapping[str, Any]
    ) -> Mapping[str, ValueTypes]:
        if self._closed:
            raise RuntimeError(f"policy {self.name} is closed")
        prompt = str(command.get("prompt", self._config.prompt)).strip()
        if not prompt:
            raise ValueError("execute prompt must be non-empty")
        max_chunks = int(cast(Any, command.get("max_chunks", self._config.max_chunks)))
        if "action_prefix_steps" in command and "action_prefix_duration_sec" in command:
            raise ValueError(
                "set only one of action_prefix_steps and action_prefix_duration_sec"
            )
        prefix = (
            _prefix_steps_from_duration(command["action_prefix_duration_sec"])
            if "action_prefix_duration_sec" in command
            else int(
                cast(
                    Any,
                    command.get(
                        "action_prefix_steps", self._config.default_prefix_steps
                    ),
                )
            )
        )
        conditioner_algorithm = str(
            command.get("conditioner_algorithm", self._config.conditioner_algorithm)
        )
        if conditioner_algorithm not in ("quintic", "toppra"):
            raise ValueError("conditioner_algorithm must be 'quintic' or 'toppra'")
        deadline_tolerance = _positive(
            command,
            "deadline_tolerance_sec",
            self._config.deadline_tolerance_sec,
        )
        progress_prop = str(command.get("progress_prop", "")).strip()
        if not 1 <= max_chunks <= 100:
            raise ValueError("max_chunks must be in [1, 100]")
        if not 1 <= prefix <= 32:
            raise ValueError("action_prefix_steps must be in [1, 32]")
        async with self._owner_lock:
            if self._execution_task is not None and not self._execution_task.done():
                raise RuntimeError("policy execution is already in progress")
            self._stop_event = asyncio.Event()
            self._generation += 1
            generation = self._generation
            self._last_result = None
            self._status = self._idle_status()
            self._status.update(
                state="connecting",
                run_id=str(uuid4()),
                prompt=prompt,
                started_at=_utc_now(),
                conditioner_algorithm=conditioner_algorithm,
                deadline_tolerance_sec=deadline_tolerance,
            )
            operation = self.get_operation(kwargs)
            task = asyncio.create_task(
                self._execute_run(
                    prompt,
                    max_chunks,
                    prefix,
                    conditioner_algorithm,
                    deadline_tolerance,
                    progress_prop,
                    operation,
                    generation,
                )
            )
            self._execution_task = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._stop_event.set()
            self._generation += 1
            await self._close_socket()
            await asyncio.shield(task)
            raise

    async def _stop_command(self) -> Mapping[str, ValueTypes]:
        self._stop_event.set()
        self._generation += 1
        task = self._execution_task
        if task is not None and not task.done():
            self._status["state"] = "stopping"
        await self._close_socket()
        immediate_errors = await self._safe_stop()
        if task is not None and not task.done() and task is not asyncio.current_task():
            await asyncio.shield(task)
        result = dict(self._last_result or self._status)
        if immediate_errors:
            result["stop_errors"] = (
                list(result.get("stop_errors", [])) + immediate_errors
            )
        return result

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = str(command.get("command", ""))
        if cmd == "execute":
            return await self._execute_command(command, kwargs)
        if cmd == "stop":
            return await self._stop_command()
        if cmd == "status":
            return dict(self._status)
        raise ValueError("unknown command; supported: execute, stop, status")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._stop_command()
