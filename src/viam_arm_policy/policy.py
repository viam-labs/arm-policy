"""Contract-driven simulator bridge for Cosmos OpenPI policies."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass, field
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
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
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

from .model_export import (
    default_data_root,
    validate_data_root,
    validate_model_path,
    validate_runtime_manifest,
)
from .policy_server import PolicyServerSupervisor, runtime_command
from .replay_source import ReplayChunkSource, ReplaySpec
from .trajectory_conditioner import (
    ALGORITHM,
    ALGORITHM_VERSION,
    TOPPRA_ALGORITHM,
    TOPPRA_ALGORITHM_VERSION,
    condition_trajectory,
    condition_trajectory_toppra,
    trajectory_metrics,
)

NAMESPACE = "viam-labs"
FAMILY = "arm-policy"


def get_attrs(config: ComponentConfig) -> dict[str, Any]:
    return struct_to_dict(config.attributes)


ACTION_SHAPE = (32, 7)
CAMERA_KEYS = (
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
)
CAMERA_ALIASES = {
    key: key.replace("observation.images.", "observation/", 1) for key in CAMERA_KEYS
}
# Fixed camera sizes the policy server accepts (same 16:9 view, full or half size).
FIXED_CAMERA_SIZES = ((1280, 720), (640, 360))


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
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


class ContractEmbodiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm_model: NonEmptyStr
    gripper_model: NonEmptyStr
    gripper_mode: Literal["binary", "position"]
    joint_names: tuple[str, ...]


class ContractActionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Literal["action"]
    type: Literal["absolute_joint_target"]
    shape: tuple[Literal[7]]
    names: tuple[str, ...]
    units: tuple[str, ...]
    fps: PositiveFiniteFloat
    gripper_convention: Literal["0=open, 1=closed"]


class ContractObservationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    joint_names: tuple[str, ...]
    joint_units: Literal["rad"]
    gripper_units: Literal["binary", "normalized"]
    gripper_convention: Literal["0=open, 1=closed"]
    camera_keys: tuple[str, ...]
    camera_aliases: dict[str, str]


class ContractFramework(BaseModel):
    model_config = ConfigDict(extra="allow")

    action_space: Literal["joint_pos"]
    joint_count: Literal[6]
    use_state: Literal[True]


class ContractTransform(BaseModel):
    model_config = ConfigDict(extra="allow")

    camera_keys: tuple[str, ...]
    camera_aliases: dict[str, str]


class PolicyContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[3]
    embodiment: ContractEmbodiment
    action_schema: ContractActionSchema
    observation_schema: ContractObservationSchema
    framework: ContractFramework
    transform: ContractTransform
    action_chunk_size: Literal[32]
    conditioning_fps: PositiveFiniteFloat
    domain_name: NonEmptyStr

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        if self.action_schema.fps != self.conditioning_fps:
            raise ValueError("action_schema.fps must equal conditioning_fps")
        return self


class Config(BaseModel):
    """Validated policy configuration, including its policy contract."""

    model_config = ConfigDict(extra="ignore")

    world: NonEmptyStr | None = None
    arm: NonEmptyStr
    gripper: NonEmptyStr
    fixed_camera: NonEmptyStr
    fixed_camera_2: str = ""
    wrist_camera: NonEmptyStr
    server_uri: WebsocketUrl | None = None
    prompt: Prompt | None = None
    policy_contract_path: NonEmptyStr | None = None

    # Fixed by the checkpoint.
    action_chunk_size: Literal[32] = 32
    # Training shrinks each fixed view to a quarter of the wrist image, so a
    # 640x360 render (half size, same aspect) loses nothing and renders faster.
    # Unset accepts either supported size; set both to require one.
    fixed_camera_width: Literal[1280, 640] | None = None
    fixed_camera_height: Literal[720, 360] | None = None
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
    conditioner_algorithm: Literal["quintic", "toppra"] = "toppra"
    binary_gripper_open_threshold: UnitFloat = 0.4
    binary_gripper_close_threshold: UnitFloat = 0.6
    binary_gripper_initial_state: BinaryFloat = 0.0
    binary_gripper_settle_sec: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 3.0
    # The upstream Cosmos policy server inverts the gripper on the way in
    # (observation/gripper_position) and on the way out (action[:, -1]) for
    # DROID variants recorded with the opposite convention. Datasets trained
    # without that flip (e.g. droid_lerobot_20260115_no_noops) need both undone.
    undo_server_gripper_flip: bool = False
    max_chunks: Annotated[int, Field(ge=1, le=100)] = 1
    action_prefix_steps: Annotated[int, Field(ge=1, le=32)] | None = None
    action_prefix_duration_sec: PositiveFiniteFloat | None = None

    # A portable export deployed by viam-server from the Viam registry, normally
    # configured as "${packages.ml_model.<package-name>}".
    model_path: str | None = None
    # Holds the managed policy server's runtime state and log.
    model_data_root: str | None = None
    policy_server_host: str = "127.0.0.1"
    policy_server_port: Annotated[int, Field(ge=1, le=65535)] = 8000
    policy_server_readiness_timeout_sec: PositiveFiniteFloat = 900.0
    policy_server_autoclose: bool = False

    _contract: PolicyContract
    _profile: SafetyProfile

    @model_validator(mode="after")
    def validate_static(self) -> Self:
        if (self.fixed_camera_width, self.fixed_camera_height) not in (
            (None, None),
            *FIXED_CAMERA_SIZES,
        ):
            raise ValueError(
                "fixed_camera_width/height must both be unset, 1280x720 or 640x360"
            )
        if self.policy_contract_path is None and self.model_path is None:
            raise ValueError("configure policy_contract_path or model_path")
        if self.model_path is not None:
            validate_model_path(self.model_path)
        if self.model_data_root is not None:
            validate_data_root(self.model_data_root)
        if not self.policy_server_host or any(
            char.isspace() for char in self.policy_server_host
        ):
            raise ValueError(
                "policy_server_host must be a non-empty hostname or IP address"
            )
        if self.server_uri is None:
            host = self.policy_server_host
            uri_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
            self.server_uri = WebsocketUrl(
                f"ws://{uri_host}:{self.policy_server_port}"
            )
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
        if not self.binary_gripper_open_threshold < self.binary_gripper_close_threshold:
            raise ValueError(
                "binary gripper thresholds must satisfy 0 <= open < close <= 1"
            )
        if (
            self.action_prefix_steps is not None
            and self.action_prefix_duration_sec is not None
        ):
            raise ValueError(
                "set only one of action_prefix_steps and action_prefix_duration_sec"
            )
        return self

    def validate_contract(
        self, contract: PolicyContract, profile: SafetyProfile
    ) -> None:
        pinned_scalars = {
            "max_reach_m": profile.max_reach_m,
            "workspace_floor_m": profile.workspace_floor_m,
            "tcp_offset_m": profile.tcp_offset_m,
        }
        for key, expected in pinned_scalars.items():
            if (value := getattr(self, key)) is not None and value != expected:
                raise ValueError(f"{key} is fixed by safety profile {profile.name}")
        pinned_vectors = {
            "joint_min_radians": profile.joint_min,
            "joint_max_radians": profile.joint_max,
            "velocity_limits_rad_s": profile.velocity_limits,
            "acceleration_limits_rad_s2": profile.acceleration_limits,
        }
        for key, expected in pinned_vectors.items():
            if (value := getattr(self, key)) is not None and value != expected:
                raise ValueError(f"{key} is fixed by safety profile {profile.name}")
        expected_gripper_units = (
            "binary" if contract.embodiment.gripper_mode == "binary" else "normalized"
        )
        if contract.observation_schema.gripper_units != expected_gripper_units:
            raise ValueError(
                "observation gripper_units must match embodiment gripper_mode"
            )
        self._contract = contract
        self._profile = profile

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    @property
    def profile(self) -> SafetyProfile:
        return self._profile

    @property
    def dependencies(self) -> tuple[str, ...]:
        required = (self.arm, self.gripper, self.fixed_camera, self.wrist_camera)
        if self.fixed_camera_2:
            required = (*required, self.fixed_camera_2)
        return (self.world, *required) if self.world is not None else required

    @property
    def fixed_camera_sizes(self) -> tuple[tuple[int, int], ...]:
        if self.fixed_camera_width is None or self.fixed_camera_height is None:
            return FIXED_CAMERA_SIZES
        return ((self.fixed_camera_width, self.fixed_camera_height),)

    @property
    def wrist_camera_sizes(self) -> tuple[tuple[int, int], ...]:
        return ((self.wrist_camera_width, self.wrist_camera_height),)

    @property
    def default_prefix_steps(self) -> int:
        if self.action_prefix_duration_sec is not None:
            return _prefix_steps_from_duration(
                self.action_prefix_duration_sec,
                1.0 / self.contract.conditioning_fps,
            )
        return self.action_prefix_steps if self.action_prefix_steps is not None else 8

    @classmethod
    def _unconfigured(cls) -> Self:
        config = cls.model_construct(
            world=None,
            arm="",
            gripper="",
            fixed_camera="",
            fixed_camera_2="",
            wrist_camera="",
            server_uri="",
            prompt=None,
            policy_contract_path="",
        )
        config._contract = PolicyContract.model_construct(
            embodiment=ContractEmbodiment.model_construct(
                arm_model="", gripper_model="", gripper_mode="position", joint_names=()
            )
        )
        config._profile = SAFETY_PROFILES[("ur20", "robotiq_2f_85")]
        return config


def _model_data_root(config: Config) -> Path:
    return (
        validate_data_root(config.model_data_root)
        if config.model_data_root is not None
        else default_data_root()
    )


class CommandModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StopCommand(CommandModel):
    command: Literal["stop"]


class StatusCommand(CommandModel):
    command: Literal["status"]


class PolicyServerCommand(CommandModel):
    command: Literal["policy_server"]
    action: Literal["start", "status", "stop"]


class ExecuteCommand(CommandModel):
    command: Literal["execute"] = "execute"
    prompt: Prompt
    max_chunks: Annotated[int, Field(ge=1, le=100)]
    action_prefix_steps: Annotated[int, Field(ge=1, le=32)] | None = None
    action_prefix_duration_sec: PositiveFiniteFloat | None = None
    conditioner_algorithm: Literal["quintic", "toppra"]
    deadline_tolerance_sec: PositiveFiniteFloat
    progress_prop: Annotated[str, StringConstraints(strip_whitespace=True)] = ""
    max_consecutive_rejections: Annotated[int, Field(ge=0, le=10)] = 0
    """How many chunks in a row may fail safety validation before the run fails. A
    rejected chunk executes nothing (the arm holds) and the policy is re-queried from a
    fresh observation; 0 keeps the fail-closed default."""
    action_smoothing_window: Annotated[int, Field(ge=1, le=15)] = 1
    """Odd moving-average window applied to the six joint columns of each returned chunk
    before conditioning; 1 disables it. Row-to-row jitter in a sampled chunk inflates the
    TOPPRA path length and stretches its duration, so a short window keeps the executed
    cadence close to the training labels without changing where the chunk goes."""
    replay: ReplaySpec | None = None

    @model_validator(mode="after")
    def validate_prefix(self) -> Self:
        if (
            self.action_prefix_steps is not None
            and self.action_prefix_duration_sec is not None
        ):
            raise ValueError(
                "set only one of action_prefix_steps and action_prefix_duration_sec"
            )
        if self.action_smoothing_window % 2 == 0:
            raise ValueError("action_smoothing_window must be odd")
        if self.action_prefix_steps is None and self.action_prefix_duration_sec is None:
            raise ValueError(
                "set one of action_prefix_steps and action_prefix_duration_sec"
            )
        return self

    def prefix_steps(self, period_s: float) -> int:
        if self.action_prefix_duration_sec is not None:
            return _prefix_steps_from_duration(
                self.action_prefix_duration_sec, period_s
            )
        assert self.action_prefix_steps is not None
        return self.action_prefix_steps


DoCommand = Annotated[
    ExecuteCommand
    | StopCommand
    | StatusCommand
    | PolicyServerCommand,
    Field(discriminator="command"),
]
DO_COMMAND_ADAPTER = TypeAdapter(DoCommand)


@dataclass(slots=True)
class RunStatus:
    deadline_tolerance_sec: float
    embodiment: dict[str, Any]
    safety_profile: str
    conditioner_algorithm: str
    gripper_mode: str
    state: str = "idle"
    run_id: str = ""
    prompt: str = ""
    chunk_index: int = 0
    action_index: int = 0
    actions_executed: int = 0
    executed_prefixes: int = 0
    rejected_chunks: int = 0
    chunks_inferred: int = 0
    inference_responses: int = 0
    inference_ms: list[float] = field(default_factory=list)
    server_timing: list[Any] = field(default_factory=list)
    raw_action_chunks: list[list[list[float]]] = field(default_factory=list)
    conditioned_action_chunks: list[list[list[float]]] = field(default_factory=list)
    conditioning: list[dict[str, Any]] = field(default_factory=list)
    progress: list[dict[str, Any]] = field(default_factory=list)
    actuation_timing: list[dict[str, Any]] = field(default_factory=list)
    server_metadata: Any = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    last_error: str = ""
    stop_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        name="lite6-integrated-gripper-sim-v4",
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
        # poses. Scripted training picks (114 episodes, 2026-10-13) drive the
        # sim arm faster: measured joint speed p99 up to 1.04 rad/s (max 1.76)
        # and acceleration p99 up to ~9.7 rad/s². Limits below those made the
        # arm lag every chunk it was trained to follow, so allow p99 motion.
        velocity_limits=(1.2,) * 6,
        acceleration_limits=(10.0,) * 6,
        max_reach_m=0.70,
        # The floor is the table top. In the sim cell the arm base sits 5 mm
        # above the table, so the table is at -5 mm in the arm frame. Scripted
        # grasps put the TCP 3-10 mm above the base, and a policy grasp went to
        # -1.3 mm (3.7 mm above the table), which a 0.0 floor rejected. The
        # rejection repeats from that pose, so the run could not recover.
        workspace_floor_m=-0.005,
        tcp_offset_m=0.0836,
        fk=lite6_tcp_position,
    ),
}


def _read_policy_contract(path_value: Any) -> tuple[PolicyContract, SafetyProfile]:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("policy_contract_path must name a policy contract JSON file")
    path = Path(path_value).expanduser()
    try:
        contract_json = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read policy contract {path}: {exc}") from exc
    contract = PolicyContract.model_validate_json(contract_json)

    identity = (
        contract.embodiment.arm_model,
        contract.embodiment.gripper_model,
    )
    profile = SAFETY_PROFILES.get(identity)
    if profile is None:
        raise ValueError(f"no safety profile for embodiment {identity!r}")
    if contract.embodiment.joint_names != profile.joint_names:
        raise ValueError(
            "policy contract joint order does not match its safety profile"
        )
    expected_mode = "binary" if profile.gripper_model == "lite6_gripper" else "position"
    expected_units = "binary" if expected_mode == "binary" else "normalized"
    if contract.embodiment.gripper_mode != expected_mode:
        raise ValueError("policy contract gripper mode does not match its embodiment")
    if contract.action_schema.names != (
        *profile.joint_names,
        "gripper",
    ) or contract.action_schema.units != (*(("rad",) * 6), expected_units):
        raise ValueError("policy contract action schema does not match its embodiment")
    observation = contract.observation_schema
    if (
        observation.joint_names != profile.joint_names
        or observation.camera_keys != CAMERA_KEYS
        or observation.camera_aliases != CAMERA_ALIASES
    ):
        raise ValueError(
            "policy contract observation schema does not match the Viam bridge"
        )
    if (
        contract.transform.camera_keys != CAMERA_KEYS
        or contract.transform.camera_aliases != CAMERA_ALIASES
    ):
        raise ValueError(
            "policy contract transform camera mapping does not match the Viam bridge"
        )
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


def _prefix_steps_from_duration(value: Any, period_s: float) -> int:
    duration = float(value)
    steps_float = duration / period_s
    steps = round(steps_float)
    if (
        not math.isfinite(duration)
        or not math.isclose(steps_float, steps, abs_tol=1e-9)
        or not 1 <= steps <= ACTION_SHAPE[0]
    ):
        maximum = ACTION_SHAPE[0] * period_s
        raise ValueError(
            "action_prefix_duration_sec must align with the policy rate and be "
            f"in [{period_s:g}, {maximum:g}]"
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
        self._world: Generic | None
        self._arm: Arm
        self._gripper: Gripper
        self._fixed_camera: Camera
        self._fixed_camera_2: Camera | None = None
        self._wrist_camera: Camera
        self._config = Config._unconfigured()
        self._binary_gripper_state = 0.0
        self._smoothing_window = 1
        self._max_consecutive_rejections = 0
        self._owner_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._execution_task: Optional[asyncio.Task[Mapping[str, ValueTypes]]] = None
        self._socket: Any = None
        self._generation = 0
        self._closed = False
        self._ready = False
        self._readiness_error = ""
        self._status: RunStatus = self._idle_status()
        self._last_result: Optional[dict[str, Any]] = None
        self._policy_server_watch_task: asyncio.Task[None] | None = None
        self._policy_server_watch_generation = 0
        self._reported_policy_server_failure: tuple[int, str] | None = None
        self._policy_server = PolicyServerSupervisor(default_data_root(), name)

    @property
    def _period_s(self) -> float:
        return 1.0 / self._config.contract.conditioning_fps

    def _detach_policy_server_watcher(self) -> None:
        self._policy_server_watch_generation += 1
        task = self._policy_server_watch_task
        self._policy_server_watch_task = None
        if task is not None and not task.done():
            task.cancel()

    def _ensure_policy_server_watcher(self, *, replace: bool = False) -> None:
        if self._closed:
            return
        task = self._policy_server_watch_task
        if task is not None and not task.done():
            if not replace:
                return
            self._detach_policy_server_watcher()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        generation = self._policy_server_watch_generation
        supervisor = self._policy_server
        self._policy_server_watch_task = loop.create_task(
            self._watch_policy_server(supervisor, generation),
            name=f"policy-server-watch:{self.name}",
        )

    async def _watch_policy_server(
        self, supervisor: PolicyServerSupervisor, generation: int
    ) -> None:
        try:
            while (
                not self._closed
                and self._policy_server is supervisor
                and self._policy_server_watch_generation == generation
            ):
                failure = await asyncio.to_thread(supervisor.poll_failure)
                if (
                    self._closed
                    or self._policy_server is not supervisor
                    or self._policy_server_watch_generation != generation
                ):
                    return
                if failure is not None:
                    if failure.identity != self._reported_policy_server_failure:
                        self._reported_policy_server_failure = failure.identity
                        exit_status = (
                            "unavailable"
                            if failure.exit_status is None
                            else failure.exit_status
                        )
                        self.logger.error(
                            "managed policy-server exited unexpectedly; pid=%s; "
                            "exit_status=%s; log_path=%s; tail=%s",
                            failure.pid,
                            exit_status,
                            failure.log_path,
                            failure.detail,
                        )
                    return
                if not await asyncio.to_thread(supervisor._has_state):
                    return

                await asyncio.sleep(0.2)
        finally:
            if self._policy_server_watch_task is asyncio.current_task():
                self._policy_server_watch_task = None

    def _contract_path(self) -> Path:
        if self._config.policy_contract_path is not None:
            return Path(self._config.policy_contract_path).expanduser()
        assert self._config.model_path is not None
        return validate_model_path(self._config.model_path) / "policy_contract.json"

    def _advance_readiness(self) -> bool:
        if self._ready:
            return True
        try:
            contract, profile = _read_policy_contract(str(self._contract_path()))
            self._config.validate_contract(contract, profile)
        except (OSError, ValueError) as exc:
            self._readiness_error = str(exc)
            self._status = self._idle_status()
            self._status.state = "wait_contract"
            self._status.last_error = self._readiness_error
            return False
        self._ready = True
        self._readiness_error = ""
        self._status = self._idle_status()
        self._status.state = "ready"
        return True

    def _require_ready(self) -> None:
        if not self._advance_readiness():
            raise RuntimeError(
                "policy contract is not ready: "
                f"{self._readiness_error}; configure a valid policy_contract_path or "
                "wait for viam-server to deploy the configured model_path package"
            )

    def _idle_status(self) -> RunStatus:
        return RunStatus(
            deadline_tolerance_sec=self._config.deadline_tolerance_sec,
            embodiment=self._config.contract.embodiment.model_dump(mode="json"),
            safety_profile=self._config.profile.name,
            conditioner_algorithm=self._config.conditioner_algorithm,
            gripper_mode=self._config.contract.embodiment.gripper_mode,
        )

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
        self._world = (
            self._dependency(dependencies, Generic, parsed.world, "world")
            if parsed.world is not None
            else None
        )
        self._arm = self._dependency(dependencies, Arm, parsed.arm, "arm")
        self._gripper = self._dependency(
            dependencies, Gripper, parsed.gripper, "gripper"
        )
        self._fixed_camera = self._dependency(
            dependencies, Camera, parsed.fixed_camera, "fixed camera"
        )
        self._fixed_camera_2 = (
            self._dependency(
                dependencies, Camera, parsed.fixed_camera_2, "second fixed camera"
            )
            if parsed.fixed_camera_2
            else None
        )
        self._wrist_camera = self._dependency(
            dependencies, Camera, parsed.wrist_camera, "wrist camera"
        )
        next_policy_server = PolicyServerSupervisor(
            _model_data_root(parsed), self.name
        )
        self._detach_policy_server_watcher()
        if parsed.policy_server_autoclose and self._policy_server is not None:
            self._policy_server.stop()
        parsed._contract = self._config.contract
        parsed._profile = self._config.profile
        self._config = parsed
        self._binary_gripper_state = parsed.binary_gripper_initial_state
        self._ready = False
        self._readiness_error = ""
        self._policy_server = next_policy_server
        self._ensure_policy_server_watcher()
        self._advance_readiness()

    @staticmethod
    def _decode_image(
        image: Any, allowed: tuple[tuple[int, int], ...], label: str
    ) -> np.ndarray:
        mime = str(getattr(image, "mime_type", ""))
        if mime and not mime.startswith("image/"):
            raise ValueError(f"{label} returned unsupported MIME type {mime!r}")
        try:
            decoded = Image.open(BytesIO(image.data)).convert("RGB")
            decoded.load()
        except Exception as exc:
            raise ValueError(f"{label} returned an invalid encoded image") from exc
        if decoded.size not in allowed:
            expected = " or ".join(f"{w}x{h}" for w, h in allowed)
            raise ValueError(f"{label} image is {decoded.size}, expected {expected}")
        return np.ascontiguousarray(np.asarray(decoded, dtype=np.uint8))

    async def _joint_radians(self) -> list[float]:
        positions = await self._arm.get_joint_positions()
        return [math.radians(float(value)) for value in positions.values]

    async def _timed_joint_radians(self) -> tuple[list[float], float]:
        joints = await self._joint_radians()
        return joints, time.monotonic()

    async def _gripper_input(self) -> list[float]:
        if self._config.contract.embodiment.gripper_mode == "binary":
            return [self._binary_gripper_state]
        return [float(value) for value in await self._gripper.get_current_inputs()]

    async def _observation(self, prompt: str) -> dict[str, Any]:
        camera_requests = [
            self._fixed_camera.get_images(),
            self._wrist_camera.get_images(),
        ]
        if self._fixed_camera_2 is not None:
            camera_requests.append(self._fixed_camera_2.get_images())
        results = await asyncio.gather(
            *camera_requests,
            self._timed_joint_radians(),
            self._gripper_input(),
        )
        fixed_images, _ = results[0]
        wrist_images, _ = results[1]
        fixed_2_images = (
            results[2][0] if self._fixed_camera_2 is not None else fixed_images
        )
        joint_values, joint_sampled_at = results[-2]
        gripper_inputs = results[-1]
        if any(
            len(images) != 1 for images in (fixed_images, wrist_images, fixed_2_images)
        ):
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
            fixed_images[0], self._config.fixed_camera_sizes, "fixed camera"
        )
        fixed_2 = self._decode_image(
            fixed_2_images[0], self._config.fixed_camera_sizes, "second fixed camera"
        )
        wrist = self._decode_image(
            wrist_images[0], self._config.wrist_camera_sizes, "wrist camera"
        )
        return {
            "prompt": prompt,
            "observation/wrist_image_left": wrist,
            "observation/exterior_image_1_left": fixed,
            "observation/exterior_image_2_left": fixed_2,
            "observation/joint_position": joints,
            "observation/gripper_position": gripper,
            "_joint_sampled_at_monotonic": joint_sampled_at,
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
                done, _ = await asyncio.wait(
                    (task,), timeout=min(self._period_s, remaining)
                )
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
                    (receive,), timeout=min(self._period_s, remaining)
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

    def _wire_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return the observation as sent, pre-inverting the gripper if needed."""
        wire = dict(observation)
        if self._config.undo_server_gripper_flip:
            gripper = np.asarray(observation["observation/gripper_position"])
            wire["observation/gripper_position"] = (1.0 - gripper).astype(gripper.dtype)
        return wire

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
        parsed = np.asarray(actions, dtype=np.float64)
        if self._config.undo_server_gripper_flip:
            parsed = parsed.copy()
            parsed[:, -1] = 1.0 - parsed[:, -1]
        return parsed, dict(timing)

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
                velocity = np.abs(joints - previous) / self._period_s
                if np.any(velocity > velocity_limits + 1e-9):
                    raise ValueError(
                        f"action {index} exceeds {profile.arm_model} joint velocity limits"
                    )
                acceleration = np.abs(joints - 2.0 * previous + previous_previous) / (
                    self._period_s * self._period_s
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
    ) -> dict[str, Any]:
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

    def _smooth_actions(self, actions: np.ndarray) -> np.ndarray:
        """Centered moving average over the joint columns; edges repeat the end rows."""
        window = self._smoothing_window
        if window <= 1:
            return actions
        pad = window // 2
        padded = np.pad(actions[:, :6], ((pad, pad), (0, 0)), mode="edge")
        kernel = np.full(window, 1.0 / window)
        smoothed = np.stack(
            [np.convolve(padded[:, joint], kernel, mode="valid") for joint in range(6)],
            axis=1,
        )
        result = np.array(actions, dtype=np.float64, copy=True)
        result[:, :6] = smoothed
        return result

    def _condition_chunk(
        self,
        response: Any,
        history: Sequence[np.ndarray],
        prefix: int,
        conditioner_algorithm: str,
    ) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
        raw_actions, timing = self._parse_chunk(response)
        unsmoothed_path_length = float(np.abs(np.diff(raw_actions[:, :6], axis=0)).sum())
        raw_actions = self._smooth_actions(raw_actions)
        algorithm, version = (
            (TOPPRA_ALGORITHM, TOPPRA_ALGORITHM_VERSION)
            if conditioner_algorithm == "toppra"
            else (ALGORITHM, ALGORITHM_VERSION)
        )
        diagnostics: dict[str, Any] = {
            "algorithm": algorithm,
            "version": version,
            "control_hz": self._config.contract.conditioning_fps,
            "raw_steps": int(raw_actions.shape[0]),
            "raw_metrics": {
                **trajectory_metrics(raw_actions, history, self._period_s),
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
            "selected_prefix_duration_s": prefix * self._period_s,
            "observation_age_at_prefix_end_s": prefix * self._period_s,
            "smoothing_window": self._smoothing_window,
            "path_length_rad_before_smoothing": unsmoothed_path_length,
            "path_length_rad": float(np.abs(np.diff(raw_actions[:, :6], axis=0)).sum()),
            "rejection_reason": "",
        }
        self._status.conditioning[-1] = _value_types(diagnostics)
        self._validate_joint_positions(raw_actions)
        try:
            conditioner = (
                condition_trajectory_toppra
                if conditioner_algorithm == "toppra"
                else condition_trajectory
            )
            actions, raw_diagnostics = conditioner(
                raw_actions, history, self._config.profile, period_s=self._period_s
            )
        except ValueError as exc:
            if conditioner_algorithm != "toppra":
                raise
            actions, raw_diagnostics = condition_trajectory(
                raw_actions, history, self._config.profile, period_s=self._period_s
            )
            raw_diagnostics = {
                **dict(raw_diagnostics),
                "requested_algorithm": TOPPRA_ALGORITHM,
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            }
        clipped_gripper = np.clip(actions[:, 6], 0.0, 1.0)
        max_gripper_delta = float(np.max(np.abs(clipped_gripper - actions[:, 6])))
        actions = actions.copy()
        actions[:, 6] = clipped_gripper
        raw_diagnostics = {
            **dict(raw_diagnostics),
            "gripper_projection_max": max_gripper_delta,
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
        diagnostics["selected_prefix_duration_s"] = prefix * self._period_s
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
        diagnostics["validated_prefix_duration_s"] = validated_prefix * self._period_s
        diagnostics["observation_age_at_prefix_end_s"] = (
            validated_prefix * self._period_s
        )
        diagnostics["rejection_reason"] = ""
        self._status.conditioned_action_chunks.append(actions.tolist())
        self._status.conditioning[-1] = _value_types(diagnostics)
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
                    time=timedelta(seconds=(index + 1) * self._period_s),
                    positions=[math.degrees(float(value)) for value in action[:6]],
                )
                for index, action in enumerate(actions)
            )
            yield points

        stream = cast(Any, self._arm).move_through_joint_positions_streamed(batches())
        async for _ in stream:
            pass

    async def _dispatch_position_gripper(
        self,
        actions: np.ndarray,
        prefix_started: float,
        operation: Any,
        generation: int,
        deadline_tolerance: float,
    ) -> list[float]:
        lateness_samples: list[float] = []
        for action_index, action in enumerate(actions):
            deadline = prefix_started + action_index * self._period_s
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
                self._period_s + deadline_tolerance,
                operation,
                generation,
                "gripper target dispatch",
            )
            self._status.action_index = action_index + 1
            self._status.actions_executed += 1
        return lateness_samples

    def _binary_prefix(self, actions: np.ndarray) -> tuple[np.ndarray, float | None]:
        for action_index, action in enumerate(actions):
            value = float(action[6])
            desired = self._binary_gripper_state
            if value <= self._config.binary_gripper_open_threshold:
                desired = 0.0
            elif value >= self._config.binary_gripper_close_threshold:
                desired = 1.0
            if desired != self._binary_gripper_state:
                return actions[: action_index + 1], desired
        return actions, None

    async def _dispatch_binary_prefix(
        self,
        actions: np.ndarray,
        current_joints: Sequence[float],
        operation: Any,
        generation: int,
    ) -> dict[str, Any]:
        prefix_started = time.monotonic()
        selected, desired = self._binary_prefix(actions)
        await self._wait_cancellable(
            self._stream_arm_prefix(selected, current_joints),
            len(selected) * self._period_s + self._config.connect_timeout_sec,
            operation,
            generation,
            "binary gripper approach trajectory",
        )
        if await self._cancelled(operation, generation):
            raise asyncio.CancelledError
        transition = None
        if desired is not None:
            transition = "open" if desired == 0.0 else "grab"
            command = self._gripper.open() if desired == 0.0 else self._gripper.grab()
            await self._wait_cancellable(
                command,
                self._config.connect_timeout_sec,
                operation,
                generation,
                f"gripper {transition} command",
            )
            self._binary_gripper_state = desired
            if self._config.binary_gripper_settle_sec:
                await self._wait_cancellable(
                    asyncio.sleep(self._config.binary_gripper_settle_sec),
                    self._config.binary_gripper_settle_sec + 1.0,
                    operation,
                    generation,
                    "gripper settlement",
                )
            if await self._cancelled(operation, generation):
                raise asyncio.CancelledError
        self._status.action_index = len(selected)
        self._status.actions_executed += len(selected)
        return {
            "gripper_mode": "binary",
            "requested_steps": len(actions),
            "scheduled_steps": len(selected),
            "barrier_transition": transition,
            "settle_sec": (
                self._config.binary_gripper_settle_sec if transition else 0.0
            ),
            "elapsed_sec": time.monotonic() - prefix_started,
            "max_dispatch_lateness_sec": 0.0,
        }

    async def _dispatch_prefix(
        self,
        actions: np.ndarray,
        current_joints: Sequence[float],
        operation: Any,
        generation: int,
        deadline_tolerance: float,
    ) -> dict[str, Any]:
        if self._config.contract.embodiment.gripper_mode == "binary":
            return await self._dispatch_binary_prefix(
                actions, current_joints, operation, generation
            )

        prefix_started = time.monotonic()
        _, lateness_samples = await self._wait_cancellable(
            asyncio.gather(
                self._stream_arm_prefix(actions, current_joints),
                self._dispatch_position_gripper(
                    actions,
                    prefix_started,
                    operation,
                    generation,
                    deadline_tolerance,
                ),
            ),
            len(actions) * self._period_s + self._config.connect_timeout_sec,
            operation,
            generation,
            "trajectory dispatch",
        )
        return {
            "gripper_mode": "position",
            "requested_steps": len(actions),
            "scheduled_steps": len(actions),
            "barrier_transition": None,
            "settle_sec": 0.0,
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
        replay: ReplayChunkSource | None = None,
    ) -> Mapping[str, ValueTypes]:
        terminal_state = "completed"
        error = ""
        try:
            # Arm execution is always timestamped; the contract selects only
            # position-target versus binary-barrier gripper behavior.
            self._status.state = "connecting"
            if replay is None:
                socket, metadata = await self._connect(operation, generation)
            else:
                # Recorded rows stand in for the policy server; nothing else changes.
                socket, metadata = None, replay.metadata()
            self._status.server_metadata = _value_types(metadata)
            consecutive_rejections = 0
            for chunk_index in range(max_chunks):
                if await self._cancelled(operation, generation):
                    raise asyncio.CancelledError
                self._status.state = "inferring"
                self._status.chunk_index = chunk_index
                self._status.action_index = 0
                older_sample, older_sampled_at = await self._wait_cancellable(
                    self._timed_joint_radians(),
                    self._config.connect_timeout_sec,
                    operation,
                    generation,
                    "first measured joint sample",
                )
                older = np.asarray(older_sample, dtype=np.float64)
                if older.shape != (6,) or not np.isfinite(older).all():
                    raise ValueError("arm returned invalid first measured joint sample")
                await self._wait_cancellable(
                    asyncio.sleep(self._period_s),
                    self._period_s + deadline_tolerance,
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
                current_sampled_at = float(
                    observation.pop("_joint_sampled_at_monotonic")
                )
                current = np.asarray(
                    observation["observation/joint_position"], dtype=np.float64
                )
                history_interval = current_sampled_at - older_sampled_at
                if not math.isfinite(history_interval) or history_interval <= 0:
                    raise ValueError("joint history sample interval must be positive")
                measured_velocity = (current - older) / history_interval
                history = [current - measured_velocity * self._period_s, current]
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
                self._status.progress.append(
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
                if replay is None:
                    await self._wait_cancellable(
                        socket.send(pack_message(self._wire_observation(observation))),
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
                else:
                    response = replay.respond(observation)
                    inference_ms = (time.monotonic() - started) * 1000.0
                self._status.inference_responses += 1
                self._status.inference_ms.append(inference_ms)
                if isinstance(response, Mapping):
                    raw_actions = response.get("action")
                    if (
                        isinstance(raw_actions, np.ndarray)
                        and raw_actions.shape == ACTION_SHAPE
                        and np.isfinite(raw_actions).all()
                    ):
                        self._status.raw_action_chunks.append(raw_actions.tolist())
                        self._status.conditioning.append(
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
                        self._status.server_timing.append(_value_types(raw_timing))
                try:
                    actions, _, diagnostics = self._condition_chunk(
                        response, history, prefix, conditioner_algorithm
                    )
                except ValueError as exc:
                    if consecutive_rejections >= self._max_consecutive_rejections:
                        raise
                    # Hold instead of failing: nothing is dispatched, the next chunk is
                    # requested from a fresh observation of the unchanged arm state.
                    consecutive_rejections += 1
                    self._status.chunks_inferred += 1
                    self._status.rejected_chunks += 1
                    if self._status.conditioning:
                        self._status.conditioning[-1]["rejection_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._status.conditioning[-1]["held"] = True
                    continue
                consecutive_rejections = 0
                execution_prefix = int(diagnostics["validated_prefix_steps"])
                self._status.chunks_inferred += 1
                self._status.state = "executing"
                timing = await self._dispatch_prefix(
                    actions[:execution_prefix],
                    current.tolist(),
                    operation,
                    generation,
                    deadline_tolerance,
                )
                self._status.actuation_timing.append(timing)
                self._status.executed_prefixes += 1
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
                self._status.progress.append(
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
            if self._status.conditioning:
                latest = self._status.conditioning[-1]
                if not latest.get("rejection_reason"):
                    latest["rejection_reason"] = error
            return self._finish_result(terminal_state, error)
        finally:
            self._generation += 1
            await self._close_socket()
            stop_errors = await self._safe_stop()
            self._status.stop_errors = stop_errors
            if stop_errors:
                stop_error = "safe stop failed: " + "; ".join(stop_errors)
                self._status.state = "failed"
                self._status.last_error = stop_error
                if self._last_result is not None:
                    self._last_result.update(state="failed", error=stop_error)
            self._status.finished_at = _utc_now()
            if self._last_result is not None:
                self._last_result.update(
                    stop_errors=stop_errors, finished_at=self._status.finished_at
                )

    def _finish_result(self, state: str, error: str) -> dict[str, Any]:
        finished_at = _utc_now()
        self._status.state = state
        self._status.last_error = error
        self._status.finished_at = finished_at
        result = {
            **self._status.to_dict(),
            "state": state,
            "error": error,
            "finished_at": finished_at,
        }
        result.pop("last_error")
        self._last_result = result
        return result

    async def _execute_command(
        self, command: ExecuteCommand, kwargs: Mapping[str, Any]
    ) -> Mapping[str, ValueTypes]:
        if self._closed:
            raise RuntimeError(f"policy {self.name} is closed")
        self._require_ready()
        prompt = command.prompt
        max_chunks = command.max_chunks
        prefix = command.prefix_steps(self._period_s)
        conditioner_algorithm = command.conditioner_algorithm
        deadline_tolerance = command.deadline_tolerance_sec
        progress_prop = command.progress_prop
        self._smoothing_window = command.action_smoothing_window
        self._max_consecutive_rejections = command.max_consecutive_rejections
        if progress_prop and self._world is None:
            raise ValueError("progress_prop requires a configured world dependency")
        replay = (
            ReplayChunkSource.from_spec(command.replay)
            if command.replay is not None
            else None
        )
        async with self._owner_lock:
            if self._execution_task is not None and not self._execution_task.done():
                raise RuntimeError("policy execution is already in progress")
            self._stop_event = asyncio.Event()
            self._generation += 1
            generation = self._generation
            self._last_result = None
            self._status = self._idle_status()
            # Each run starts from the configured initial gripper state: the workflow's
            # scene setup establishes it physically, and the policy must observe that
            # value rather than the last barrier of the previous run.
            self._binary_gripper_state = self._config.binary_gripper_initial_state
            self._status.state = "connecting"
            self._status.run_id = str(uuid4())
            self._status.prompt = prompt
            self._status.started_at = _utc_now()
            self._status.conditioner_algorithm = conditioner_algorithm
            self._status.deadline_tolerance_sec = deadline_tolerance
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
                    replay,
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
            self._status.state = "stopping"
        await self._close_socket()
        immediate_errors = await self._safe_stop()
        if task is not None and not task.done() and task is not asyncio.current_task():
            await asyncio.shield(task)
        result = (
            self._last_result.copy() if self._last_result else self._status.to_dict()
        )
        if immediate_errors:
            result["stop_errors"] = (
                list(result.get("stop_errors", [])) + immediate_errors
            )
        return result

    async def _policy_server_model(self) -> Path:
        if self._config.model_path is None:
            raise NotImplementedError("policy_server start requires configured model_path")
        model = validate_model_path(self._config.model_path)
        await asyncio.to_thread(validate_runtime_manifest, model)
        return model

    async def _policy_server_command(
        self, command: PolicyServerCommand
    ) -> Mapping[str, ValueTypes]:
        if command.action == "status":
            self._ensure_policy_server_watcher()
            return await asyncio.to_thread(self._policy_server.status)
        if command.action == "stop":
            self._detach_policy_server_watcher()
            result = await asyncio.to_thread(self._policy_server.stop)
            if not result.get("stopped", False):
                self._ensure_policy_server_watcher()
            return result
        self.logger.info(
            "managed policy-server start requested; host=%s; port=%s",
            self._config.policy_server_host,
            self._config.policy_server_port,
        )
        model = await self._policy_server_model()
        runtime_root = Path(__file__).resolve().parents[2] / "policy_server_runtime"
        try:
            result = await asyncio.to_thread(
                self._policy_server.start,
                runtime_command(
                    runtime_root,
                    model,
                    self._config.policy_server_host,
                    self._config.policy_server_port,
                ),
                self._config.policy_server_host,
                self._config.policy_server_port,
                self._config.policy_server_readiness_timeout_sec,
            )
        except (OSError, RuntimeError) as exc:
            self.logger.error("managed policy-server failed during startup: %s", exc)
            raise
        self.logger.info(
            "managed policy-server ready; pid=%s; host=%s; port=%s; "
            "log_path=%s; idempotent=%s",
            result.get("pid", "unavailable"),
            result.get("host", self._config.policy_server_host),
            result.get("port", self._config.policy_server_port),
            result.get("log_path", "unavailable"),
            result.get("idempotent", False),
        )
        self._ensure_policy_server_watcher(replace=True)
        return result

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        self._ensure_policy_server_watcher()
        payload: dict[str, Any] = dict(command)
        if payload.get("command") == "execute":
            defaults: dict[str, Any] = {
                "command": "execute",
                "max_chunks": self._config.max_chunks,
                "conditioner_algorithm": self._config.conditioner_algorithm,
                "deadline_tolerance_sec": self._config.deadline_tolerance_sec,
            }
            if "prompt" not in payload:
                if self._config.prompt is None:
                    raise ValueError(
                        "execute requires prompt because no default prompt is configured"
                    )
                defaults["prompt"] = self._config.prompt
            if (
                "action_prefix_steps" not in payload
                and "action_prefix_duration_sec" not in payload
            ):
                defaults["action_prefix_steps"] = self._config.default_prefix_steps
            payload = {**defaults, **payload}
        request = DO_COMMAND_ADAPTER.validate_python(payload)
        if isinstance(request, ExecuteCommand):
            return await self._execute_command(request, kwargs)
        if isinstance(request, StopCommand):
            return await self._stop_command()
        if isinstance(request, StatusCommand):
            self._advance_readiness()
            return self._status.to_dict()
        return await self._policy_server_command(request)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._detach_policy_server_watcher()
        await self._stop_command()
        if self._config.policy_server_autoclose and self._policy_server is not None:
            await asyncio.to_thread(self._policy_server.stop)
