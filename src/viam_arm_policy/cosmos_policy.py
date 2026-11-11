"""MLModel service that owns the Cosmos policy export and its serving process."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import websockets
import websockets.exceptions
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Self
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.proto.service.mlmodel import Metadata, TensorInfo
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.mlmodel import MLModel
from viam.utils import ValueTypes, dict_to_struct, struct_to_dict

from .model_export import (
    default_data_root,
    validate_data_root,
    validate_model_path,
    validate_runtime_manifest,
)
from .policy_server import PolicyServerSupervisor, runtime_command
from .wire import (
    ACTION_SHAPE,
    ACTION_TENSOR,
    CONTRACT_EXTRA,
    GRIPPER_TENSOR,
    IMAGE_TENSORS,
    JOINT_TENSOR,
    OBSERVATION_TENSORS,
    PROMPT_TENSOR,
    SERVER_TIMING_PREFIX,
    pack_message,
    prompt_from_tensor,
    unpack_message,
)

NAMESPACE = "viam-labs"
FAMILY = "arm-policy"

PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
WebsocketUrl = Annotated[str, StringConstraints(pattern=r"^wss?://\S+$")]
_RUNTIME_ROOT = Path(__file__).resolve().parents[2] / "policy_server_runtime"


class CosmosConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # A portable export deployed by viam-server from the Viam registry, normally
    # configured as "${packages.ml_model.<package-name>}".
    model_path: str
    # Holds the managed policy server's log.
    model_data_root: str | None = "/opt/viam/arm-policy"
    policy_server_host: str = "127.0.0.1"
    policy_server_port: Annotated[int, Field(ge=1, le=65535)] = 8000
    policy_server_readiness_timeout_sec: PositiveFiniteFloat = 900.0
    # Set to reach a server this service does not own; no subprocess is started.
    server_uri: WebsocketUrl | None = None
    connect_timeout_sec: PositiveFiniteFloat = 10.0
    inference_timeout_sec: PositiveFiniteFloat = 120.0
    # Written by the Viam app on "Deploy model"; informational only.
    package_reference: str | None = None

    @field_validator("model_path")
    @classmethod
    def strip_legacy_model_path_suffix(cls, value: str) -> str:
        return value.removesuffix("/runtime/vae/Wan2.2_VAE.pth")

    @model_validator(mode="after")
    def validate_static(self) -> Self:
        validate_model_path(self.model_path)
        if self.model_data_root is not None:
            validate_data_root(self.model_data_root)
        if not self.policy_server_host or any(
            char.isspace() for char in self.policy_server_host
        ):
            raise ValueError(
                "policy_server_host must be a non-empty hostname or IP address"
            )
        return self

    @property
    def data_root(self) -> Path:
        if self.model_data_root is not None:
            return validate_data_root(self.model_data_root)
        return default_data_root()

    @property
    def uri(self) -> str:
        if self.server_uri is not None:
            return self.server_uri
        host = self.policy_server_host
        uri_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        return f"ws://{uri_host}:{self.policy_server_port}"

    @property
    def contract_path(self) -> Path:
        return validate_model_path(self.model_path) / "policy_contract.json"


def _tensor_info(
    name: str,
    data_type: str,
    shape: Sequence[int],
    description: str,
    extra: Mapping[str, Any] | None = None,
) -> TensorInfo:
    return TensorInfo(
        name=name,
        description=description,
        data_type=data_type,
        shape=list(shape),
        extra=dict_to_struct(extra) if extra else None,
    )


class CosmosPolicy(MLModel, EasyResource):
    """Serves a Cosmos action-policy export over the MLModel API.

    Inputs are the OpenPI observation keys as tensors plus `prompt` as UTF-8 bytes;
    the output is the raw `action` chunk. Safety conditioning stays in the controller.
    The managed policy server starts in the background from `reconfigure` and lives
    exactly as long as this instance; the policy contract is published via `metadata()`.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "cosmos")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._config: CosmosConfig | None = None
        self._contract_json: str | None = None
        self._contract_error = "service is not configured"
        self._policy_server = PolicyServerSupervisor(default_data_root(), name)
        self._server_task: asyncio.Task[None] | None = None
        self._server_error = ""
        # The upstream server handles one request at a time; serialize callers here.
        self._infer_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    def new(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate_config(
        cls, config: ServiceConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        CosmosConfig.model_validate(struct_to_dict(config.attributes))
        return [], []

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        if self._config is not None:
            # The module SDK closes the old instance and builds a new one on every
            # config edit; reconfiguring a live instance would orphan its server.
            raise RuntimeError(f"cosmos policy {self.name} is already configured")
        parsed = CosmosConfig.model_validate(struct_to_dict(config.attributes))
        self._config = parsed
        self._policy_server = PolicyServerSupervisor(parsed.data_root, self.name)
        self._load_contract()
        if parsed.server_uri is None:
            self._server_task = asyncio.get_running_loop().create_task(
                self._run_policy_server(), name=f"policy-server:{self.name}"
            )

    def _require_config(self) -> CosmosConfig:
        if self._config is None:
            raise RuntimeError(f"cosmos policy {self.name} is not configured")
        return self._config

    def _load_contract(self) -> bool:
        if self._contract_json is not None:
            return True
        try:
            self._contract_json = self._require_config().contract_path.read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            self._contract_error = (
                f"policy contract is not available yet: {exc}; "
                "wait for viam-server to deploy the configured model_path package"
            )
            return False
        self._contract_error = ""
        return True

    # Policy-server supervision ------------------------------------------------

    async def _run_policy_server(self) -> None:
        try:
            await self._start_policy_server()
        except Exception as exc:
            self._server_error = f"{type(exc).__name__}: {exc}"
            self.logger.error("managed policy-server start failed: %s", self._server_error)
            return
        await self._watch_policy_server()

    def _launch(self, command: Sequence[str], config: CosmosConfig) -> dict[str, Any]:
        result = self._policy_server.start(
            command,
            config.policy_server_host,
            config.policy_server_port,
            config.policy_server_readiness_timeout_sec,
        )
        if self._closed:
            # close() ran before this thread launched the child, so its stop() found nothing.
            self._policy_server.stop()
            raise RuntimeError("policy server was stopped during startup")
        return result

    async def _start_policy_server(self) -> Mapping[str, ValueTypes]:
        config = self._require_config()
        self.logger.info(
            "managed policy-server start requested; host=%s; port=%s",
            config.policy_server_host,
            config.policy_server_port,
        )
        model = validate_model_path(config.model_path)
        await asyncio.to_thread(validate_runtime_manifest, model)
        command = runtime_command(
            _RUNTIME_ROOT, model, config.policy_server_host, config.policy_server_port
        )
        result = await asyncio.to_thread(self._launch, command, config)
        self._server_error = ""
        self.logger.info(
            "managed policy-server ready; pid=%s; host=%s; port=%s; "
            "log_path=%s; idempotent=%s",
            result.get("pid", "unavailable"),
            result.get("host", config.policy_server_host),
            result.get("port", config.policy_server_port),
            result.get("log_path", "unavailable"),
            result.get("idempotent", False),
        )
        return result

    async def _watch_policy_server(self) -> None:
        """Log an unexpected exit of the owned server once, then return."""
        supervisor = self._policy_server
        while await asyncio.to_thread(supervisor.running):
            failure = await asyncio.to_thread(supervisor.poll_failure)
            if failure is not None:
                self._server_error = f"exited with status {failure.exit_status}"
                self.logger.error(
                    "managed policy-server exited unexpectedly; pid=%s; "
                    "exit_status=%s; log_path=%s; tail=%s",
                    failure.process.pid,
                    failure.exit_status,
                    failure.log_path,
                    failure.detail,
                )
                return
            await asyncio.sleep(0.2)

    async def _unreachable_detail(self) -> str:
        config = self._require_config()
        if config.server_uri is not None:
            return f"external policy server at {config.uri} is unreachable"
        status = await asyncio.to_thread(self._policy_server.status)
        detail = f"managed policy server is {status.get('state', 'unknown')}"
        if self._server_error:
            detail += f"; last error: {self._server_error}"
        elif self._server_task is not None and not self._server_task.done():
            detail += "; startup is still in progress"
        return detail

    # WebSocket client -----------------------------------------------------------

    async def _connect(self) -> Any:
        """Open a socket and consume the server's metadata frame."""
        config = self._require_config()
        try:
            socket = await asyncio.wait_for(
                websockets.connect(
                    config.uri,
                    ping_interval=None,
                    compression=None,
                    max_size=64 * 1024 * 1024,
                    open_timeout=config.connect_timeout_sec,
                ),
                timeout=config.connect_timeout_sec,
            )
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            raise RuntimeError(
                f"cannot connect to {config.uri}: {exc}; {await self._unreachable_detail()}"
            ) from exc
        try:
            raw = await asyncio.wait_for(socket.recv(), timeout=config.connect_timeout_sec)
            if not isinstance(raw, bytes):
                raise RuntimeError(f"policy server metadata was text: {raw}")
            if not isinstance(unpack_message(raw), Mapping):
                raise ValueError("policy server metadata must be an object")
        except BaseException:
            await socket.close()
            raise
        return socket

    @staticmethod
    def _observation(input_tensors: Mapping[str, NDArray]) -> dict[str, Any]:
        missing = [
            name for name in (*OBSERVATION_TENSORS, PROMPT_TENSOR) if name not in input_tensors
        ]
        if missing:
            raise ValueError(f"missing input tensors: {', '.join(missing)}")
        observation: dict[str, Any] = {
            PROMPT_TENSOR: prompt_from_tensor(np.asarray(input_tensors[PROMPT_TENSOR]))
        }
        for name in IMAGE_TENSORS:
            image = np.asarray(input_tensors[name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name} must be a uint8 [H, W, 3] image")
            observation[name] = np.ascontiguousarray(image)
        joints = np.asarray(input_tensors[JOINT_TENSOR], dtype=np.float32)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError(f"{JOINT_TENSOR} must be six finite joint radians")
        gripper = np.asarray(input_tensors[GRIPPER_TENSOR], dtype=np.float32)
        if gripper.shape != (1,) or not np.isfinite(gripper).all():
            raise ValueError(f"{GRIPPER_TENSOR} must be one finite normalized value")
        observation[JOINT_TENSOR] = joints
        observation[GRIPPER_TENSOR] = gripper
        return observation

    @staticmethod
    def _output_tensors(response: Any) -> Dict[str, NDArray]:
        if not isinstance(response, Mapping):
            raise ValueError("policy response must be an object")
        actions = response.get(ACTION_TENSOR)
        if not isinstance(actions, np.ndarray) or actions.shape != ACTION_SHAPE:
            shape = getattr(actions, "shape", None)
            raise ValueError(
                f"policy action must be a NumPy array with shape {ACTION_SHAPE}, got {shape}"
            )
        if actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
            raise ValueError("policy action must contain only finite numeric values")
        outputs: Dict[str, NDArray] = {ACTION_TENSOR: np.asarray(actions, dtype=np.float32)}
        timing = response.get("server_timing", {})
        if isinstance(timing, Mapping):
            for key, value in timing.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    outputs[f"{SERVER_TIMING_PREFIX}{key}"] = np.asarray([value], dtype=np.float32)
        return outputs

    async def infer(
        self,
        input_tensors: Dict[str, NDArray],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, NDArray]:
        config = self._require_config()
        request = pack_message(self._observation(input_tensors))
        async with self._infer_lock:
            socket = await self._connect()
            try:
                await asyncio.wait_for(socket.send(request), timeout=config.inference_timeout_sec)
                raw = await asyncio.wait_for(socket.recv(), timeout=config.inference_timeout_sec)
            finally:
                await socket.close()
        if not isinstance(raw, bytes):
            # The upstream server answers a failed request with a text traceback.
            raise RuntimeError(f"policy server returned text error: {raw}")
        return self._output_tensors(unpack_message(raw))

    async def metadata(
        self, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None
    ) -> Metadata:
        config = self._require_config()
        if not self._load_contract():
            raise RuntimeError(self._contract_error)
        image_description = "RGB image, uint8 [height, width, 3]; the server resizes"
        inputs = [
            _tensor_info(name, "uint8", (-1, -1, 3), image_description) for name in IMAGE_TENSORS
        ]
        inputs.append(_tensor_info(JOINT_TENSOR, "float32", (6,), "joint positions in radians"))
        inputs.append(_tensor_info(GRIPPER_TENSOR, "float32", (1,), "gripper state in [0, 1], 0=open"))
        inputs.append(_tensor_info(PROMPT_TENSOR, "uint8", (-1,), "task prompt as UTF-8 bytes"))
        outputs = [
            _tensor_info(
                ACTION_TENSOR,
                "float32",
                ACTION_SHAPE,
                "absolute joint targets in radians plus gripper in [0, 1], one row per step",
                # The controller reads its safety contract from here; MLModel has no DoCommand.
                {CONTRACT_EXTRA: self._contract_json},
            ),
            _tensor_info(f"{SERVER_TIMING_PREFIX}<stage>", "float32", (1,), "server-side timing, when reported"),
        ]
        return Metadata(
            name=self.name,
            type="cosmos_action_policy",
            description=f"OpenPI WebSocket policy at {config.uri}",
            input_info=inputs,
            output_info=outputs,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        task, self._server_task = self._server_task, None
        if task is not None:
            # The start thread itself cannot be cancelled; stop() below interrupts it.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(self._policy_server.stop)
