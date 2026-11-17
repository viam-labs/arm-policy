"""MessagePack framing for the OpenPI policy server and the MLModel tensor names."""

from __future__ import annotations

import math
from typing import Any, cast

import msgpack
import numpy as np

ACTION_SHAPE = (32, 7)
ACTION_TENSOR = "action"
PROMPT_TENSOR = "prompt"
SERVER_TIMING_PREFIX = "server_timing/"
# Key in the `action` output TensorInfo.extra carrying policy_contract.json verbatim.
CONTRACT_EXTRA = "policy_contract_json"
IMAGE_TENSORS = (
    "observation/wrist_image_left",
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
)
JOINT_TENSOR = "observation/joint_position"
GRIPPER_TENSOR = "observation/gripper_position"
OBSERVATION_TENSORS = (*IMAGE_TENSORS, JOINT_TENSOR, GRIPPER_TENSOR)


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


def prompt_to_tensor(prompt: str) -> np.ndarray:
    """Strings cannot cross the MLModel API as tensors; viam-server also drops `extra`."""
    return np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).copy()


def prompt_from_tensor(value: np.ndarray) -> str:
    if value.dtype != np.uint8 or value.ndim != 1 or value.size == 0:
        raise ValueError(f"{PROMPT_TENSOR} must be a non-empty uint8 vector of UTF-8 bytes")
    try:
        return value.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{PROMPT_TENSOR} is not valid UTF-8") from exc
