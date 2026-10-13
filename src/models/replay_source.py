"""Replay recorded action rows in place of a policy server.

An ``execute`` command may carry a ``replay`` object. The component then skips the
websocket and answers each observation from the recorded rows, so the whole execution
path (conditioning, prefix validation, gripper barrier, dispatch, re-observation) runs
on an expert's own labels. This is the control experiment for "can these labels be
executed at all under the deployment limits".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

JOINT_COUNT = 6
ACTION_DIM = 7
ACTION_CHUNK_SIZE = 32


class ReplaySpec(BaseModel):
    """Recorded rows to replay, inline or from a JSON file on the module host."""

    model_config = ConfigDict(extra="forbid")

    actions: list[list[float]] | None = None
    actions_path: str | None = None
    mode: Literal["nearest", "clock"] = "nearest"
    advance_steps: Annotated[int, Field(ge=1, le=ACTION_CHUNK_SIZE)] = 16
    start_frame: Annotated[int, Field(ge=0)] = 0
    allow_backtrack: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if (self.actions is None) == (self.actions_path is None):
            raise ValueError("set exactly one of replay.actions and replay.actions_path")
        return self

    def load_rows(self) -> np.ndarray:
        """Return the rows as a finite ``[T, 7]`` float64 array with gripper in [0, 1]."""
        if self.actions is not None:
            raw: Any = self.actions
        else:
            assert self.actions_path is not None
            path = Path(self.actions_path).expanduser()
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"cannot read replay actions {path}: {exc}") from exc
            raw = document.get("actions") if isinstance(document, dict) else document
        try:
            rows = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"replay actions must be numeric rows: {exc}") from exc
        if rows.ndim != 2 or rows.shape[1] != ACTION_DIM or rows.shape[0] < 1:
            raise ValueError(
                f"replay actions must have shape [T, {ACTION_DIM}] with T >= 1, got {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("replay actions must be finite")
        if np.any(rows[:, JOINT_COUNT] < 0.0) or np.any(rows[:, JOINT_COUNT] > 1.0):
            raise ValueError("replay gripper values must be within [0, 1]")
        if self.start_frame >= rows.shape[0]:
            raise ValueError("replay.start_frame is past the last row")
        return rows


class ReplayChunkSource:
    """Serve ``ACTION_CHUNK_SIZE``-row chunks from recorded rows, one per observation."""

    def __init__(self, rows: np.ndarray, spec: ReplaySpec) -> None:
        self._rows = np.asarray(rows, dtype=np.float64)
        self._spec = spec
        self._last_frame = -1
        self.served: list[int] = []

    @property
    def frame_count(self) -> int:
        return int(self._rows.shape[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "replay",
            "mode": self._spec.mode,
            "frames": self.frame_count,
            "advance_steps": self._spec.advance_steps,
            "allow_backtrack": self._spec.allow_backtrack,
        }

    def _nearest(self, joints: np.ndarray) -> tuple[int, float]:
        start = (
            0
            if self._spec.allow_backtrack or self._last_frame < 0
            else self._last_frame
        )
        candidates = self._rows[start:, :JOINT_COUNT]
        distance = np.linalg.norm(candidates - joints[None, :], axis=1)
        best = int(np.argmin(distance))
        return start + best, float(distance[best])

    def _select(self, joints: np.ndarray | None) -> tuple[int, float]:
        if self._spec.mode == "clock" or joints is None:
            frame = (
                self._spec.start_frame
                if self._last_frame < 0
                else self._last_frame + self._spec.advance_steps
            )
            frame = min(frame, self.frame_count - 1)
            distance = (
                float(np.linalg.norm(self._rows[frame, :JOINT_COUNT] - joints))
                if joints is not None
                else float("nan")
            )
            return frame, distance
        return self._nearest(joints)

    def chunk(self, frame: int) -> np.ndarray:
        """Rows ``frame .. frame + 32``, holding the last row past the end of the recording."""
        rows = self._rows[frame : frame + ACTION_CHUNK_SIZE]
        if rows.shape[0] < ACTION_CHUNK_SIZE:
            pad = np.repeat(rows[-1:], ACTION_CHUNK_SIZE - rows.shape[0], axis=0)
            rows = np.vstack((rows, pad))
        return np.array(rows, dtype=np.float32)

    def respond(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return the same shape of response a policy server would send."""
        raw = observation.get("observation/joint_position")
        joints = (
            None
            if raw is None
            else np.asarray(raw, dtype=np.float64).reshape(-1)[-JOINT_COUNT:]
        )
        frame, distance = self._select(joints)
        self._last_frame = frame
        self.served.append(frame)
        return {
            "action": self.chunk(frame),
            "server_timing": {
                "replay_frame": float(frame),
                "replay_distance_rad": distance if np.isfinite(distance) else -1.0,
                "replay_mode": self._spec.mode,
            },
        }
