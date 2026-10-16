"""Replay recorded action rows in place of a policy server.

An ``execute`` command may carry a ``replay`` object. The component then skips the
websocket and answers each observation from the recorded rows, so the whole execution
path (conditioning, prefix validation, gripper barrier, dispatch, re-observation) runs
on an expert's own labels. This is the control experiment for "can these labels be
executed at all under the deployment limits".

With ``episodes`` given (one episode id per row) the rows may span a whole dataset and
``nearest`` mode becomes a retrieval policy: every observation is answered with the 32
rows that follow its nearest recorded state anywhere in the dataset, with the gripper
state included in the distance so pre- and post-grasp states do not alias.
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
    episodes: list[int] | None = None
    mode: Literal["nearest", "clock"] = "nearest"
    advance_steps: Annotated[int, Field(ge=1, le=ACTION_CHUNK_SIZE)] = 16
    start_frame: Annotated[int, Field(ge=0)] = 0
    allow_backtrack: bool = False
    gripper_weight: Annotated[float, Field(ge=0.0)] = 1.0
    """Radians of joint distance that one unit of gripper mismatch counts for (dataset mode)."""

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if (self.actions is None) == (self.actions_path is None):
            raise ValueError("set exactly one of replay.actions and replay.actions_path")
        if self.episodes is not None and self.actions is None:
            raise ValueError("replay.episodes needs inline replay.actions; a file carries its own")
        return self

    def _document(self) -> Any:
        if self.actions is not None:
            return {"actions": self.actions, "episodes": self.episodes}
        assert self.actions_path is not None
        path = Path(self.actions_path).expanduser()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read replay actions {path}: {exc}") from exc
        return document if isinstance(document, dict) else {"actions": document}

    def load_rows(self) -> np.ndarray:
        """Return the rows as a finite ``[T, 7]`` float64 array with gripper in [0, 1]."""
        return self.load()[0]

    def load(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return ``(rows, episodes)``; ``episodes`` is ``None`` for a single recording."""
        document = self._document()
        try:
            rows = np.asarray(document.get("actions"), dtype=np.float64)
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
        raw_episodes = document.get("episodes")
        if raw_episodes is None:
            return rows, None
        try:
            episodes = np.asarray(raw_episodes, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"replay episodes must be integers: {exc}") from exc
        if episodes.shape != (rows.shape[0],):
            raise ValueError("replay.episodes must have one entry per action row")
        return rows, episodes


class ReplayChunkSource:
    """Serve ``ACTION_CHUNK_SIZE``-row chunks from recorded rows, one per observation."""

    def __init__(
        self, rows: np.ndarray, spec: ReplaySpec, episodes: np.ndarray | None = None
    ) -> None:
        self._rows = np.asarray(rows, dtype=np.float64)
        self._spec = spec
        self._episodes = None if episodes is None else np.asarray(episodes, dtype=np.int64)
        self._last_frame = -1
        self.served: list[int] = []
        if self._episodes is not None:
            # Last row index of each row's episode, for chunk padding at episode ends.
            last = np.empty_like(self._episodes)
            end = len(self._episodes) - 1
            for index in range(len(self._episodes) - 1, -1, -1):
                if index < end and self._episodes[index] != self._episodes[index + 1]:
                    end = index
                last[index] = end
            self._episode_end = last

    @classmethod
    def from_spec(cls, spec: ReplaySpec) -> "ReplayChunkSource":
        rows, episodes = spec.load()
        return cls(rows, spec, episodes)

    @property
    def frame_count(self) -> int:
        return int(self._rows.shape[0])

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "replay" if self._episodes is None else "replay-dataset",
            "mode": self._spec.mode,
            "frames": self.frame_count,
            "episodes": 1 if self._episodes is None else int(len(np.unique(self._episodes))),
            "advance_steps": self._spec.advance_steps,
            "allow_backtrack": self._spec.allow_backtrack,
            "gripper_weight": self._spec.gripper_weight,
        }

    def _nearest(self, joints: np.ndarray, gripper: float | None) -> tuple[int, float]:
        if self._episodes is not None:
            # Retrieval over the whole dataset: no forward constraint, gripper-aware.
            distance = np.linalg.norm(self._rows[:, :JOINT_COUNT] - joints[None, :], axis=1)
            if gripper is not None and self._spec.gripper_weight > 0.0:
                distance = np.hypot(
                    distance,
                    self._spec.gripper_weight * np.abs(self._rows[:, JOINT_COUNT] - gripper),
                )
            best = int(np.argmin(distance))
            return best, float(distance[best])
        start = (
            0
            if self._spec.allow_backtrack or self._last_frame < 0
            else self._last_frame
        )
        candidates = self._rows[start:, :JOINT_COUNT]
        distance = np.linalg.norm(candidates - joints[None, :], axis=1)
        best = int(np.argmin(distance))
        return start + best, float(distance[best])

    def _select(
        self, joints: np.ndarray | None, gripper: float | None
    ) -> tuple[int, float]:
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
        return self._nearest(joints, gripper)

    def chunk(self, frame: int) -> np.ndarray:
        """Rows ``frame .. frame + 32``, holding the last row of the recording (or episode)."""
        end = self.frame_count if self._episodes is None else int(self._episode_end[frame]) + 1
        rows = self._rows[frame : min(frame + ACTION_CHUNK_SIZE, end)]
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
        raw_gripper = observation.get("observation/gripper_position")
        gripper = (
            None
            if raw_gripper is None
            else float(np.asarray(raw_gripper, dtype=np.float64).reshape(-1)[-1])
        )
        frame, distance = self._select(joints, gripper)
        self._last_frame = frame
        self.served.append(frame)
        timing: dict[str, Any] = {
            "replay_frame": float(frame),
            "replay_distance_rad": distance if np.isfinite(distance) else -1.0,
            "replay_mode": self._spec.mode,
        }
        if self._episodes is not None:
            timing["replay_episode"] = float(self._episodes[frame])
        return {"action": self.chunk(frame), "server_timing": timing}
