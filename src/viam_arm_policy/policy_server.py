"""Module-owned subprocess supervision for the optional policy-server runtime."""

from __future__ import annotations

import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Mapping, Sequence

_HEALTH_CHECK_TIMEOUT = 0.2
_HEALTH_RESPONSE = b"OK\n"
_TAIL_BYTES = 4096
_TOKEN_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:hf_token|huggingface_hub_token|token|api[_-]?key)\b"
    r"(\s*[:=]\s*[\"']?)([^\s\"',}\]]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)(\bauthorization\s*:\s*bearer\s+)(\S+)")
_HUGGING_FACE_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9_-]{6,}\b")


@dataclass(frozen=True)
class _ProcessInfo:
    start_ticks: str
    state: str


@dataclass(frozen=True)
class _PolicyServerExit:
    identity: tuple[int, str]
    pid: int
    exit_status: int | None
    log_path: str
    detail: str


def _process_info(pid: int) -> _ProcessInfo | None:
    try:
        # `/proc/<pid>/stat` wraps the command in parentheses, so split after
        # the final delimiter before indexing fields. Field 22 is start time.
        _, fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(
            ") ", 1
        )
        values = fields.split()
        return _ProcessInfo(start_ticks=values[19], state=values[0])
    except (IndexError, OSError, ValueError):
        return None


def _open_identity_pidfd(identity: tuple[int, str]) -> int | None:
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None or not hasattr(signal, "pidfd_send_signal"):
        return None
    try:
        pidfd = pidfd_open(identity[0])
    except OSError:
        return None
    info = _process_info(identity[0])
    if (
        info is None
        or info.start_ticks != identity[1]
        or info.state == "Z"
    ):
        os.close(pidfd)
        return None
    return pidfd


def _send_pidfd_signal(pidfd: int, signum: int) -> bool:
    try:
        signal.pidfd_send_signal(pidfd, signum)
    except OSError:
        return False
    return True


def _health_ready(host: str, port: int) -> bool:
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=_HEALTH_CHECK_TIMEOUT)
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        return (
            response.status == HTTPStatus.OK
            and response.read(len(_HEALTH_RESPONSE) + 1) == _HEALTH_RESPONSE
        )
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        if connection is not None:
            connection.close()


class PolicyServerSupervisor:
    """State-file supervisor with process-group cleanup and PID identity checks."""

    def __init__(self, root: Path, name: str) -> None:
        self.root = root / "policy-server"
        self.state_path = self.root / f"{name}.json"
        self.log_path = self.root / f"{name}.log"
        self._lock = threading.RLock()
        self._process: subprocess.Popen[Any] | None = None
        self._owned_identity: tuple[int, str] | None = None

    @staticmethod
    def _state_identity(state: Mapping[str, Any]) -> tuple[int, str] | None:
        pid = state.get("pid")
        ticks = state.get("start_ticks")
        if not isinstance(pid, int) or not isinstance(ticks, str):
            return None
        return pid, ticks

    def _read_state(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _clear(self) -> None:
        self.state_path.unlink(missing_ok=True)


    @staticmethod
    def _redact_tail(tail: str) -> str:
        tail = _TOKEN_ASSIGNMENT_RE.sub(r"[REDACTED]\1[REDACTED]", tail)
        tail = _BEARER_TOKEN_RE.sub(r"\1[REDACTED]", tail)
        return _HUGGING_FACE_TOKEN_RE.sub("[REDACTED]", tail)

    def _log_tail(self) -> str:
        try:
            with self.log_path.open("rb") as log:
                log.seek(0, os.SEEK_END)
                tail_start = max(0, log.tell() - _TAIL_BYTES)
                if tail_start:
                    log.seek(tail_start - 1, os.SEEK_SET)
                    starts_midline = log.read(1) != b"\n"
                else:
                    starts_midline = False
                log.seek(tail_start, os.SEEK_SET)
                tail = log.read(_TAIL_BYTES)
                if starts_midline:
                    _, separator, tail = tail.partition(b"\n")
                    if not separator:
                        tail = b""
                redacted = self._redact_tail(tail.decode("utf-8", errors="replace"))
                return redacted[-_TAIL_BYTES:]
        except OSError:
            return "no server log was created"

    def _owned_exit_status(self, identity: tuple[int, str]) -> int | None:
        if self._owned_identity != identity or self._process is None:
            return None
        return self._process.poll()

    def _exit_response(
        self, identity: tuple[int, str], exit_status: int | None
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "state": "exited",
            "pid": identity[0],
            "log_path": str(self.log_path),
            "detail": self._log_tail(),
        }
        if exit_status is not None:
            response["exit_status"] = exit_status
        return response

    def _status_locked(self) -> dict[str, Any]:
        state = self._read_state()
        if state is None:
            return {"state": "absent", "log_path": str(self.log_path)}
        identity = self._state_identity(state)
        if identity is None:
            self._clear()
            return {
                "state": "absent",
                "detail": "malformed process state removed",
                "log_path": str(self.log_path),
            }
        exit_status = self._owned_exit_status(identity)
        if exit_status is not None:
            return self._exit_response(identity, exit_status)
        info = _process_info(identity[0])
        if info is None or info.state == "Z":
            return self._exit_response(identity, None)
        if info.start_ticks != identity[1]:
            self._clear()
            return {
                "state": "absent",
                "detail": "stale process identity removed",
                "log_path": str(self.log_path),
            }
        host, port = state.get("host"), state.get("port")
        if not isinstance(host, str) or not isinstance(port, int):
            self._clear()
            return {
                "state": "absent",
                "detail": "malformed process state removed",
                "log_path": str(self.log_path),
            }
        if _health_ready(host, port):
            return {
                "state": "healthy",
                "pid": identity[0],
                "host": host,
                "port": port,
                "log_path": str(self.log_path),
            }
        deadline = state.get("ready_deadline", 0.0)
        if isinstance(deadline, (int, float)) and time.monotonic() < deadline:
            return {
                "state": "starting",
                "pid": identity[0],
                "host": host,
                "port": port,
                "log_path": str(self.log_path),
            }
        return {
            "state": "unhealthy",
            "pid": identity[0],
            "host": host,
            "port": port,
            "log_path": str(self.log_path),
            "detail": self._log_tail(),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def _poll_failure_locked(self) -> _PolicyServerExit | None:
        state = self._read_state()
        if state is None:
            return None
        identity = self._state_identity(state)
        if identity is None:
            self._clear()
            return None
        exit_status = self._owned_exit_status(identity)
        if exit_status is not None:
            return _PolicyServerExit(
                identity, identity[0], exit_status, str(self.log_path), self._log_tail()
            )
        info = _process_info(identity[0])
        if info is None or info.state == "Z":
            return _PolicyServerExit(
                identity, identity[0], None, str(self.log_path), self._log_tail()
            )
        if info.start_ticks != identity[1]:
            self._clear()
        return None

    def poll_failure(self) -> _PolicyServerExit | None:
        """Return an unexpected owned or recovered server exit, if one is observed."""
        with self._lock:
            return self._poll_failure_locked()

    def _has_state(self) -> bool:
        with self._lock:
            return self._read_state() is not None

    def _startup_exit_error(self, exit_status: int | None) -> RuntimeError:
        status = "unavailable" if exit_status is None else str(exit_status)
        return RuntimeError(
            "policy server exited before readiness; "
            f"exit_status={status}; log_path={self.log_path}; tail={self._log_tail()}"
        )

    def serves_other(self, model: Path | None, host: str, port: int) -> bool:
        """Whether a live server was started for a different model or address.

        State written before `model` was recorded is treated as matching so a
        module upgrade does not restart a healthy server.
        """
        with self._lock:
            if not self.state_path.is_file():
                return False
            state = self._read_state() or {}
            if self._status_locked()["state"] not in ("starting", "healthy"):
                return False
            recorded = state.get("model")
            model_changed = (
                model is not None and isinstance(recorded, str) and recorded != str(model)
            )
            return model_changed or (state.get("host"), state.get("port")) != (host, port)

    def start(
        self,
        command: Sequence[str],
        host: str,
        port: int,
        readiness_timeout: float = 30.0,
        environment: Mapping[str, str] | None = None,
        model: Path | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            current = self._status_locked()
            if current["state"] in ("starting", "healthy"):
                if not self.serves_other(model, host, port):
                    return {**current, "idempotent": True}
                cleanup = self._stop_locked()
                if not cleanup.get("stopped", False) and self._read_state() is not None:
                    raise RuntimeError(
                        "cannot safely stop policy server serving a different model; "
                        f"detail={cleanup.get('detail', 'unknown')}"
                    )
            if current["state"] in ("unhealthy", "exited"):
                cleanup = self._stop_locked()
                if not cleanup.get("stopped", False) and self._read_state() is not None:
                    raise RuntimeError(
                        "cannot safely stop previous policy server; "
                        f"detail={cleanup.get('detail', 'unknown')}"
                    )
            if not command:
                raise ValueError("policy server launch command is empty")
            self.root.mkdir(parents=True, exist_ok=True)
            process_environment = os.environ.copy()
            if environment is not None:
                process_environment.update(environment)
            with self.log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    list(command),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    env=process_environment,
                )
            info = _process_info(process.pid)
            if info is None:
                exit_status = process.poll()
                if exit_status is not None:
                    raise self._startup_exit_error(exit_status)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise RuntimeError(
                    "cannot record policy server process identity; "
                    f"log_path={self.log_path}; tail={self._log_tail()}"
                )
            identity = (process.pid, info.start_ticks)
            self._process = process
            self._owned_identity = identity
            state = {
                "pid": process.pid,
                "start_ticks": info.start_ticks,
                "host": host,
                "port": port,
                **({} if model is None else {"model": str(model)}),
                "ready_deadline": time.monotonic() + readiness_timeout,
            }
            self.state_path.write_text(
                json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
            )
            while time.monotonic() < state["ready_deadline"]:
                exit_status = process.poll()
                if exit_status is not None:
                    self._clear()
                    raise self._startup_exit_error(exit_status)
                status = self._status_locked()
                if status["state"] == "healthy":
                    return {**status, "idempotent": False}
                exit_status = process.poll()
                if exit_status is not None:
                    self._clear()
                    raise self._startup_exit_error(exit_status)
                time.sleep(0.1)
            status = self._status_locked()
            exit_status = process.poll()
            if exit_status is not None:
                self._clear()
                raise self._startup_exit_error(exit_status)
            detail = self._log_tail()
            cleanup = self._stop_locked(
                grace_seconds=min(5.0, max(0.1, readiness_timeout))
            )
            raise RuntimeError(
                f"policy server did not become ready; state={status['state']}; "
                f"cleanup={cleanup}; log_path={self.log_path}; tail={detail}"
            )

    def _stop_locked(self, grace_seconds: float = 5.0) -> dict[str, Any]:
        state = self._read_state()
        if state is None:
            return {"state": "absent", "stopped": False, "log_path": str(self.log_path)}
        identity = self._state_identity(state)
        if identity is None:
            self._clear()
            return {
                "state": "absent",
                "stopped": False,
                "detail": "malformed process state removed",
                "log_path": str(self.log_path),
            }
        exit_status = self._owned_exit_status(identity)
        if exit_status is not None:
            self._clear()
            return {
                "state": "absent",
                "stopped": True,
                "forced": False,
                "log_path": str(self.log_path),
            }
        info = _process_info(identity[0])
        if info is None or info.start_ticks != identity[1]:
            self._clear()
            return {
                "state": "absent",
                "stopped": False,
                "detail": "stale process identity removed",
                "log_path": str(self.log_path),
            }
        owned = self._owned_identity == identity and self._process is not None
        pidfd: int | None = None
        if owned:
            # An unreaped Popen child cannot have its PID recycled, so its
            # process group remains safe to clean up as a unit.
            try:
                os.killpg(identity[0], signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            if info.state == "Z":
                self._clear()
                return {
                    "state": "absent",
                    "stopped": True,
                    "forced": False,
                    "log_path": str(self.log_path),
                }
            pidfd = _open_identity_pidfd(identity)
            if pidfd is None:
                current = _process_info(identity[0])
                if (
                    current is None
                    or current.start_ticks != identity[1]
                    or current.state == "Z"
                ):
                    self._clear()
                    return {
                        "state": "absent",
                        "stopped": True,
                        "forced": False,
                        "log_path": str(self.log_path),
                    }
                return {
                    "state": "unhealthy",
                    "stopped": False,
                    "detail": "cannot safely signal recovered policy server",
                    "log_path": str(self.log_path),
                }
            if not _send_pidfd_signal(pidfd, signal.SIGTERM):
                os.close(pidfd)
                current = _process_info(identity[0])
                if (
                    current is None
                    or current.start_ticks != identity[1]
                    or current.state == "Z"
                ):
                    self._clear()
                    return {
                        "state": "absent",
                        "stopped": True,
                        "forced": False,
                        "log_path": str(self.log_path),
                    }
                return {
                    "state": "unhealthy",
                    "stopped": False,
                    "detail": "cannot safely signal recovered policy server",
                    "log_path": str(self.log_path),
                }
        try:
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline:
                if self._owned_exit_status(identity) is not None:
                    break
                current = _process_info(identity[0])
                if (
                    current is None
                    or current.start_ticks != identity[1]
                    or current.state == "Z"
                ):
                    break
                time.sleep(0.05)
            current = _process_info(identity[0])
            forced = (
                self._owned_exit_status(identity) is None
                and current is not None
                and current.start_ticks == identity[1]
                and current.state != "Z"
            )
            if forced:
                if owned:
                    try:
                        os.killpg(identity[0], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    assert self._process is not None
                    self._process.wait()
                else:
                    assert pidfd is not None
                    _send_pidfd_signal(pidfd, signal.SIGKILL)
            self._clear()
            return {
                "state": "absent",
                "stopped": True,
                "forced": forced,
                "log_path": str(self.log_path),
            }
        finally:
            if pidfd is not None:
                os.close(pidfd)

    def stop(self, grace_seconds: float = 5.0) -> dict[str, Any]:
        with self._lock:
            return self._stop_locked(grace_seconds)


def runtime_command(runtime_root: Path, model: Path, host: str, port: int) -> list[str]:
    """Start the lightweight provisioner; it enters the pinned upstream environment."""
    return [
        sys.executable,
        str(runtime_root / "launcher.py"),
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
    ]
