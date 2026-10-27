"""Module-owned subprocess supervision for the optional policy-server runtime."""

from __future__ import annotations

import http.client
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
from typing import Any, Mapping, NoReturn, Sequence

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
class _PolicyServerExit:
    process: subprocess.Popen[Any]
    exit_status: int
    log_path: str
    detail: str


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


def _killpg(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


class PolicyServerSupervisor:
    """In-memory supervisor: the server lives exactly as long as this object's owner."""

    def __init__(self, root: Path, name: str) -> None:
        self.root = root / "policy-server"
        self.log_path = self.root / f"{name}.log"
        self._lock = threading.RLock()
        self._process: subprocess.Popen[Any] | None = None
        self._host = ""
        self._port = 0
        self._model = ""
        self._ready_deadline = 0.0

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

    def _status_locked(self) -> dict[str, Any]:
        process = self._process
        if process is None:
            return {"state": "absent", "log_path": str(self.log_path)}
        exit_status = process.poll()
        if exit_status is not None:
            return {
                "state": "exited",
                "pid": process.pid,
                "exit_status": exit_status,
                "log_path": str(self.log_path),
                "detail": self._log_tail(),
            }
        running = {
            "pid": process.pid,
            "host": self._host,
            "port": self._port,
            # What this process was launched with; the configured model_path may differ
            # while viam-server is still downloading a package for a pending config.
            "model": self._model,
            "log_path": str(self.log_path),
        }
        if _health_ready(self._host, self._port):
            return {"state": "healthy", **running}
        if time.monotonic() < self._ready_deadline:
            return {"state": "starting", **running}
        return {"state": "unhealthy", **running, "detail": self._log_tail()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def running(self) -> bool:
        with self._lock:
            return self._process is not None

    def poll_failure(self) -> _PolicyServerExit | None:
        """Return the owned server's exit, if it has exited without being stopped."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is None:
                return None
            return _PolicyServerExit(
                process, process.returncode, str(self.log_path), self._log_tail()
            )

    def _fail_startup(self, process: subprocess.Popen[Any], exit_status: int) -> NoReturn:
        with self._lock:
            if self._process is process:
                self._process = None
        raise RuntimeError(
            "policy server exited before readiness; "
            f"exit_status={exit_status}; log_path={self.log_path}; tail={self._log_tail()}"
        )

    def start(
        self,
        command: Sequence[str],
        host: str,
        port: int,
        readiness_timeout: float = 30.0,
        environment: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            current = self._status_locked()
            if current["state"] in ("starting", "healthy"):
                return {**current, "idempotent": True}
            if self._process is not None:
                self._stop_locked()
            if not command:
                raise ValueError("policy server launch command is empty")
            if _health_ready(host, port):
                # Nothing of ours is running, so this is a server left behind by a
                # crashed module or an older release; adopting it would serve its weights.
                raise RuntimeError(
                    f"{host}:{port} already serves a policy server this component does "
                    "not own; stop it and retry"
                )
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
            self._process, self._host, self._port = process, host, port
            self._model = _model_from_command(command)
            self._ready_deadline = time.monotonic() + readiness_timeout
        # Poll unlocked so stop()/status() are not blocked for the whole model load.
        while time.monotonic() < self._ready_deadline:
            with self._lock:
                if self._process is not process:
                    raise RuntimeError("policy server was stopped during startup")
            exit_status = process.poll()
            if exit_status is not None:
                self._fail_startup(process, exit_status)
            if _health_ready(host, port):
                return {
                    "state": "healthy",
                    "pid": process.pid,
                    "host": host,
                    "port": port,
                    "model": self._model,
                    "log_path": str(self.log_path),
                    "idempotent": False,
                }
            time.sleep(0.1)
        exit_status = process.poll()
        if exit_status is not None:
            self._fail_startup(process, exit_status)
        detail = self._log_tail()
        with self._lock:
            cleanup = (
                self._stop_locked(grace_seconds=min(5.0, max(0.1, readiness_timeout)))
                if self._process is process
                else {"state": "absent", "stopped": False}
            )
        raise RuntimeError(
            "policy server did not become ready; "
            f"cleanup={cleanup}; log_path={self.log_path}; tail={detail}"
        )

    def _stop_locked(self, grace_seconds: float = 5.0) -> dict[str, Any]:
        process = self._process
        if process is None:
            return {"state": "absent", "stopped": False, "log_path": str(self.log_path)}
        self._process = None
        forced = False
        if process.poll() is None:
            # The launcher execs the venv python in the same session, so the
            # group signal is what reaches the GPU process.
            _killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(grace_seconds)
            except subprocess.TimeoutExpired:
                forced = True
                _killpg(process.pid, signal.SIGKILL)
                process.wait()
        return {
            "state": "absent",
            "stopped": True,
            "forced": forced,
            "log_path": str(self.log_path),
        }

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


def _model_from_command(command: Sequence[str]) -> str:
    """Inverse of runtime_command for the `--model` argument; empty for other commands."""
    args = list(command)
    return args[args.index("--model") + 1] if "--model" in args[:-1] else ""
