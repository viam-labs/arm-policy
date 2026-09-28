"""Module-owned subprocess supervision for the optional policy-server runtime."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def _process_start_ticks(pid: int) -> str | None:
    try:
        # Field 22 is the Linux process start time.  It prevents PID-reuse kills.
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError):
        return None


def _tcp_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


class PolicyServerSupervisor:
    """State-file supervisor with process-group cleanup and PID identity checks."""

    def __init__(self, root: Path, name: str) -> None:
        self.root = root / "policy-server"
        self.state_path = self.root / f"{name}.json"
        self.log_path = self.root / f"{name}.log"

    def _read_state(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _clear(self) -> None:
        self.state_path.unlink(missing_ok=True)

    def _identity(self, state: dict[str, Any]) -> bool:
        pid = state.get("pid")
        ticks = state.get("start_ticks")
        return isinstance(pid, int) and isinstance(ticks, str) and _process_start_ticks(pid) == ticks

    def status(self) -> dict[str, Any]:
        state = self._read_state()
        if state is None:
            return {"state": "absent", "log_path": str(self.log_path)}
        pid, ticks = state.get("pid"), state.get("start_ticks")
        actual_ticks = _process_start_ticks(pid) if isinstance(pid, int) else None
        if actual_ticks is None:
            return {"state": "exited", "pid": pid, "log_path": str(self.log_path), "detail": self._log_tail()}
        if not isinstance(ticks, str) or actual_ticks != ticks:
            self._clear()
            return {"state": "absent", "detail": "stale process identity removed", "log_path": str(self.log_path)}
        assert isinstance(pid, int)
        pid = state["pid"]
        host, port = state["host"], state["port"]
        if _tcp_ready(host, port):
            return {"state": "healthy", "pid": pid, "host": host, "port": port, "log_path": str(self.log_path)}
        deadline = state.get("ready_deadline", 0.0)
        if isinstance(deadline, (int, float)) and time.monotonic() < deadline:
            return {"state": "starting", "pid": pid, "host": host, "port": port, "log_path": str(self.log_path)}
        return {"state": "unhealthy", "pid": pid, "host": host, "port": port, "log_path": str(self.log_path), "detail": self._log_tail()}

    def _log_tail(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            return "no server log was created"

    def start(self, command: Sequence[str], host: str, port: int, readiness_timeout: float = 30.0) -> dict[str, Any]:
        current = self.status()
        if current["state"] in ("starting", "healthy"):
            return {**current, "idempotent": True}
        if current["state"] in ("unhealthy", "exited"):
            self.stop()
        if not command:
            raise ValueError("policy server launch command is empty")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        ticks = _process_start_ticks(process.pid)
        if ticks is None:
            process.kill()
            raise RuntimeError("cannot record policy server process identity")
        state = {"pid": process.pid, "start_ticks": ticks, "host": host, "port": port, "ready_deadline": time.monotonic() + readiness_timeout}
        self.state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        while time.monotonic() < state["ready_deadline"]:
            status = self.status()
            if status["state"] == "healthy":
                return {**status, "idempotent": False}
            if _process_start_ticks(process.pid) != ticks:
                self._clear()
                raise RuntimeError(f"policy server exited before readiness; log: {self._log_tail()}")
            time.sleep(0.1)
        status = self.status()
        detail = self._log_tail()
        cleanup = self.stop(grace_seconds=min(5.0, max(0.1, readiness_timeout)))
        raise RuntimeError(
            f"policy server did not become ready; state={status['state']}; "
            f"cleanup={cleanup}; log: {detail}"
        )

    def stop(self, grace_seconds: float = 5.0) -> dict[str, Any]:
        state = self._read_state()
        if state is None:
            return {"state": "absent", "stopped": False, "log_path": str(self.log_path)}
        if not self._identity(state):
            self._clear()
            return {"state": "absent", "stopped": False, "detail": "stale process identity removed", "log_path": str(self.log_path)}
        pid = state["pid"]
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and _process_start_ticks(pid) == state["start_ticks"]:
            time.sleep(0.05)
        forced = _process_start_ticks(pid) == state["start_ticks"]
        if forced:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._clear()
        return {"state": "absent", "stopped": True, "forced": forced, "log_path": str(self.log_path)}


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
