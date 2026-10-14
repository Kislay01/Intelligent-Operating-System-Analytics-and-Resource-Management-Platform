"""Actuator: the only component allowed to change process settings.

Safety rules:
* dry-run by default (``KS_DRY_RUN=0`` or ``dry_run=False`` to apply)
* never touches PID 0/1/2, protected names or processes of other users
  (unless running as root)
* records every action (applied, simulated or refused) as JSON lines and keeps
  the previous value so actions can be rolled back
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from ..config import settings
from .policy import PROTECTED_NAMES, SYSTEM_PIDS, Action


@dataclass
class ActionResult:
    ts: float
    pid: int
    name: str
    kind: str
    previous: object
    target: object
    status: str  # applied | dry-run | refused | failed | advisory
    message: str


class Actuator:
    def __init__(self, dry_run: bool | None = None, log_path: Path | None = None, allow_pids: set[int] | None = None):
        self.dry_run = settings.dry_run if dry_run is None else dry_run
        self.log_path = log_path or settings.action_log_path
        self.allow_pids = allow_pids
        self.history: list[ActionResult] = []
        self._is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        try:
            self._me = psutil.Process().username()
        except psutil.Error:
            self._me = None

    # -------------------------------------------------------------- checks
    def _permitted(self, proc: psutil.Process) -> tuple[bool, str]:
        if proc.pid in SYSTEM_PIDS:
            return False, "system process"
        if self.allow_pids is not None and proc.pid not in self.allow_pids:
            return False, "not in allow-list"
        try:
            if proc.name() in PROTECTED_NAMES:
                return False, "protected process"
            if not self._is_root and proc.username() != self._me:
                return False, "owned by another user"
        except psutil.Error as exc:
            return False, str(exc)
        return True, ""

    # ------------------------------------------------------------- execute
    def execute(self, action: Action) -> ActionResult:
        def result(status: str, message: str, previous=None) -> ActionResult:
            r = ActionResult(time.time(), action.pid, action.name, action.kind,
                             previous if previous is not None else action.current, action.target, status, message)
            self._record(r)
            return r

        if not action.applicable:
            return result("advisory", "Recommendation only (needs root or a later phase).")
        try:
            proc = psutil.Process(action.pid)
        except psutil.NoSuchProcess:
            return result("failed", "process no longer exists")
        ok, why = self._permitted(proc)
        if not ok:
            return result("refused", why)

        try:
            if action.kind == "nice":
                prev = proc.nice()
                if int(action.target) < prev and not self._is_root:
                    return result("refused", "lowering nice requires root (CAP_SYS_NICE)", prev)
                if self.dry_run:
                    return result("dry-run", f"would set nice {prev} -> {action.target}", prev)
                proc.nice(int(action.target))
                return result("applied", f"nice {prev} -> {action.target}", prev)
            if action.kind == "affinity":
                if not hasattr(proc, "cpu_affinity"):
                    return result("failed", "CPU affinity not supported on this platform")
                prev = proc.cpu_affinity()
                if self.dry_run:
                    return result("dry-run", f"would set affinity {prev} -> {action.target}", prev)
                proc.cpu_affinity(list(action.target))
                return result("applied", f"affinity {prev} -> {action.target}", prev)
            if action.kind == "ionice":
                if not hasattr(proc, "ionice"):
                    return result("failed", "ionice not supported on this platform")
                prev_io = proc.ionice()
                prev = f"{getattr(prev_io.ioclass, 'name', prev_io.ioclass)}/{prev_io.value}"
                level = int(str(action.target).split("/")[-1])
                if self.dry_run:
                    return result("dry-run", f"would set ionice {prev} -> best-effort/{level}", prev)
                proc.ionice(psutil.IOPRIO_CLASS_BE, value=level)
                return result("applied", f"ionice {prev} -> best-effort/{level}", prev)
        except psutil.AccessDenied:
            return result("refused", "access denied by the kernel")
        except (psutil.Error, OSError, ValueError) as exc:
            return result("failed", str(exc))
        return result("advisory", f"no actuator for '{action.kind}'")

    def execute_all(self, actions: list[Action]) -> list[ActionResult]:
        return [self.execute(a) for a in actions]

    def rollback(self, results: list[ActionResult]) -> list[ActionResult]:
        """Restore the previous value of every applied action, newest first."""
        undo = []
        for r in reversed(results):
            if r.status != "applied":
                continue
            undo.append(self.execute(Action(r.pid, r.name, r.kind, r.target, r.previous, "rollback")))
        return undo

    # ------------------------------------------------------------- logging
    def _record(self, r: ActionResult) -> None:
        self.history.append(r)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(asdict(r), default=str) + "\n")
        except OSError:
            pass

    def read_log(self, limit: int = 200) -> list[dict]:
        try:
            lines = self.log_path.read_text().splitlines()[-limit:]
        except OSError:
            return []
        return [json.loads(l) for l in lines if l.strip()]
