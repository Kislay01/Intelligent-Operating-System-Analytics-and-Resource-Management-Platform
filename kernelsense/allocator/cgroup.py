"""cgroup controller for resource limits (v2 preferred, v1 fallback).

Control groups are the kernel's mechanism for limiting and weighting resources
for a set of processes:

=================  ==============================  ================================
Purpose            cgroup v2 file                  cgroup v1 equivalent
=================  ==============================  ================================
CPU share          cpu.weight (1..10000, def 100)  cpu/cpu.shares (2..262144, 1024)
CPU hard cap       cpu.max ("quota period")        cpu/cpu.cfs_quota_us, cfs_period_us
Memory throttling  memory.high                     memory/memory.soft_limit_in_bytes
Memory hard limit  memory.max                      memory/memory.limit_in_bytes
=================  ==============================  ================================

All groups are created under a ``kernelsense`` parent so the platform never
touches anything it did not create, and ``remove()`` moves processes back to
the parent before deleting the group.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/sys/fs/cgroup")


@dataclass
class CgroupInfo:
    version: int
    mount: Path
    controllers: list[str]
    writable: bool


def detect() -> CgroupInfo | None:
    """Find a usable cgroup hierarchy: unified v2 with controllers, else v1 cpu/memory."""
    for mount in (ROOT, ROOT / "unified"):
        ctrl = mount / "cgroup.controllers"
        if ctrl.exists():
            controllers = ctrl.read_text().split()
            if {"cpu", "memory"} & set(controllers):
                return CgroupInfo(2, mount, controllers, os.access(mount, os.W_OK))
    if (ROOT / "cpu" / "cpu.shares").exists():
        controllers = [c for c in ("cpu", "memory", "cpuset", "blkio") if (ROOT / c).exists()]
        return CgroupInfo(1, ROOT, controllers, os.access(ROOT / "cpu", os.W_OK))
    return None


class Cgroup:
    """A managed cgroup named ``kernelsense/<name>``."""

    def __init__(self, name: str, info: CgroupInfo | None = None, dry_run: bool = False) -> None:
        self.info = info or detect()
        if self.info is None:
            raise RuntimeError("no usable cgroup hierarchy found")
        self.name = name
        self.dry_run = dry_run
        self.log: list[str] = []

    # ------------------------------------------------------------------ paths
    def _dir(self, controller: str | None = None) -> Path:
        if self.info.version == 2:
            return self.info.mount / "kernelsense" / self.name
        return self.info.mount / (controller or "cpu") / "kernelsense" / self.name

    def _controllers_v1(self) -> list[str]:
        return [c for c in ("cpu", "memory") if c in self.info.controllers]

    def _write(self, path: Path, value: str) -> None:
        self.log.append(f"{path} <- {value}")
        if self.dry_run:
            return
        path.write_text(value)

    # -------------------------------------------------------------- lifecycle
    def create(self) -> "Cgroup":
        if self.info.version == 2:
            parent = self.info.mount / "kernelsense"
            if not self.dry_run:
                parent.mkdir(exist_ok=True)
                # enable controllers for children of the parent
                try:
                    wanted = [c for c in ("cpu", "memory") if c in self.info.controllers]
                    (self.info.mount / "cgroup.subtree_control").write_text(" ".join(f"+{c}" for c in wanted))
                    (parent / "cgroup.subtree_control").write_text(" ".join(f"+{c}" for c in wanted))
                except OSError:
                    pass
                self._dir().mkdir(exist_ok=True)
        else:
            for c in self._controllers_v1():
                if not self.dry_run:
                    self._dir(c).mkdir(parents=True, exist_ok=True)
        self.log.append(f"created {self._dir()}")
        return self

    def add(self, pid: int) -> None:
        if self.info.version == 2:
            self._write(self._dir() / "cgroup.procs", str(pid))
        else:
            for c in self._controllers_v1():
                self._write(self._dir(c) / "cgroup.procs", str(pid))

    def pids(self) -> list[int]:
        f = self._dir() / "cgroup.procs" if self.info.version == 2 else self._dir("cpu") / "cgroup.procs"
        try:
            return [int(x) for x in f.read_text().split()]
        except OSError:
            return []

    def remove(self) -> None:
        """Move member processes back to the root group and delete the cgroup."""
        if self.info.version == 2:
            dirs = [(self._dir(), self.info.mount / "cgroup.procs")]
        else:
            dirs = [(self._dir(c), self.info.mount / c / "cgroup.procs") for c in self._controllers_v1()]
        for d, parent_procs in dirs:
            try:
                for pid in (d / "cgroup.procs").read_text().split():
                    try:
                        self._write(parent_procs, pid)
                    except OSError:
                        pass
                if not self.dry_run:
                    d.rmdir()
            except OSError:
                continue

    # --------------------------------------------------------------- controls
    def set_cpu_weight(self, weight: int) -> None:
        """Relative CPU share; 100 = default. Uses cpu.shares (x10.24) on v1."""
        weight = max(1, min(10000, int(weight)))
        if self.info.version == 2:
            self._write(self._dir() / "cpu.weight", str(weight))
        else:
            self._write(self._dir("cpu") / "cpu.shares", str(max(2, int(weight * 1024 / 100))))

    def set_cpu_max(self, percent_of_one_cpu: float | None, period_us: int = 100_000) -> None:
        """Hard cap: e.g. 50 = half of one CPU, 200 = two CPUs, None = unlimited."""
        if self.info.version == 2:
            val = "max" if percent_of_one_cpu is None else f"{int(period_us * percent_of_one_cpu / 100)} {period_us}"
            self._write(self._dir() / "cpu.max", f"{val}" if val != "max" else f"max {period_us}")
        else:
            self._write(self._dir("cpu") / "cpu.cfs_period_us", str(period_us))
            quota = -1 if percent_of_one_cpu is None else int(period_us * percent_of_one_cpu / 100)
            self._write(self._dir("cpu") / "cpu.cfs_quota_us", str(quota))

    def set_memory_high(self, bytes_: int | None) -> None:
        if self.info.version == 2:
            self._write(self._dir() / "memory.high", "max" if bytes_ is None else str(int(bytes_)))
        elif "memory" in self.info.controllers:
            self._write(self._dir("memory") / "memory.soft_limit_in_bytes", str(-1 if bytes_ is None else int(bytes_)))

    def set_memory_max(self, bytes_: int | None) -> None:
        if self.info.version == 2:
            self._write(self._dir() / "memory.max", "max" if bytes_ is None else str(int(bytes_)))
        elif "memory" in self.info.controllers:
            self._write(self._dir("memory") / "memory.limit_in_bytes", str(-1 if bytes_ is None else int(bytes_)))

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        out: dict = {"version": self.info.version, "pids": self.pids()}
        try:
            if self.info.version == 2:
                for line in (self._dir() / "cpu.stat").read_text().splitlines():
                    k, v = line.split()
                    if k in ("usage_usec", "nr_throttled", "throttled_usec"):
                        out[k] = int(v)
                cur = self._dir() / "memory.current"
                if cur.exists():
                    out["memory_bytes"] = int(cur.read_text())
            else:
                out["usage_usec"] = int((ROOT / "cpuacct" / "kernelsense" / self.name / "cpuacct.usage").read_text()) // 1000 \
                    if (ROOT / "cpuacct" / "kernelsense" / self.name / "cpuacct.usage").exists() else None
                stat = self._dir("cpu") / "cpu.stat"
                if stat.exists():
                    for line in stat.read_text().splitlines():
                        k, v = line.split()
                        if k in ("nr_throttled", "throttled_time"):
                            out[k] = int(v)
                mem = self._dir("memory") / "memory.usage_in_bytes"
                if mem.exists():
                    out["memory_bytes"] = int(mem.read_text())
        except (OSError, ValueError):
            pass
        return out
